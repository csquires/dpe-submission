"""detector classes for early stopping: monitor loss and signal when to stop training."""

import collections
import math


class NanDetector:
    """monitor consecutive non-finite loss values; trigger when count exceeds patience.

    interface: observe(step, loss) increments _count on non-finite; resets to 0 on finite.
    should_stop() returns (True, "nan_streak") if _count >= patience, else (False, None).
    """

    def __init__(self, patience: int = 3):
        self._count: int = 0
        self._patience: int = patience

    def observe(self, step: int, loss: float) -> None:
        """increment _count on non-finite loss; reset to 0 on finite."""
        if math.isfinite(loss):
            self._count = 0
        else:
            self._count += 1

    def should_stop(self) -> tuple[bool, str | None]:
        """return (True, 'nan_streak') if _count >= patience, else (False, None)."""
        if self._count >= self._patience:
            return (True, "nan_streak")
        return (False, None)


class PlateauDetector:
    """monitor loss plateau using sliding window and hybrid absolute+relative criterion.

    interface: observe(step, loss) skips non-finite; tracks _best and fills _buf.
    plateau criterion (when window fills): abs(mean - _best) < eps_abs + eps_rel * abs(_best).
    should_stop() returns (True, "plateau") if plateau count reaches patience, else (False, None).
    """

    def __init__(
        self, window: int = 50, eps_abs: float = 1e-4, eps_rel: float = 0.01, patience: int = 500
    ):
        self._buf: collections.deque[float] = collections.deque(maxlen=window)
        self._best: float | None = None
        self._count: int = 0
        self._eps_abs: float = eps_abs
        self._eps_rel: float = eps_rel
        self._patience: int = patience

    def observe(self, step: int, loss: float) -> None:
        """skip if non-finite; update _best to min(_best, loss); append to _buf; check plateau only when full."""
        if not math.isfinite(loss):
            return
        self._best = loss if self._best is None else min(self._best, loss)
        self._buf.append(loss)
        if len(self._buf) == self._buf.maxlen:
            self._check_plateau()

    def _check_plateau(self) -> None:
        """check if mean is within hybrid criterion; increment or reset _count."""
        mean = sum(self._buf) / len(self._buf)
        threshold = self._eps_abs + self._eps_rel * abs(self._best)
        if abs(mean - self._best) < threshold:
            self._count += 1
        else:
            self._count = 0

    def should_stop(self) -> tuple[bool, str | None]:
        """return (True, 'plateau') if _count >= patience, else (False, None)."""
        if self._count >= self._patience:
            return (True, "plateau")
        return (False, None)


class AnyDetector:
    """variadic composition: observe calls all detectors; should_stop returns first positive result.

    interface: observe(step, loss) delegates to each detector in _dets.
    should_stop() iterates _dets; returns (True, reason) for first detector signaling stop;
    if all return False, returns (False, None).
    """

    def __init__(self, *detectors):
        self._dets: tuple = detectors

    def observe(self, step: int, loss: float) -> None:
        """call observe(step, loss) on each detector."""
        for det in self._dets:
            det.observe(step, loss)

    def should_stop(self) -> tuple[bool, str | None]:
        """iterate _dets; return (True, reason) for first detector signaling stop; else (False, None)."""
        for det in self._dets:
            stop, reason = det.should_stop()
            if stop:
                return (True, reason)
        return (False, None)
