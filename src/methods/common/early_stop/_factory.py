"""factory for early stopping (observe, should_stop) callable pairs."""
from typing import Callable

from ._detectors import AnyDetector, NanDetector, PlateauDetector


def _noop_observe(step: int, loss: float) -> None:
    """no-op observe function: accepts step and loss, returns None."""
    pass


def _noop_should_stop() -> tuple[bool, str | None]:
    """no-op should_stop function: returns (False, None)."""
    return (False, None)


def make_early_stopper(
    cfg: dict | None,
) -> tuple[Callable[[int, float], None], Callable[[], tuple[bool, str | None]]]:
    """
    convert early_stop config dict into (observe, should_stop) callable pair.

    when disabled or cfg is None, returns module-level no-op pair with minimal
    hot-path overhead. when enabled, instantiates detectors and wraps in
    AnyDetector.

    args:
        cfg: early_stop config dict, or None.
             if not None, expects keys "enabled" (bool), "nan" (dict),
             "plateau" (dict). missing "nan"/"plateau" dicts default to {}.
             missing "enabled" key defaults to False.

    returns:
        (observe, should_stop) tuple where:
            observe(step: int, loss: float) -> None: record a training step.
            should_stop() -> (bool, str | None): return (True, reason) to stop.

    behavior:
        - cfg is None or cfg["enabled"] is falsy: returns module-level
          _noop_observe and _noop_should_stop (singletons, picklable).
        - cfg["enabled"] is truthy: instantiates NanDetector with cfg.get("nan", {}),
          PlateauDetector with cfg.get("plateau", {}), wraps in AnyDetector,
          returns (any_detector.observe, any_detector.should_stop).

    edge cases:
        - cfg is None: return noop pair.
        - cfg["enabled"] missing or False: return noop pair.
        - cfg["nan"] missing: NanDetector({}) uses default patience=3.
        - cfg["plateau"] missing: PlateauDetector({}) uses defaults.
        - extra keys in cfg sub-dicts: detector constructors raise TypeError.
    """
    if cfg is None or not cfg.get("enabled", False):
        return _noop_observe, _noop_should_stop

    nan_cfg = cfg.get("nan", {})
    plateau_cfg = cfg.get("plateau", {})

    nan_detector = NanDetector(**nan_cfg)
    plateau_detector = PlateauDetector(**plateau_cfg)
    any_detector = AnyDetector(nan_detector, plateau_detector)

    return any_detector.observe, any_detector.should_stop
