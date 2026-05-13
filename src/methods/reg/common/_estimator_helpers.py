"""helper for shared estimator constructor validation and slot assignment."""

from typing import Optional
import torch


def _validate_and_store_slots(
    estimator,
    *,
    input_dim: int,
    path,
    time,
    curve,
    integrator,
    expected_path_type: type,
    expected_curve_dim: int,
    device: Optional[str] = None,
) -> None:
    """validate four estimator slots and assign to instance.

    validates path type, curve dimension, time sampler callability, and
    integrator callability. resolves device (cuda if available, else cpu).
    assigns all validated slots plus device to the estimator instance.

    arguments:
      estimator: the estimator object receiving slot assignments.
      input_dim: feature dimension.
      path: a path dataclass instance.
      time: a time sampler callable.
      curve: a curve callable with dim attribute.
      integrator: an integrator callable.
      expected_path_type: type(s) that path must satisfy.
      expected_curve_dim: expected value of curve.dim.
      device: optional device string. if None, auto-selects cuda if available.

    raises:
      TypeError: if path is wrong type, curve.dim mismatches, time is not
                 callable, or integrator is not callable.

    side effects:
      assigns estimator.input_dim, estimator.path, estimator.time,
      estimator.curve, estimator.integrator, estimator.device.
    """
    # path type validation
    if not isinstance(path, expected_path_type):
        raise TypeError(
            f"path must be {expected_path_type.__name__}, "
            f"got {type(path).__name__}"
        )

    # curve dimension validation
    curve_dim = getattr(curve, "dim", None)
    if curve_dim != expected_curve_dim:
        raise TypeError(
            f"curve.dim must be {expected_curve_dim}, got {curve_dim}"
        )

    # time sampler callability
    if not callable(time):
        raise TypeError("time must be a callable TimeSampler (1D or 2D)")

    # integrator callability
    if not callable(integrator):
        raise TypeError("integrator must be a callable Integrator")

    # device resolution
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # assign slots
    estimator.input_dim = input_dim
    estimator.path = path
    estimator.time = time
    estimator.curve = curve
    estimator.integrator = integrator
    estimator.device = device
