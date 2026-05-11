"""DEPRECATED: this module has been renamed to src.density_ratio_estimation.vfm.

Imports from this path are preserved for one cycle but will emit a DeprecationWarning.
"""
import warnings

warnings.warn(
    "src.density_ratio_estimation.spatial_velo_denoiser2 has been renamed to "
    "src.density_ratio_estimation.vfm. SpatialVeloDenoiser is aliased as VFM. "
    "Update imports to `from src.density_ratio_estimation.vfm import VFM`.",
    DeprecationWarning,
    stacklevel=2,
)

# re-export for backward compat
from src.density_ratio_estimation.vfm import VFM as SpatialVeloDenoiser  # noqa: F401, E402
from src.models.common.mlp import MLP  # noqa: F401, E402
from src.models.flow.div_estimators import compute_divergence  # noqa: F401, E402
