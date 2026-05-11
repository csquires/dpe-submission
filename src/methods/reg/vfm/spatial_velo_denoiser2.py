"""DEPRECATED: this module has been renamed to src.methods.reg.vfm.

Imports from this path are preserved for one cycle but will emit a DeprecationWarning.
"""
import warnings

warnings.warn(
    "src.methods.reg.vfm.spatial_velo_denoiser2 has been renamed to "
    "src.methods.reg.vfm. SpatialVeloDenoiser is aliased as VFM. "
    "Update imports to `from src.methods.reg.vfm import VFM`.",
    DeprecationWarning,
    stacklevel=2,
)

# re-export for backward compat
from . import VFM as SpatialVeloDenoiser  # noqa: F401, E402
from src.models.common.mlp import MLP  # noqa: F401, E402
from src.models.flow.div_estimators import compute_divergence  # noqa: F401, E402
