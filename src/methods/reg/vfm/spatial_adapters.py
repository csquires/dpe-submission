"""DEPRECATED: `make_spatial_velo_denoiser` has been renamed to `make_vfm`.

Import the factory from the package root instead:

    from src.methods.reg.vfm import make_vfm

This shim re-exports the old name for one deprecation cycle.
"""
import warnings

warnings.warn(
    "src.methods.reg.vfm.spatial_adapters is deprecated; "
    "`make_spatial_velo_denoiser` is renamed to `make_vfm`. "
    "Use `from src.methods.reg.vfm import make_vfm`.",
    DeprecationWarning,
    stacklevel=2,
)

from . import make_vfm as make_spatial_velo_denoiser  # noqa: F401, E402
