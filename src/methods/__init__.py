"""Public API for density-ratio estimation methods.

Layout:
    src.methods.common  -- DRE, ELDR base classes shared across all methods.
    src.methods.cls     -- classification-based estimators (BCE / CE losses).
    src.methods.reg     -- regression / score-based estimators (L2-style losses).

Each family is its own sub-package, e.g. src.methods.reg.fmdre exports FMDRE
(s1), FMDRE_S2 (s2), and TriangularFMDRE (tri). Multi-variant triangular
families (CTSM, VFM) nest variants under `tri.v1`, `tri.v2`.
"""
from .common import DRE, ELDR
from .cls import (
    BDRE,
    MDRE,
    TriangularMDRE,
    TDRE,
    MultiHeadTDRE,
    TriangularTDRE,
    MultiHeadTriangularTDRE,
    TabularPluginDRE,
    SmoothedTabularPluginDRE,
)
from .reg import (
    TSM,
    TriangularTSM,
    CTSM,
    TriangularCTSMV1,
    TriangularCTSMV2,
    TriangularCTSM2D,
    VFM,
    TriangularVFMV1,
    TriangularVFMV2,
    TriangularVFM2D,
    FMDRE,
    FMDRE_S2,
    TriangularFMDRE,
)

SpatialVeloDenoiser = VFM

__all__ = [
    "DRE", "ELDR",
    "BDRE", "MDRE", "TDRE", "MultiHeadTDRE",
    "TSM", "CTSM", "TriangularTSM",
    "TriangularCTSMV1", "TriangularCTSMV2", "TriangularCTSM2D",
    "TriangularTDRE", "MultiHeadTriangularTDRE", "TriangularMDRE",
    "FMDRE", "FMDRE_S2", "TriangularFMDRE",
    "VFM", "SpatialVeloDenoiser",
    "TriangularVFMV1", "TriangularVFMV2", "TriangularVFM2D",
    "TabularPluginDRE", "SmoothedTabularPluginDRE",
]
