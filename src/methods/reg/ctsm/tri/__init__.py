"""Triangular CTSM variants (barycentric V1, piecewise-SB V2, 2D V3)."""
from .v1 import TriangularCTSMV1
from .v2 import TriangularCTSMV2
from .v3 import TriangularCTSM2D

__all__ = ["TriangularCTSMV1", "TriangularCTSMV2", "TriangularCTSM2D"]
