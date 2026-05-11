"""Triangular VFM variants (barycentric V1, piecewise-SB V2, 2D V3)."""
from .v1 import TriangularVFMV1
from .v2 import TriangularVFMV2
from .v3 import TriangularVFM2D

__all__ = ["TriangularVFMV1", "TriangularVFMV2", "TriangularVFM2D"]
