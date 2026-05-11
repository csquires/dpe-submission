"""Multiclass-classifier-based DRE and its triangular ELDR variant."""
from .s1 import MDRE
from .tri import TriangularMDRE

__all__ = ["MDRE", "TriangularMDRE"]
