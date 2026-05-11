"""TDRE family: telescoping classifiers (single / multi-head, two-source / triangular)."""
from .s1 import TDRE
from .mh import MultiHeadTDRE
from .tri import TriangularTDRE
from .mh_tri import MultiHeadTriangularTDRE

__all__ = ["TDRE", "MultiHeadTDRE", "TriangularTDRE", "MultiHeadTriangularTDRE"]
