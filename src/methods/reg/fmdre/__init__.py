"""FMDRE family: flow-matching DRE (S1 default, S2 alternate, triangular ELDR)."""
from .s1 import FMDRE
from .s2 import FMDRE_S2
from .tri import TriangularFMDRE

__all__ = ["FMDRE", "FMDRE_S2", "TriangularFMDRE"]
