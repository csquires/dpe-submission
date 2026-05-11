"""Tabular plug-in density-ratio estimators (empirical counts + Laplace smoothing)."""
from .s1 import TabularPluginDRE
from .smoothed import SmoothedTabularPluginDRE

__all__ = ["TabularPluginDRE", "SmoothedTabularPluginDRE"]
