"""shared matplotlib rcParams + per-method color/marker registry for step4 line plots.

apply() sets a consistent rcParams baseline (font sizes, grid, legend frame).

color is driven by METHOD_FAMILIES (which methods share a hue) -- each head
method + its triangular/variant siblings gets one base hue, with variants
distinguished by a lightness sweep. panel layout is driven by METHOD_GROUPS
(which methods are drawn on which subplot); multiple families can coexist on
one panel so e.g. BDRE / MDRE / TDRE are visually separable. triangular
methods carry a `^` marker.

intended usage in any step4:

    from ex.utils.plot_style import apply, METHOD_GROUPS, style_for
    apply()
    for method in METHOD_GROUPS["cls"]:
        ax.plot(..., **style_for(method), label=method)
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.colors as mcolors


FONT_SIZE = 11
ERROR_BAND_ALPHA = 0.18
LINE_WIDTH = 1.8


# panel layout: which methods are drawn on which subplot.
METHOD_GROUPS: dict[str, list[str]] = {
    "vfm":      ["VFM", "TriangularVFM_V1", "TriangularVFM_V2", "TriangularVFM_V3"],
    "tsm_ctsm": ["TSM", "CTSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3"],
    "cls":      ["BDRE", "MDRE", "MultiHeadTDRE", "TriangularMDRE", "MultiHeadTriangularTDRE"],
    "fmdre":    ["FMDRE", "FMDRE_S2", "TriangularFMDRE"],
}

GROUP_LABEL: dict[str, str] = {
    "vfm":      "VFM family",
    "tsm_ctsm": "TSM / CTSM family",
    "cls":      "Classifier-based DRE",
    "fmdre":    "FMDRE family",
}

# color families: methods within a family share a base hue, ordered so the
# head method anchors at the dark end and variants sweep lighter.
METHOD_FAMILIES: dict[str, list[str]] = {
    "vfm":   ["VFM", "TriangularVFM_V1", "TriangularVFM_V2", "TriangularVFM_V3"],
    "tsm":   ["TSM"],
    "ctsm":  ["CTSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3"],
    "bdre":  ["BDRE"],
    "mdre":  ["MDRE", "TriangularMDRE"],
    "tdre":  ["MultiHeadTDRE", "MultiHeadTriangularTDRE"],
    "fmdre": ["FMDRE", "FMDRE_S2", "TriangularFMDRE"],
}

# base hue per family (matplotlib named color). picked so families that
# coexist on the same panel are visually distinct:
#   cls panel: green / olive / cyan
#   tsm_ctsm panel: orange / red
_FAMILY_BASE: dict[str, str] = {
    "vfm":   "tab:blue",
    "tsm":   "tab:orange",
    "ctsm":  "tab:red",
    "bdre":  "tab:green",
    "mdre":  "tab:olive",
    "tdre":  "tab:cyan",
    "fmdre": "tab:purple",
}


def _shade(base_color: str, lightness: float) -> tuple[float, float, float]:
    """blend base_color toward white (lightness=1) or black (lightness=0).

    lightness=0.5 returns the base color unchanged; <0.5 darkens, >0.5 lightens.
    """
    r, g, b = mcolors.to_rgb(base_color)
    if lightness >= 0.5:
        t = (lightness - 0.5) * 2.0   # 0..1
        return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)
    t = (0.5 - lightness) * 2.0       # 0..1
    return (r * (1 - t), g * (1 - t), b * (1 - t))


def _build_color_marker_tables():
    """build per-method color + marker maps from METHOD_FAMILIES.

    within each family, lightness sweeps over [0.20, 0.65] so the head method
    (idx 0) anchors at the dark end and the last variant sits a touch lighter
    than the base hue; the 0.65 ceiling keeps variants well-saturated so the
    family hue stays recognisable.
    a singleton family resolves to lightness 0.45 (almost the base hue).
    """
    colors: dict[str, tuple[float, float, float]] = {}
    markers: dict[str, str] = {}
    for family, methods in METHOD_FAMILIES.items():
        base = _FAMILY_BASE[family]
        n = len(methods)
        for idx, m in enumerate(methods):
            light = 0.20 + (0.45 * idx / max(1, n - 1)) if n > 1 else 0.45
            colors[m] = _shade(base, light)
            markers[m] = "^" if m.startswith("Triangular") or m.startswith("MultiHeadTriangular") else "o"
    return colors, markers


METHOD_COLORS, METHOD_MARKERS = _build_color_marker_tables()


def apply() -> None:
    """set rcParams shared across all line-plot step4s.

    keep this lean: only knobs that need to be CONSISTENT across step4s belong
    here. per-figure choices (figure size, axis log/linear) stay in each step4.
    """
    mpl.rcParams.update({
        "font.size":           FONT_SIZE,
        "axes.titlesize":      FONT_SIZE,
        "axes.labelsize":      FONT_SIZE,
        "xtick.labelsize":     FONT_SIZE - 1,
        "ytick.labelsize":     FONT_SIZE - 1,
        "legend.fontsize":     FONT_SIZE - 2,
        "axes.grid":           True,
        "grid.alpha":          0.3,
        "legend.frameon":      True,
        "legend.framealpha":   0.85,
        "lines.linewidth":     LINE_WIDTH,
        "lines.markersize":    5,
        "savefig.bbox":        "tight",
        "savefig.dpi":         300,
    })


def style_for(method: str) -> dict:
    """return kwargs to splat into ax.plot() for a given method.

    falls back to gray + circle marker for methods outside METHOD_FAMILIES.
    """
    color = METHOD_COLORS.get(method, (0.5, 0.5, 0.5))
    marker = METHOD_MARKERS.get(method, "o")
    return {"color": color, "marker": marker}


def group_of(method: str) -> str | None:
    """return the panel-group key (vfm / tsm_ctsm / cls / fmdre) for a method, or None."""
    for g, ms in METHOD_GROUPS.items():
        if method in ms:
            return g
    return None
