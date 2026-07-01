"""shared faceted line-plot + standalone-legend helpers for step4 figures.

one thin-lined figure per facet (e.g. one per alpha / per test-set), every method
drawn on the same axes, colors/markers from ex.utils.plot_style. the legend is
emitted as its own figure so the per-facet panels stay uncluttered.

intended usage:

    from ex.utils.faceted_lines import plot_panels, plot_legend, order_methods
    methods = order_methods(mean.keys())
    plot_panels(x, facets, mean, se, xlabel=..., ylabel=..., out_dir=..., prefix=...)
    plot_legend(methods, out_dir, prefix)
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ex.utils.plot_style import apply as apply_style, style_for, ERROR_BAND_ALPHA


# canonical draw/legend order for the 18 eldr-estimation methods (family-grouped).
METHOD_ORDER = [
    "BDRE", "MDRE_15", "MultiHeadTDRE", "TriangularMDRE", "MultiHeadTriangularTDRE",
    "TSM", "TriangularTSM", "CTSM", "TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3",
    "VFM", "TriangularVFM_V1", "TriangularVFM_V2", "TriangularVFM_V3",
    "FMDRE", "FMDRE_S2", "TriangularFMDRE",
]

THIN_LW = 1.0
MARKER_SIZE = 3.0


def _alias(method: str) -> str:
    """map an experiment method name to its plot_style family key (MDRE_15 -> MDRE)."""
    return "MDRE" if method == "MDRE_15" else method


def order_methods(methods) -> list[str]:
    """sort a method iterable by METHOD_ORDER, unknown names appended alphabetically."""
    ms = list(methods)
    known = [m for m in METHOD_ORDER if m in ms]
    extra = sorted(m for m in ms if m not in METHOD_ORDER)
    return known + extra


def _style(method: str) -> dict:
    """color + marker for a method, with the MDRE_15 alias applied."""
    return style_for(_alias(method))


def plot_panels(x, facets, mean, se, *, xlabel, ylabel, out_dir, prefix,
                xscale="linear", yscale="linear", title_prefix="",
                shared_y=True) -> list[str]:
    """one thin-lined figure per facet; all methods on the same axes, +/- SE band.

    Args:
      x: 1D array of x-axis values (length L).
      facets: list of (facet_key, facet_label); one figure each.
      mean, se: dict method -> array [L, n_facets]; nan entries are skipped.
      xlabel, ylabel: axis labels. out_dir/prefix: output path + filename stem.
      xscale/yscale: matplotlib axis scales. title_prefix: prepended to facet_label.
      shared_y: if True, all facet figures share one y-range for comparability.

    a method whose column is all-nan in a facet is dropped from that figure; a
    facet with no finite data at all is skipped entirely.
    returns the ordered list of methods that appear in at least one facet.
    """
    apply_style()
    os.makedirs(out_dir, exist_ok=True)
    methods = order_methods(mean.keys())
    x = np.asarray(x)

    y_hi = 0.0
    y_lo = np.inf
    if shared_y:
        for m in methods:
            hi = mean[m] + se[m]
            lo = mean[m] - se[m]
            if np.isfinite(hi).any():
                y_hi = max(y_hi, float(np.nanmax(hi)))
            if np.isfinite(lo).any():
                y_lo = min(y_lo, float(np.nanmin(lo)))
        if yscale == "log":
            y_lo = max(y_lo, 1e-4)
        y_hi *= 1.08 if y_hi > 0 else 1.0

    drawn = []
    for fi, (fkey, flabel) in enumerate(facets):
        any_method = False
        fig, ax = plt.subplots(figsize=(5, 4))
        for m in methods:
            y = np.asarray(mean[m])[:, fi]
            if not np.isfinite(y).any():
                continue
            any_method = True
            if m not in drawn:
                drawn.append(m)
            e = np.asarray(se[m])[:, fi]
            kw = _style(m)
            ax.plot(x, y, label=m, linewidth=THIN_LW, markersize=MARKER_SIZE, **kw)
            band_lo = y - e
            band_hi = y + e
            if yscale == "log":
                band_lo = np.maximum(band_lo, y_lo if shared_y else 1e-4)
            ax.fill_between(x, band_lo, band_hi, color=kw["color"],
                            alpha=ERROR_BAND_ALPHA, linewidth=0)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix}{flabel}")
        if shared_y and y_hi > 0:
            ax.set_ylim(y_lo if yscale == "log" else min(0.0, y_lo), y_hi)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if not any_method:
            plt.close(fig)
            print(f"  skip {prefix}_{fkey}: no finite data")
            continue
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"{prefix}_{fkey}.{ext}"), dpi=150)
        plt.close(fig)
        print(f"  saved {prefix}_{fkey}.{{pdf,png}}")
    return order_methods(drawn)


def plot_legend(methods, out_dir, prefix, ncol=3) -> None:
    """emit a standalone legend figure (thin line + marker per method)."""
    apply_style()
    os.makedirs(out_dir, exist_ok=True)
    methods = order_methods(methods)
    handles = [
        Line2D([0], [0], linewidth=THIN_LW, markersize=MARKER_SIZE + 1,
               label=m, **_style(m))
        for m in methods
    ]
    nrow = int(np.ceil(len(methods) / ncol))
    fig = plt.figure(figsize=(2.6 * ncol, 0.34 * nrow + 0.3))
    fig.legend(handles=handles, loc="center", ncol=ncol, frameon=True,
               framealpha=0.9, handlelength=2.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{prefix}_legend.{ext}"), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {prefix}_legend.{{pdf,png}}")
