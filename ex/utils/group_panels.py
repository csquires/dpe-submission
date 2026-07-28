"""single-figure row of method-stratified line panels for step4 figures.

one axis per METHOD_GROUPS entry (vfm_fmdre / tsm_ctsm / cls), concatenated left
to right and sharing the y axis, so one image carries all stratifications of one
metric. bands are drawn from explicit lo/hi arrays (caller decides SE vs IQR).

    from ex.utils.group_panels import plot_group_row
    plot_group_row(x, mean, lo, hi, xlabel=..., ylabel=..., out_dir=..., prefix=...)
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ex.utils.plot_style import (
    apply as apply_style,
    style_for,
    METHOD_GROUPS,
    ERROR_BAND_ALPHA,
)


# experiment method name -> plot_style registry name.
ALIAS = {"MDRE_15": "MDRE"}


def _resolve(members: list[str], present) -> list[str]:
    """one experiment method per registry slot: exact name, else alias fallback.

    keeps a raw file that carries both MDRE and MDRE_15 from drawing two
    identically-styled traces on the same panel.
    """
    out = []
    for member in members:
        if member in present:
            out.append(member)
            continue
        for m in present:
            if ALIAS.get(m) == member:
                out.append(m)
                break
    return out


def plot_group_row(x, mean, lo, hi, *, xlabel, ylabel, out_dir, prefix,
                   xscale="linear", yscale="linear",
                   ylim=None, panel_w=5.2, panel_h=4.2) -> list[str]:
    """one figure: len(METHOD_GROUPS) shared-y panels, methods split by group.

    untitled by design: panel membership is carried by the per-panel legends and
    the facet (e.g. alpha) by the filename; captions title the figure in-paper.

    Args:
      x: 1D array (length L).
      mean, lo, hi: dict method -> array (L,); lo/hi are the band edges.
      xlabel/ylabel/out_dir/prefix/xscale/yscale: as named.
      ylim: optional (lo, hi) shared y-range; computed from the data if None.

    a method with no finite mean is skipped; a group with no methods is dropped.
    emits {out_dir}/{prefix}.{pdf,png}; returns the methods drawn (figure order).
    """
    apply_style()
    os.makedirs(out_dir, exist_ok=True)
    x = np.asarray(x)

    groups = []
    for g, members in METHOD_GROUPS.items():
        ms = [m for m in _resolve(members, mean) if np.isfinite(mean[m]).any()]
        if ms:
            groups.append((g, ms))
    if not groups:
        print(f"  skip {prefix}: no finite data")
        return []

    if ylim is None:
        all_lo = [np.nanmin(lo[m]) for _, ms in groups for m in ms if np.isfinite(lo[m]).any()]
        all_hi = [np.nanmax(hi[m]) for _, ms in groups for m in ms if np.isfinite(hi[m]).any()]
        y_lo, y_hi = min(all_lo), max(all_hi)
        if yscale == "log":
            y_lo = max(y_lo, 1e-4)
            # extra headroom on log axes gives the in-panel legend clean space.
            ylim = (y_lo * 0.8, y_hi * 4.5)
        else:
            ylim = (min(0.0, y_lo), y_hi * 1.08)

    fig, axes = plt.subplots(1, len(groups), figsize=(panel_w * len(groups), panel_h),
                             sharey=True)
    axes = np.atleast_1d(axes)
    drawn = []
    for ax, (g, ms) in zip(axes, groups):
        for m in ms:
            kw = style_for(ALIAS.get(m, m))
            # thin + slightly translucent (ms-style) so overlaps stay visible
            ax.plot(x, mean[m], label=m, linewidth=0.85, markersize=3,
                    alpha=0.75, **kw)
            band_lo = np.asarray(lo[m], dtype=float)
            if yscale == "log":
                band_lo = np.maximum(band_lo, ylim[0])
            ax.fill_between(x, band_lo, hi[m], color=kw["color"],
                            alpha=ERROR_BAND_ALPHA, linewidth=0)
            drawn.append(m)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        # each panel legends its own methods; 'best' avoids the traces. two
        # columns keep big panels' legends short enough to fit the headroom.
        ax.legend(loc="best", fontsize=12, framealpha=0.92, ncol=2 if len(ms) > 5 else 1,
                  handlelength=1.4, labelspacing=0.25, borderaxespad=0.3,
                  columnspacing=0.8)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout(pad=0.5, w_pad=0.8)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{prefix}.{ext}"), dpi=150)
    plt.close(fig)
    print(f"  saved {prefix}.{{pdf,png}}")
    return drawn
