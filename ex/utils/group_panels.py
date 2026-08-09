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
import matplotlib.ticker as mticker
import numpy as np

from ex.utils.plot_style import (
    apply as apply_style,
    style_for,
    short_label,
    METHOD_GROUPS,
    ERROR_BAND_ALPHA,
)


# experiment method name -> plot_style registry name.
ALIAS = {"MDRE_15": "MDRE"}

# print-size fonts (printed pt = canvas pt * print_width / canvas_width; the
# 3-panel row at 0.85\linewidth prints ticks near 8pt).
SIZES = dict(tick=16, axlab=17.5, leg=14)


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
                   xscale="linear", yscale="linear", linthresh=None,
                   ylim=None, panel_w=3.2, panel_h=2.6) -> list[str]:
    """one figure: len(METHOD_GROUPS) shared-y panels, methods split by group.

    print-size layout: fonts per SIZES, one pooled legend below the row (the
    groups are disjoint, so every method appears exactly once), the x label
    once as a supxlabel. untitled by design; captions title the figure in-paper.

    Args:
      x: 1D array (length L).
      mean, lo, hi: dict method -> array (L,); lo/hi are the band edges.
      xlabel/ylabel/out_dir/prefix/xscale/yscale: as named.
      linthresh: symlog linear/log crossover (only used when yscale="symlog";
                 values below it are linear so exact 0 renders; default 1e-3).
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
            ylim = (max(y_lo, 1e-4) * 0.8, y_hi * 1.6)
        elif yscale == "symlog":
            ylim = (0.0, y_hi * 1.4)
        else:
            ylim = (min(0.0, y_lo), y_hi * 1.08)

    n_meth = sum(len(ms) for _, ms in groups)
    rows_leg = int(np.ceil(n_meth / 5))
    legend_h = 0.25 * (SIZES["leg"] / 12.0) * rows_leg + 0.04
    supx_h = 0.30
    fig_h = panel_h + supx_h + legend_h
    fig, axes = plt.subplots(1, len(groups), figsize=(panel_w * len(groups), fig_h),
                             sharey=True)
    axes = np.atleast_1d(axes)
    drawn = []
    for ax, (g, ms) in zip(axes, groups):
        for m in ms:
            kw = style_for(ALIAS.get(m, m))
            ax.plot(x, mean[m], label=short_label(m), linewidth=1.1, markersize=3,
                    alpha=0.75, **kw)
            band_lo = np.asarray(lo[m], dtype=float)
            if yscale == "log":
                band_lo = np.maximum(band_lo, ylim[0])
            ax.fill_between(x, band_lo, hi[m], color=kw["color"],
                            alpha=ERROR_BAND_ALPHA, linewidth=0)
            drawn.append(m)
        ax.set_xscale(xscale)
        if yscale == "symlog":
            ax.set_yscale("symlog", linthresh=(linthresh or 1e-3), linscale=0.5)
        else:
            ax.set_yscale(yscale)
        if yscale == "log":
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=SIZES["tick"])
    axes[0].set_ylabel(ylabel, fontsize=SIZES["axlab"])

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=SIZES["leg"],
               framealpha=0.9, handlelength=1.2, columnspacing=1.0,
               labelspacing=0.25, borderpad=0.3, handletextpad=0.4)
    fig.supxlabel(xlabel, fontsize=SIZES["axlab"], y=(legend_h + 0.03) / fig_h)
    fig.tight_layout(pad=0.3, w_pad=0.4, rect=(0, (legend_h + supx_h) / fig_h, 1, 1))

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{prefix}.{ext}"), dpi=150)
    plt.close(fig)
    print(f"  saved {prefix}.{{pdf,png}}")
    return drawn


def plot_group_singles(x, mean, lo, hi, *, xlabel, ylabel, out_dir, prefix_fmt,
                       yscale="linear", ylim=None) -> list[str]:
    """one standalone panel per method group, legend below its own panel.

    sized for a 0.32\\linewidth subfigure (three-up figure*): 3.5in canvas so
    canvas fonts print near body text at that width. emits
    {out_dir}/{prefix_fmt.format(group=g)}.{pdf,png} per non-empty group;
    returns the methods drawn.
    """
    apply_style()
    os.makedirs(out_dir, exist_ok=True)
    x = np.asarray(x)
    drawn = []
    for g, members in METHOD_GROUPS.items():
        ms = [m for m in _resolve(members, mean) if np.isfinite(mean[m]).any()]
        if not ms:
            continue
        rows_leg = int(np.ceil(len(ms) / 2))
        legend_h = 0.30 * (13.5 / 12.0) * rows_leg + 0.05
        panel_h = 2.9
        fig_h = panel_h + legend_h
        fig, ax = plt.subplots(figsize=(3.5, fig_h))
        for m in ms:
            kw = style_for(ALIAS.get(m, m))
            ax.plot(x, mean[m], label=short_label(m), linewidth=1.3, markersize=4,
                    alpha=0.85, **kw)
            ax.fill_between(x, lo[m], hi[m], color=kw["color"],
                            alpha=ERROR_BAND_ALPHA, linewidth=0)
            drawn.append(m)
        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=16)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        fig.legend(loc="lower center", ncol=2, fontsize=13.5, framealpha=0.9,
                   handlelength=1.2, columnspacing=0.8, labelspacing=0.25,
                   borderpad=0.3, handletextpad=0.4)
        fig.tight_layout(pad=0.35, rect=(0, legend_h / fig_h, 1, 1))
        prefix = prefix_fmt.format(group=g)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"{prefix}.{ext}"), dpi=150)
        plt.close(fig)
        print(f"  saved {prefix}.{{pdf,png}}")
    return drawn
