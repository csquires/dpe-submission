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
        elif yscale != "symlog":                       # linear: denser major + minor ticks
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.set_ylim(*ylim)
        ax.grid(True, which="major", alpha=0.3)
        if yscale not in ("log", "symlog"):
            ax.grid(True, which="minor", alpha=0.12)
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


def plot_alpha_group_row(x, mean, lo, hi, col_labels, ylim, *, xlabel, ylabel,
                         out_dir, prefix, xscale="linear", yscale="linear",
                         linthresh=None, panel_w=1.7, panel_h=2.6,
                         font_scale=1.0) -> list[str]:
    """wide companion: one block of METHOD_GROUPS panels per column label, blocks
    left to right. ALL panels are flush (no horizontal gap) and share ONE global
    y-range (ylim), so the y axis is drawn once on the leftmost panel. a title
    over each block's centre names it (e.g. "alpha = 0.1").

    Args:
      mean/lo/hi: dict method -> (L, n_col) grid (per-column, e.g. per-alpha).
      col_labels: block title per column index.
      ylim: single (lo, hi) shared by every panel.
    one pooled legend below, x label once as supxlabel. emits {prefix}.{pdf,png};
    returns the methods drawn.
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
    n_col, n_g = len(col_labels), len(groups)

    # font sizes (scaled by font_scale); vertical allowances scale with them.
    fs = font_scale
    tick_fs, title_fs = (SIZES["tick"] - 2) * fs, (SIZES["axlab"] - 2) * fs
    lab_fs, leg_fs = SIZES["axlab"] * fs, SIZES["leg"] * fs

    ph = panel_h * (1 + 0.7 * (fs - 1))                   # taller panels so scaled y label fits
    n_meth = sum(len(ms) for _, ms in groups)
    rows_leg = int(np.ceil(n_meth / 9))
    legend_h = 0.25 * (leg_fs / 12.0) * rows_leg + 0.04
    supx_h, title_h = 0.30 * fs, 0.30 * fs
    fig_h = ph + supx_h + legend_h + title_h
    fig, axes = plt.subplots(1, n_col * n_g, figsize=(panel_w * n_col * n_g, fig_h),
                             sharey=True, gridspec_kw={"wspace": 0.0})
    axes = np.atleast_1d(axes)
    drawn = []
    for ci in range(n_col):
        for gi, (g, ms) in enumerate(groups):
            idx = ci * n_g + gi
            ax = axes[idx]
            for m in ms:
                kw = style_for(ALIAS.get(m, m))
                ax.plot(x, mean[m][:, ci], label=short_label(m), linewidth=1.0,
                        markersize=2.5, alpha=0.75, **kw)
                band_lo = np.asarray(lo[m][:, ci], dtype=float)
                if yscale == "log":
                    band_lo = np.maximum(band_lo, ylim[0])
                ax.fill_between(x, band_lo, hi[m][:, ci], color=kw["color"],
                                alpha=ERROR_BAND_ALPHA, linewidth=0)
                if m not in drawn:
                    drawn.append(m)
            ax.set_xscale(xscale)
            if yscale == "symlog":
                ax.set_yscale("symlog", linthresh=(linthresh or 1e-3), linscale=0.5)
            else:
                ax.set_yscale(yscale)
            ax.set_ylim(*ylim)
            # strictly-interior x ticks so no label lands on a flush boundary;
            # fewer of them when fonts are scaled up (else big labels collide);
            # minor ticks add density without labels.
            nb = 3 if fs > 1.2 else 5
            xt = [t for t in mticker.MaxNLocator(nbins=nb).tick_values(x.min(), x.max())
                  if x.min() < t < x.max()]
            ax.set_xticks(xt)
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
            ax.grid(True, which="major", alpha=0.3)
            ax.tick_params(labelsize=tick_fs)
            if idx != 0:                                   # y axis drawn once (leftmost)
                ax.tick_params(left=False, labelleft=False)
        axes[ci * n_g + n_g // 2].set_title(col_labels[ci], fontsize=title_fs)

    ax0 = axes[0]                                          # denser y ticks, once
    if yscale == "log":
        ax0.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
        ax0.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=12))
    elif yscale != "symlog":
        ax0.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
        ax0.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax0.set_ylabel(ylabel, fontsize=lab_fs)

    handles, labels, seen = [], [], set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h); labels.append(l); seen.add(l)
    fig.legend(handles, labels, loc="lower center", ncol=min(9, len(labels)),
               fontsize=leg_fs, framealpha=0.9, handlelength=1.2,
               columnspacing=1.0, labelspacing=0.25, borderpad=0.3, handletextpad=0.4)
    fig.tight_layout(pad=0.3, rect=(0, (legend_h + supx_h) / fig_h, 1, 1))
    fig.subplots_adjust(wspace=0.0)                        # keep panels flush after tight_layout
    # place the x label just under the tick row (close the gap to the panels)
    fig.supxlabel(xlabel, fontsize=lab_fs, y=(legend_h + supx_h - 0.20 * fs) / fig_h)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{prefix}.{ext}"), dpi=150)
    plt.close(fig)
    print(f"  saved {prefix}.{{pdf,png}}")
    return drawn


def plot_group_singles(x, mean, lo, hi, *, xlabel, ylabel, out_dir, prefix_fmt,
                       yscale="linear", ylim=None, legend_ncol=9,
                       legend_groups=("cls", "tsm_ctsm", "vfm_fmdre")) -> list[str]:
    """one standalone panel per method group plus one shared legend strip.

    sized for a 0.32\\linewidth subfigure (three-up figure*): 3.5in canvas so
    canvas fonts print near body text at that width. the panels carry no
    legend; a separate wide strip lists every drawn method row-major over
    legend_ncol columns, groups in legend_groups order, so it can sit under
    the row at full width. emits {out_dir}/{prefix_fmt.format(group=g)}.{pdf,png}
    per non-empty group and once more with group="legend"; returns the methods
    drawn.
    """
    apply_style()
    os.makedirs(out_dir, exist_ok=True)
    x = np.asarray(x)
    drawn = []
    handles = {}
    for g, members in METHOD_GROUPS.items():
        ms = [m for m in _resolve(members, mean) if np.isfinite(mean[m]).any()]
        if not ms:
            continue
        fig, ax = plt.subplots(figsize=(3.5, 2.9))
        for m in ms:
            kw = style_for(ALIAS.get(m, m))
            (line,) = ax.plot(x, mean[m], label=short_label(m), linewidth=1.3,
                              markersize=4, alpha=0.85, **kw)
            ax.fill_between(x, lo[m], hi[m], color=kw["color"],
                            alpha=ERROR_BAND_ALPHA, linewidth=0)
            drawn.append(m)
            handles.setdefault(g, []).append(line)
        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=16)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        fig.tight_layout(pad=0.35)
        _save(fig, out_dir, prefix_fmt.format(group=g))

    ordered = [h for g in legend_groups for h in handles.get(g, [])]
    ordered += [h for g in handles if g not in legend_groups for h in handles[g]]
    _save(_legend_strip(ordered, legend_ncol), out_dir,
          prefix_fmt.format(group="legend"))
    return drawn


def _legend_strip(handles, ncol):
    """axes-free figure holding one legend, entries laid out row-major.

    matplotlib fills legend columns top-down, so the handles are permuted
    such that reading across a row follows the given order. the canvas is
    7.5in wide (near a two-column text width, so fonts print at canvas size)
    and only as tall as the legend rows need.
    """
    n = len(handles)
    nrow = int(np.ceil(n / ncol))
    grid = [None] * (nrow * ncol)
    for i, h in enumerate(handles):
        grid[(i % ncol) * nrow + i // ncol] = h
    ordered = [h for h in grid if h is not None]
    fig = plt.figure(figsize=(7.5, 0.28 * nrow + 0.12))
    fig.legend(handles=ordered, labels=[h.get_label() for h in ordered],
               loc="center", ncol=ncol, fontsize=10.5, framealpha=0.9,
               handlelength=1.0, columnspacing=0.6, labelspacing=0.2,
               borderpad=0.25, handletextpad=0.3)
    return fig


def _save(fig, out_dir, prefix):
    """write {prefix}.{pdf,png} into out_dir and close the figure."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{prefix}.{ext}"), dpi=150,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  saved {prefix}.{{pdf,png}}")
