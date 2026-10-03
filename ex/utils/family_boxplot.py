"""generic family-grouped box plot (pendulum-style) for eldr-estimation metrics.

per family: base method = wide steel-blue box; triangular variant(s) = nested
narrower boxes (V1 orange / V2 green / V3 red); optional sigma2 sibling (purple).
a sweep axis (k1 / alpha / ...) is encoded as box lightness within each family
slot. hue -> method identity, lightness -> sweep value. y-axis log by default.

ported from ex/semisynth/pendulum/step4_plot_results.py::plot_boxplot, generalized
so the sweep axis and the per-method seed source are caller-supplied.

    from ex.utils.family_boxplot import plot_family_boxplot
    plot_family_boxplot(per_pair, alphas, sweep_name='alpha',
                        ylabel='Pointwise LDR MAE', out_dir=fig_dir, prefix='mnist_mae')
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

from ex.utils.plot_style import display_name, short_label
from ex.utils.tables import fmt_iqr, write_tables


COLOR_NON_TRI = "#4878d0"                        # steel blue (base methods)
TRI_COLORS = ["#ff7f0e", "#2ca02c", "#d62728"]   # V1 orange, V2 green, V3 red
S2_COLOR = "#9467bd"                             # purple (sigma2 sibling)
BOX_ALPHA = 0.6                                  # translucent so nested overlaps show

# base method -> sigma2 sibling, drawn as an extra nested box on the same column.
S2_OF = {"FMDRE": "FMDRE_S2"}

# print-size fonts: sized so a \linewidth-included figure lands near the 10pt
# body text (printed pt = canvas pt * print_width / canvas_width).
SIZES = dict(tickx=22, ticky=20, ylab=23, leg=18, leg_one_row=15)

# vertical budget below and above the axes, in pixels at DPI. held fixed as the
# canvas grows, so extra height goes to the axes and not to the legend block.
DPI = 150
TOP_PX = 23
LEGEND_PX = dict(one_row=151, two_rows=199)
MAX_ONE_ROW = 9        # hue keys + sweep keys that still fit on a single row
DEFAULT_HEIGHT = 3.9

# sweep axis display names for the lightness legend.
SWEEP_DISP = {"alpha": r"$\alpha$", "K1": r"$K_1$"}

# each family: (base_method_or_None, [triangular_variants]). uses MDRE_15 (the
# classifier base name in the eldr-estimation experiments, cf. pendulum's "MDRE").
DEFAULT_FAMILIES = [
    ("BDRE",          []),
    ("MDRE_15",       ["TriangularMDRE"]),
    ("MultiHeadTDRE", ["MultiHeadTriangularTDRE"]),
    ("TSM",           ["TriangularTSM"]),
    ("CTSM",          ["TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3"]),
    ("VFM",           ["TriangularVFM_V1", "TriangularVFM_V2", "TriangularVFM_V3"]),
    ("FMDRE",         ["TriangularFMDRE"]),
]


def _shade(hex_color: str, frac: float) -> tuple:
    """blend hex_color toward white. frac=1 -> full color, smaller -> lighter."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    return tuple(1.0 - frac * (1.0 - rgb))


def _drawn_families(families, data, s2_of):
    """families that have at least one series present in data."""
    return [(b, vs) for b, vs in families
            if (b and b in data) or any(v in data for v in vs)
            or (s2_of.get(b) in data)]


def _clip_axis(ax, data, yscale) -> None:
    """limit the y-axis to the whisker envelope and count what falls outside.

    envelope = the 1.5*IQR whisker ends over every drawn box, padded by 5% of the
    span. an annotation in the top right reports the number of hidden points, so
    the reader knows the panel is cropped.
    """
    lo, hi = np.inf, -np.inf
    for arr in data.values():
        for row in np.atleast_2d(arr):
            v = row[np.isfinite(row)]
            if v.size == 0:
                continue
            q1, q3 = np.percentile(v, [25, 75])
            iqr = q3 - q1
            lo = min(lo, v[v >= q1 - 1.5 * iqr].min())
            hi = max(hi, v[v <= q3 + 1.5 * iqr].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return
    flat = np.concatenate([np.asarray(v).ravel() for v in data.values()])
    flat = flat[np.isfinite(flat)]
    n_out = int((flat > hi).sum() + (flat < lo).sum())

    if yscale == 'log':
        pad = 0.05 * np.log10(hi / lo)
        ax.set_ylim(lo * 10 ** -pad, hi * 10 ** pad)
    else:
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
    if n_out:
        ax.text(0.995, 0.97, f'{n_out} points outside axis', transform=ax.transAxes,
                ha='right', va='top', fontsize=16, color='0.35')


def plot_family_boxplot(data, sweep_values, *, sweep_name="K1",
                        ylabel="Pointwise LDR MAE", out_dir=".", prefix="mae",
                        families=DEFAULT_FAMILIES, s2_of=S2_OF, yscale="log",
                        height=DEFAULT_HEIGHT, clip_to_whiskers=False,
                        table_data=None) -> None:
    """family-grouped box plot; one box per sweep value within each family slot.

    wide print-size layout: short horizontal family labels, fonts per SIZES, one
    frameless legend row (hue keys then sweep lightness keys) under the axes, or
    two rows if the keys do not fit on one. margins are set from a fixed pixel
    budget so the canvas is deterministic (saved without a tight bbox) and extra
    height reaches the axes.

    Args:
      data: dict method -> array [n_sweep, n_seeds] (nan-padded ok). box at sweep
            index k = distribution of data[method][k].
      sweep_values: length n_sweep; drives box lightness + the lightness legend.
      sweep_name: legend label for the sweep axis (e.g. 'alpha', 'K1').
      height: canvas height in inches. raise it when the boxes look flat; the
            legend block keeps its size, so the axes take the difference.
      clip_to_whiskers: limit the y-axis to the whisker envelope and annotate how
            many points fall outside. use it when a handful of fliers claim most
            of the panel. clipping never changes the emitted tables.
      table_data: series for the tables, if they differ from the plotted ones.
            use it to report a method the panel leaves out. defaults to data.
      ylabel / out_dir / prefix / families / s2_of / yscale: as named.

    emits {out_dir}/{prefix}_boxplot.{pdf,png}.
    """
    data = {m: np.atleast_2d(v) for m, v in data.items() if np.isfinite(v).any()}
    valid = _drawn_families(families, data, s2_of)
    if not valid:
        print(f"no data for {prefix} boxplot; skipping")
        return

    n_fam = len(valid)
    n_sw = len(sweep_values)
    offsets = np.linspace(-0.26, 0.26, n_sw) if n_sw > 1 else np.array([0.0])
    fracs = np.linspace(0.45, 1.0, n_sw) if n_sw > 1 else np.array([1.0])

    fig, ax = plt.subplots(figsize=(max(12.5, n_fam * 1.8), height))

    def _draw_box(values, pos, width, color, zorder):
        vals = np.asarray(values)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return
        bp = ax.boxplot(
            vals, positions=[pos], widths=width, patch_artist=True, showfliers=True,
            flierprops=dict(marker='.', markersize=2, alpha=0.2,
                            markerfacecolor=color, markeredgecolor=color, linestyle='none'),
            medianprops=dict(color='black', linewidth=1.2, zorder=zorder + 1),
            whiskerprops=dict(color=color, linewidth=1.1, zorder=zorder),
            capprops=dict(color=color, linewidth=1.1, zorder=zorder),
            boxprops=dict(edgecolor='black', linewidth=0.6),
            manage_ticks=False, zorder=zorder,
        )
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(BOX_ALPHA)
        bp['boxes'][0].set_zorder(zorder)

    base_w = 0.20
    xticks, xlabels = [], []
    for fam_idx, (base, variants) in enumerate(valid):
        pos = fam_idx + 1
        xticks.append(pos)
        if base and base in data:
            xlabels.append(display_name(base))
        else:
            # base absent: name the slot after the variant that is drawn
            drawn = next((v for v in variants if v in data), None)
            xlabels.append(short_label(drawn) if drawn
                           else display_name(base or variants[0]))
        overlays = [(v, TRI_COLORS[vi % len(TRI_COLORS)])
                    for vi, v in enumerate(variants) if v in data]
        s2 = s2_of.get(base)
        if s2 and s2 in data:
            overlays.append((s2, S2_COLOR))
        for ki in range(n_sw):
            xk = pos + offsets[ki]
            if base and base in data:
                _draw_box(data[base][ki], xk, base_w, _shade(COLOR_NON_TRI, fracs[ki]), zorder=2)
            for oi, (m, c) in enumerate(overlays):
                _draw_box(data[m][ki], xk, base_w * (0.7 - 0.18 * oi),
                          _shade(c, fracs[ki]), zorder=3 + oi)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=SIZES['tickx'])
    ax.set_xlim(0.4, n_fam + 0.6)
    ax.set_ylabel(ylabel, fontsize=SIZES['ylab'])
    ax.set_yscale(yscale)
    ax.tick_params(axis='y', labelsize=SIZES['ticky'])
    if clip_to_whiskers:
        _clip_axis(ax, data, yscale)
    elif yscale == 'log':
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
    ax.grid(True, axis='y', alpha=0.3)

    hue_handles = [
        Patch(facecolor=COLOR_NON_TRI, alpha=BOX_ALPHA, label='Base'),
        Patch(facecolor=TRI_COLORS[0], alpha=BOX_ALPHA, label='Tri V1'),
        Patch(facecolor=TRI_COLORS[1], alpha=BOX_ALPHA, label='Tri V2'),
        Patch(facecolor=TRI_COLORS[2], alpha=BOX_ALPHA, label='Tri V3'),
        Patch(facecolor=S2_COLOR, alpha=BOX_ALPHA, label='FMDRE S2'),
    ]
    sweep_disp = SWEEP_DISP.get(sweep_name, sweep_name)
    sw_handles = [Patch(facecolor=_shade(COLOR_NON_TRI, fracs[ki]), alpha=BOX_ALPHA,
                        label=f'{sweep_disp} = {sweep_values[ki]:g}') for ki in range(n_sw)]
    one_row = len(hue_handles) + n_sw <= MAX_ONE_ROW
    canvas_px = height * DPI
    key = 'one_row' if one_row else 'two_rows'
    fig.subplots_adjust(left=0.085, right=0.995,
                        top=1.0 - TOP_PX / canvas_px,
                        bottom=LEGEND_PX[key] / canvas_px)
    # anchors are figure fractions; scale them so the rows hold their pixel offset.
    scale = DEFAULT_HEIGHT / height
    common = dict(fontsize=SIZES['leg'], loc='lower center', frameon=False,
                  handlelength=1.2, handletextpad=0.5)
    if one_row:
        fig.legend(handles=hue_handles + sw_handles,
                   ncol=len(hue_handles) + n_sw, columnspacing=0.9,
                   bbox_to_anchor=(0.54, 0.005 * scale),
                   **{**common, 'fontsize': SIZES['leg_one_row']})
    else:
        fig.legend(handles=hue_handles, ncol=5, columnspacing=1.2,
                   bbox_to_anchor=(0.54, 0.105 * scale), **common)
        fig.legend(handles=sw_handles, ncol=n_sw, columnspacing=1.2,
                   bbox_to_anchor=(0.54, 0.0), **common)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{prefix}_boxplot.{ext}'), dpi=150)
    print(f"saved {prefix}_boxplot.{{pdf,png}}")
    plt.close(fig)

    tabulated = data if table_data is None else {
        m: np.atleast_2d(v) for m, v in table_data.items() if np.isfinite(v).any()}
    _write_box_tables(tabulated, sweep_values, sweep_name,
                      _drawn_families(families, tabulated, s2_of), s2_of, ylabel,
                      os.path.join(out_dir, f'{prefix}_table'))


def _write_box_tables(data, sweep_values, sweep_name, families, s2_of, ylabel, stem):
    """emit {stem}.md/.tex: per (method, sweep) median [q1, q3] of the box data.

    row order mirrors the figure: per family the base method, then triangular
    variants, then the sigma2 sibling. columns are the sweep values.
    """
    header = ['Method'] + [f'{sweep_name}={v:g}' for v in sweep_values]
    rows = []
    for base, variants in families:
        drawn = ([base] if base and base in data else []) \
            + [v for v in variants if v in data] \
            + ([s2_of[base]] if s2_of.get(base) in data else [])
        for m in drawn:
            cells = []
            for ki in range(len(sweep_values)):
                vals = np.asarray(data[m][ki])
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    cells.append('--')
                else:
                    q1, med, q3 = np.percentile(vals, [25, 50, 75])
                    cells.append(fmt_iqr(med, q1, q3))
            rows.append([display_name(m)] + cells)
    write_tables(stem, [(f'{ylabel} -- median [q1, q3] per {sweep_name}', header, rows)])
