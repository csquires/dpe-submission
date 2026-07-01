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
import numpy as np
from matplotlib.patches import Patch


COLOR_NON_TRI = "#4878d0"                        # steel blue (base methods)
TRI_COLORS = ["#ff7f0e", "#2ca02c", "#d62728"]   # V1 orange, V2 green, V3 red
S2_COLOR = "#9467bd"                             # purple (sigma2 sibling)
BOX_ALPHA = 0.6                                  # translucent so nested overlaps show

# base method -> sigma2 sibling, drawn as an extra nested box on the same column.
S2_OF = {"FMDRE": "FMDRE_S2"}

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


def plot_family_boxplot(data, sweep_values, *, sweep_name="K1",
                        ylabel="Pointwise LDR MAE", out_dir=".", prefix="mae",
                        families=DEFAULT_FAMILIES, s2_of=S2_OF, yscale="log") -> None:
    """family-grouped box plot; one box per sweep value within each family slot.

    Args:
      data: dict method -> array [n_sweep, n_seeds] (nan-padded ok). box at sweep
            index k = distribution of data[method][k].
      sweep_values: length n_sweep; drives box lightness + the lightness legend.
      sweep_name: legend label for the sweep axis (e.g. 'alpha', 'K1').
      ylabel / out_dir / prefix / families / s2_of / yscale: as named.

    emits {out_dir}/{prefix}_boxplot.{pdf,png}.
    """
    data = {m: np.atleast_2d(v) for m, v in data.items() if np.isfinite(v).any()}
    valid = [(b, vs) for b, vs in families
             if (b and b in data) or any(v in data for v in vs) or (s2_of.get(b) in data)]
    if not valid:
        print(f"no data for {prefix} boxplot; skipping")
        return

    n_fam = len(valid)
    n_sw = len(sweep_values)
    offsets = np.linspace(-0.26, 0.26, n_sw) if n_sw > 1 else np.array([0.0])
    fracs = np.linspace(0.45, 1.0, n_sw) if n_sw > 1 else np.array([1.0])

    fig, ax = plt.subplots(figsize=(max(11, n_fam * 1.6), 5))

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
        xlabels.append((base or variants[0]).replace('Triangular', 'Tri').replace('MultiHead', 'MH'))
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
    ax.set_xticklabels(xlabels, rotation=40, ha='right', fontsize=10)
    ax.set_xlim(0.4, n_fam + 0.6)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_yscale(yscale)
    ax.grid(True, axis='y', alpha=0.3)

    hue_handles = [
        Patch(facecolor=COLOR_NON_TRI, alpha=BOX_ALPHA, label='Base'),
        Patch(facecolor=TRI_COLORS[0], alpha=BOX_ALPHA, label='Tri V1'),
        Patch(facecolor=TRI_COLORS[1], alpha=BOX_ALPHA, label='Tri V2'),
        Patch(facecolor=TRI_COLORS[2], alpha=BOX_ALPHA, label='Tri V3'),
        Patch(facecolor=S2_COLOR, alpha=BOX_ALPHA, label='FMDRE S2'),
    ]
    sw_handles = [Patch(facecolor=_shade(COLOR_NON_TRI, fracs[ki]), alpha=BOX_ALPHA,
                        label=f'{sweep_name} = {sweep_values[ki]:g}') for ki in range(n_sw)]
    leg1 = ax.legend(handles=hue_handles, title='Method (hue)', fontsize=9,
                     loc='upper left', bbox_to_anchor=(1.005, 1.0),
                     borderaxespad=0, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=sw_handles, title=f'{sweep_name} (lightness)', fontsize=9,
              loc='upper left', bbox_to_anchor=(1.005, 0.5),
              borderaxespad=0, framealpha=0.9)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{prefix}_boxplot.{ext}'), dpi=150, bbox_inches='tight')
    print(f"saved {prefix}_boxplot.{{pdf,png}}")
    plt.close(fig)
