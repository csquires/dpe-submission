"""flat box-per-method plotting for eldr-estimation metrics.

one box per method (distribution over cells, nan dropped), ordered by
ex.utils.faceted_lines.order_methods, colored via ex.utils.plot_style.
saves {out_dir}/{prefix}_boxplot.{png,pdf}. returns ordered methods plotted.

ported from family_boxplot patterns but simplified: no nesting/families, one box
per method, flat axis layout.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ex.utils.faceted_lines import order_methods
from ex.utils.plot_style import style_for


def plot_metric_boxes(data: dict[str, np.ndarray], out_dir: str, prefix: str,
                      *, ylabel: str, yscale: str = "log") -> list[str]:
    """one box per method (distribution over cells, nan dropped), ordered by
    order_methods, colored via style_for. saves {out_dir}/{prefix}_boxplot.{png,pdf}.
    returns ordered methods plotted.

    procedure:
    1. order methods canonically via order_methods(data.keys()).
    2. filter to methods with >=1 finite value.
    3. for each method: extract finite vals; guard non-positive if yscale='log'.
    4. create fig/ax with width scaling to method count.
    5. for each method: ax.boxplot + set facecolor from style_for.
    6. configure axes (xticks, ylabel, yscale, grid).
    7. save to {out_dir}/{prefix}_boxplot.{pdf,png}.
    8. return ordered methods plotted (possibly subset of input if some all-nan).
    """
    # order and filter methods.
    ordered = order_methods(data.keys())
    methods_to_plot = []
    vals_by_method = {}
    for m in ordered:
        finite = data[m][np.isfinite(data[m])]
        if yscale == "log":
            finite = finite[finite > 0]
        if finite.size > 0:
            methods_to_plot.append(m)
            vals_by_method[m] = finite

    if not methods_to_plot:
        print(f"no finite data for {prefix}_boxplot; skipping")
        return []

    # create figure and axes. width scales with method count.
    n_methods = len(methods_to_plot)
    figsize = (max(8, n_methods * 0.8), 5)
    fig, ax = plt.subplots(figsize=figsize)

    # draw boxes. one box per method at position 1..n_methods.
    for pos, method in enumerate(methods_to_plot, start=1):
        vals = vals_by_method[method]
        color = style_for(method)["color"]
        bp = ax.boxplot(
            [vals],
            positions=[pos],
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=color, linewidth=1.1),
            capprops=dict(color=color, linewidth=1.1),
            boxprops=dict(edgecolor="black", linewidth=0.6),
            manage_ticks=False,
        )
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)

    # configure axes.
    ax.set_xticks(range(1, n_methods + 1))
    ax.set_xticklabels(methods_to_plot, rotation=40, ha='right', fontsize=10)
    ax.set_xlim(0.4, n_methods + 0.6)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_yscale(yscale)
    ax.grid(True, axis='y', alpha=0.3)

    # save to both pdf and png.
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fpath = os.path.join(out_dir, f'{prefix}_boxplot.{ext}')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
    print(f"saved {prefix}_boxplot.{{pdf,png}}")
    plt.close(fig)

    return methods_to_plot
