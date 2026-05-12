"""
generate static (PNG) and interactive (HTML) figures from a completed optuna study.

no external rendering engine (kaleido/Chrome) required:
- matplotlib backend (PNG): optuna.visualization.matplotlib.*
- plotly backend (HTML): optuna.visualization.*
"""

from pathlib import Path
import logging
from typing import Union

import optuna
from optuna.importance import PedAnovaImportanceEvaluator

# matplotlib backend (PNG): no kaleido needed
from optuna.visualization.matplotlib import (
    plot_optimization_history as plot_opt_hist_mpl,
    plot_intermediate_values as plot_inter_mpl,
    plot_parallel_coordinate as plot_parallel_mpl,
    plot_slice as plot_slice_mpl,
    plot_param_importances as plot_importance_mpl,
)

# plotly backend (HTML): pure JS serialization
from optuna.visualization import (
    plot_optimization_history as plot_opt_hist_plotly,
    plot_intermediate_values as plot_inter_plotly,
    plot_parallel_coordinate as plot_parallel_plotly,
    plot_slice as plot_slice_plotly,
    plot_param_importances as plot_importance_plotly,
)

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def generate_all_figures(
    study: optuna.Study,
    output_dir: Path,
    formats: tuple[str, ...] = ("html", "png"),
) -> dict[str, dict[str, Path]]:
    """
    generate optimization figures in specified formats.

    tries to generate all 5 figures independently; logs and skips on per-figure
    failures (too few trials, no intermediate values, etc.).

    input: completed optuna.Study, destination directory, format tuple
    action: generate figures via matplotlib (PNG) and plotly (HTML) backends;
            save to output_dir/{figure_name}.{format}; handle per-figure errors
    output: {figure_name: {format: Path}} nested dict; omits failed figures

    Args:
        study: completed optuna.Study instance
        output_dir: destination for saved figures (created if missing)
        formats: tuple of "html" and/or "png"; defaults to both

    Returns:
        {figure_name: {format: Path}} nested dict. Example:
        {
            "optimization_history": {"html": Path(...), "png": Path(...)},
            "slice": {"html": Path(...), "png": Path(...)},
            ...
        }
        Figures that fail to generate are omitted entirely.

    Raises:
        ValueError: if formats is empty or contains no valid entries
    """
    # validate formats
    if not formats:
        raise ValueError("formats must contain at least one of 'html' or 'png'")

    valid_formats = {"html", "png"}
    invalid = set(formats) - valid_formats
    if invalid:
        logger.warning(f"skipping invalid formats: {invalid}")
    formats = tuple(f for f in formats if f in valid_formats)
    if not formats:
        raise ValueError("no valid formats in request")

    # create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {}

    # optimization_history: trial value progression
    result["optimization_history"] = {}
    for fmt in formats:
        try:
            if fmt == "png":
                ax = plot_opt_hist_mpl(study)
                fig = ax.figure
                path = output_dir / "optimization_history.png"
                fig.savefig(str(path), dpi=150, bbox_inches="tight")
                plt.close(fig)
                result["optimization_history"][fmt] = path
            elif fmt == "html":
                fig = plot_opt_hist_plotly(study)
                path = output_dir / "optimization_history.html"
                fig.write_html(str(path))
                result["optimization_history"][fmt] = path
        except Exception as e:
            logger.warning(f"optimization_history ({fmt}) failed: {e}")
    if not result["optimization_history"]:
        del result["optimization_history"]

    # intermediate_values: per-trial learning curves (pruned trials)
    result["intermediate_values"] = {}
    for fmt in formats:
        try:
            if fmt == "png":
                ax = plot_inter_mpl(study)
                fig = ax.figure
                path = output_dir / "intermediate_values.png"
                fig.savefig(str(path), dpi=150, bbox_inches="tight")
                plt.close(fig)
                result["intermediate_values"][fmt] = path
            elif fmt == "html":
                fig = plot_inter_plotly(study)
                path = output_dir / "intermediate_values.html"
                fig.write_html(str(path))
                result["intermediate_values"][fmt] = path
        except Exception as e:
            logger.warning(f"intermediate_values ({fmt}) failed: {e}")
    if not result["intermediate_values"]:
        del result["intermediate_values"]

    # parallel_coordinate: parameter-value correlations
    result["parallel_coordinate"] = {}
    for fmt in formats:
        try:
            if fmt == "png":
                ax = plot_parallel_mpl(study)
                fig = ax.figure
                path = output_dir / "parallel_coordinate.png"
                fig.savefig(str(path), dpi=150, bbox_inches="tight")
                plt.close(fig)
                result["parallel_coordinate"][fmt] = path
            elif fmt == "html":
                fig = plot_parallel_plotly(study)
                path = output_dir / "parallel_coordinate.html"
                fig.write_html(str(path))
                result["parallel_coordinate"][fmt] = path
        except Exception as e:
            logger.warning(f"parallel_coordinate ({fmt}) failed: {e}")
    if not result["parallel_coordinate"]:
        del result["parallel_coordinate"]

    # slice: per-parameter impact (1D marginal effects)
    result["slice"] = {}
    for fmt in formats:
        try:
            if fmt == "png":
                ax = plot_slice_mpl(study)
                fig = ax.figure
                path = output_dir / "slice.png"
                fig.savefig(str(path), dpi=150, bbox_inches="tight")
                plt.close(fig)
                result["slice"][fmt] = path
            elif fmt == "html":
                fig = plot_slice_plotly(study)
                path = output_dir / "slice.html"
                fig.write_html(str(path))
                result["slice"][fmt] = path
        except Exception as e:
            logger.warning(f"slice ({fmt}) failed: {e}")
    if not result["slice"]:
        del result["slice"]

    # param_importances: feature importance via PedAnova evaluator
    result["param_importances"] = {}
    evaluator = PedAnovaImportanceEvaluator()
    for fmt in formats:
        try:
            if fmt == "png":
                ax = plot_importance_mpl(study, evaluator=evaluator)
                fig = ax.figure
                path = output_dir / "param_importances.png"
                fig.savefig(str(path), dpi=150, bbox_inches="tight")
                plt.close(fig)
                result["param_importances"][fmt] = path
            elif fmt == "html":
                fig = plot_importance_plotly(study, evaluator=evaluator)
                path = output_dir / "param_importances.html"
                fig.write_html(str(path))
                result["param_importances"][fmt] = path
        except Exception as e:
            logger.warning(f"param_importances ({fmt}) failed: {e}")
    if not result["param_importances"]:
        del result["param_importances"]

    return result
