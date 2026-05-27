"""re-evaluate top-K hyperparameters on holdout cells at full budget."""

import json
import logging
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import optuna

from . import probe
from ex.utils.hpo import suggest_hp

logger = logging.getLogger(__name__)


def run_holdout(
    study: optuna.Study,
    adapter,
    method: str,
    builder: Callable,
    top_k: int = 5,
    full_budget_steps: int = 6400,
    device: str | None = None,
    output_dir: Path | None = None,
    fixed_hp: dict | None = None,
) -> pd.DataFrame:
    """
    evaluate top-K optuna study hyperparameters on adapter.holdout_pool() cells.

    Args:
        study: loaded optuna study (typically from create_or_load).
        adapter: ExperimentAdapter with holdout_pool() and eval_cell methods.
        method: method name in SUGGEST_HP_REGISTRY (e.g., 'BDRE').
        builder: callable[hp_dict, ...] from builders.py; indexed via METADATA['builder'].
        top_k: number of best HPs (by study best value at full_budget_steps) to re-evaluate. default 5.
        full_budget_steps: step budget for each holdout trial; must match study max_resource. default 6400.
        device: torch device (None -> infer from adapter.default_device() or 'cpu'). optional.
        output_dir: if provided, write JSON per (hp, cell) and CSV summary. path created if missing.

    Returns:
        DataFrame with columns [hp_idx, hp_dict_json, cell, cell_metric, mean_metric, std_metric, n_cells].
        Rows = all (hp_idx, cell) pairs; summary stats repeated per hp_idx group.

        If output_dir: also writes {output_dir}/holdout_hp{i}_cell{j}.json (raw) and
        {output_dir}/holdout_summary.csv (grouped by hp_idx).

    Raises:
        ValueError: if adapter is None, holdout_pool is empty, or full_budget_steps is invalid.
        KeyError: if method not in suggest_hp registry.
    """
    # validate inputs
    assert adapter is not None, "adapter required"
    assert isinstance(full_budget_steps, int) and full_budget_steps > 0, "full_budget_steps must be positive int"

    # get holdout cells
    holdout_cells = adapter.holdout_pool()
    if not holdout_cells:
        raise ValueError("adapter.holdout_pool() is empty")

    # infer device
    if device is None:
        device = getattr(adapter, "default_device", lambda: "cpu")()

    # fetch top-K HPs and reconstruct the FULL hp dict the way objective.py did
    # at HPO time:
    #   1. probe returns trial.params (the suggest_* values only -- user-facing).
    #   2. some suggest_hp modules add constants directly (e.g. n_steps=6400)
    #      that are NOT in trial.params. replay each partial through suggest_hp
    #      via FixedTrial to recover those.
    #   3. overlay config.fixed_hp on top, same order as objective.py.
    from optuna.trial import FixedTrial
    from ex.utils.hpo.suggest_hp import suggest_hp as _suggest_hp
    top_k_partials = probe.best_at_budget(study, budget_step=full_budget_steps, k=top_k)
    top_k_hps = []
    for partial in top_k_partials:
        full = _suggest_hp(FixedTrial(partial), method)
        if fixed_hp:
            full = {**full, **fixed_hp}
        top_k_hps.append(full)
    if len(top_k_hps) < top_k:
        logger.warning(
            f"only {len(top_k_hps)} HPs available at budget {full_budget_steps}; requested top_k={top_k}"
        )

    # fetch method metadata
    metadata = suggest_hp.get_metadata(method)
    requires_pstar = metadata["requires_pstar"]
    builder_name = metadata["builder"]

    # evaluate all (hp_idx, cell) pairs
    results = []

    for hp_idx, hp_dict in enumerate(top_k_hps):
        hp_dict_json = json.dumps(hp_dict)

        for cell in holdout_cells:
            logger.info(f"evaluating hp_idx={hp_idx} cell={cell}")

            try:
                data = adapter.load_cell_data(cell, device=device)
                cell_metric = adapter.eval_cell(
                    cell,
                    method,
                    builder,
                    hp_dict,
                    requires_pstar=requires_pstar,
                    device=device,
                    step_cb=None,
                    data=data,
                )

                # validate cell_metric is finite
                if not np.isfinite(cell_metric):
                    logger.warning(
                        f"non-finite metric for hp_idx={hp_idx} cell={cell}: {cell_metric}; setting to inf"
                    )
                    cell_metric = np.inf

            except (RuntimeError, ValueError) as e:
                logger.warning(f"hp_idx={hp_idx} cell={cell} failed: {type(e).__name__}: {e}")
                cell_metric = np.inf

            results.append(
                {
                    "hp_idx": hp_idx,
                    "hp_dict_json": hp_dict_json,
                    "cell": str(cell),
                    "cell_metric": cell_metric,
                }
            )

    # build DataFrame
    df = pd.DataFrame(results)

    # compute summary statistics per hp_idx
    # exclude inf from mean/std via skipna=True (pandas default for numeric operations)
    summary = (
        df[np.isfinite(df["cell_metric"])]
        .groupby("hp_idx")["cell_metric"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_metric", "std": "std_metric", "count": "n_cells"})
    )

    # merge summary back to main DataFrame
    df = df.merge(summary, on="hp_idx", how="left")

    # write optional output files
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # write per (hp, cell) JSON files
        for _, row in df.iterrows():
            hp_idx = int(row["hp_idx"])
            cell_id = str(row["cell"]).replace("/", "_")  # sanitize cell ID for filenames
            hp_dict = json.loads(row["hp_dict_json"])

            json_data = {
                "hp_idx": hp_idx,
                "cell": row["cell"],
                "hp_dict": hp_dict,
                "cell_metric": float(row["cell_metric"]) if np.isfinite(row["cell_metric"]) else None,
                "status": "success" if np.isfinite(row["cell_metric"]) else "failed",
            }

            json_path = output_dir / f"holdout_hp{hp_idx}_cell{cell_id}.json"
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)

        # write summary CSV (one row per hp_idx)
        summary_df = df.drop_duplicates(subset=["hp_idx"])[
            ["hp_idx", "hp_dict_json", "mean_metric", "std_metric", "n_cells"]
        ].reset_index(drop=True)

        # add n_finite count (cells with finite metrics)
        n_finite_list = []
        for hp_idx in summary_df["hp_idx"]:
            hp_df = df[df["hp_idx"] == hp_idx]
            n_finite = np.sum(np.isfinite(hp_df["cell_metric"]))
            n_finite_list.append(n_finite)
        summary_df["n_finite"] = n_finite_list

        csv_path = output_dir / "holdout_summary.csv"
        summary_df.to_csv(csv_path, index=False)

    return df
