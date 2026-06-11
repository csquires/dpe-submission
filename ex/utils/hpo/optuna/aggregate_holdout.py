"""two-phase aggregation of per-(candidate, cell) holdout outputs.

phase 1 (per candidate):
    read every cell_*.json under cand_<n>/, aggregate trajectories per step
    across cells, pick best_step by argmin median MAE across cells (robust
    to per-cell outliers), write candidate_summary.json.

phase 2 (per study):
    build a (candidate x cell) MAE matrix at each candidate's own best_step.
    restrict to cells covered by at least COVERAGE_THRESHOLD of the
    candidates, drop candidates that don't cover the shared set, then rank
    candidates per cell (1 = best). winner = argmin mean rank (Borda), with
    median MAE on the shared cells as the tiebreaker. write best_hp.json
    (winner) and aggregate_summary.csv (full table).

usage:
    python -m ex.utils.hpo.optuna.aggregate_holdout \
        --config <dotted> --method <name>
"""
import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ex.utils.hpo.optuna.study_config import load_config


COVERAGE_THRESHOLD = 0.8  # cells in >= this fraction of candidates form the shared set


def aggregate_candidate(cand_dir: Path) -> dict | None:
    """phase 1: read all cell_*.json under cand_dir, pick a robust best_step,
    write candidate_summary.json.

    procedure:
        read cell_*.json --> per_cell_traj[cell] = [(step, mae), ...]
        per step, collect finite MAEs across cells --> step_vals[step]
        best_step = argmin median(step_vals[step])
        per_cell_at_best[cell] = MAE at best_step (used by phase 2)

    args:
        cand_dir: directory containing this candidate's cell_*.json files.

    returns:
        summary dict with best_step + per-step mean/median/std/n_finite
        + per_cell_at_best, or None if no cells produced any data.
    """
    cell_files = sorted(cand_dir.glob("cell_*.json"))
    if not cell_files:
        logging.warning(f"no cell_*.json under {cand_dir}")
        return None

    per_cell_traj: dict[str, list] = {}
    hp_dict = None
    n_total = len(cell_files)
    n_failed = 0
    trial_number = None
    pool_idx = None

    for f in cell_files:
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            logging.warning(f"cannot parse {f}: {e}")
            n_failed += 1
            continue
        if hp_dict is None:
            hp_dict = d.get("hp_dict")
            trial_number = d.get("candidate_trial_number")
            pool_idx = d.get("candidate_pool_idx")
        if not str(d.get("status", "")).startswith("success"):
            n_failed += 1
        per_cell_traj[f.stem] = d.get("trajectory") or []

    # preindex each cell's trajectory to dict[step, value] once, so the
    # per-step aggregation below is O(N_steps * N_cells) instead of
    # O(N_steps * N_cells * N_traj) from a nested linear scan.
    cell_step_to_val: dict[str, dict[int, float]] = {
        cid: {int(ts): float(tv) for ts, tv in traj
              if tv is not None and np.isfinite(tv)}
        for cid, traj in per_cell_traj.items()
    }
    all_steps = sorted({s for d in cell_step_to_val.values() for s in d})
    step_vals: dict[int, list] = {}
    for s in all_steps:
        vals = [d[s] for d in cell_step_to_val.values() if s in d]
        if vals:
            step_vals[s] = vals
    mean = {s: float(np.mean(v)) for s, v in step_vals.items()}
    median = {s: float(np.median(v)) for s, v in step_vals.items()}
    std = {s: float(np.std(v, ddof=0)) for s, v in step_vals.items()}
    n_finite = {s: len(v) for s, v in step_vals.items()}

    if median:
        best_step = min(median.keys(), key=lambda s: median[s])
        best_value_median = median[best_step]
        best_value_mean = mean[best_step]
        best_std = std[best_step]
        best_n = n_finite[best_step]
    else:
        best_step = None
        best_value_median = None
        best_value_mean = None
        best_std = None
        best_n = 0

    per_cell_at_best: dict[str, float] = {}
    if best_step is not None:
        for fname, step_map in cell_step_to_val.items():
            if best_step in step_map:
                per_cell_at_best[fname.replace("cell_", "")] = step_map[best_step]

    summary = {
        "candidate_trial_number": trial_number,
        "candidate_pool_idx": pool_idx,
        "hp_dict": hp_dict,
        "n_cells_total": n_total,
        "n_cells_failed": n_failed,
        "per_step_mean": {str(k): v for k, v in mean.items()},
        "per_step_median": {str(k): v for k, v in median.items()},
        "per_step_std": {str(k): v for k, v in std.items()},
        "per_step_n_finite": {str(k): v for k, v in n_finite.items()},
        "best_step": best_step,
        "best_value_median": best_value_median,
        "best_value_mean": best_value_mean,
        "best_value_std": best_std,
        "best_n_finite": best_n,
        "per_cell_at_best": per_cell_at_best,
    }
    (cand_dir / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


def pick_winner(summaries: list[dict]) -> tuple[dict | None, dict[int, dict]]:
    """phase 2: rank candidates by median normalised MAE regret over shared cells.

    procedure:
        per_cell_at_best across candidates --> shared cells (>= COVERAGE_THRESHOLD
        of candidates).
        keep candidates that cover all shared cells.
        M[cand, cell] = MAE at cand's best_step on shared cells.
        per cell, normalise: regret[c, cell] = (M[c, cell] - min_c M) / (max - min).
            ties (max == min, all candidates agree) collapse to 0.
            non-finite MAE for a candidate maps to NaN.
        per candidate, score = median over shared cells of regret.
        winner = argmin score. tiebreak: median MAE on shared cells.

    args:
        summaries: list of candidate summary dicts from aggregate_candidate.

    returns:
        (winner_summary, per_candidate_metrics)
        winner_summary augmented with selection metadata; if no candidate has
        cell data it falls back to argmin best_value_median.
        per_candidate_metrics: {trial_number: {"median_regret",
        "median_on_shared", "n_shared_present"}}.
    """
    metrics: dict[int, dict] = {}
    cands = [s for s in summaries if s.get("per_cell_at_best")]
    if not cands:
        finite = [s for s in summaries if s.get("best_value_median") is not None]
        if not finite:
            return None, metrics
        win = min(finite, key=lambda s: s["best_value_median"])
        win = dict(win)
        win["selection"] = {
            "metric": "fallback_median_no_cells",
            "n_shared_cells": 0,
        }
        return win, metrics

    cell_counter: Counter = Counter()
    for s in cands:
        for c in s["per_cell_at_best"]:
            cell_counter[c] += 1
    n_cands = len(cands)
    thr = max(1, int(np.ceil(COVERAGE_THRESHOLD * n_cands)))
    shared = sorted([c for c, n in cell_counter.items() if n >= thr])

    if not shared:
        win = min(cands, key=lambda s: s["best_value_median"])
        win = dict(win)
        win["selection"] = {
            "metric": "fallback_median_no_shared",
            "coverage_threshold": COVERAGE_THRESHOLD,
            "n_shared_cells": 0,
        }
        return win, metrics

    kept = [s for s in cands
            if all(c in s["per_cell_at_best"] for c in shared)]
    if not kept:
        kept = sorted(cands,
                      key=lambda s: -sum(1 for c in shared
                                         if c in s["per_cell_at_best"]))[:1]

    # M[cand, cell]: MAE at each candidate's own best_step on shared cells.
    mat = np.array([[s["per_cell_at_best"][c] for c in shared] for s in kept],
                   dtype=np.float64)
    n_cand, n_cell = mat.shape

    # per-cell normalisation across candidates.
    finite_mask = np.isfinite(mat)
    cell_min = np.where(finite_mask, mat,  np.inf).min(axis=0)   # (n_cell,)
    cell_max = np.where(finite_mask, mat, -np.inf).max(axis=0)
    span = cell_max - cell_min
    denom = np.where(span > 0, span, np.nan)
    regret = (mat - cell_min[None]) / denom[None]
    tied = (span == 0) & np.isfinite(cell_min)
    regret = np.where(np.broadcast_to(tied[None], regret.shape) & finite_mask,
                      0.0, regret)

    # median normalised regret per candidate across cells.
    point = np.nanmedian(regret, axis=1)

    medians = np.median(mat, axis=1)
    for i, s in enumerate(kept):
        tn = s["candidate_trial_number"]
        if tn is None:
            continue
        metrics[int(tn)] = {
            "median_regret":     float(point[i]) if np.isfinite(point[i]) else None,
            "median_on_shared":  float(medians[i]),
            "n_shared_present":  len(shared),
        }

    sortable = np.where(np.isfinite(point), point, np.inf)
    order = sorted(range(len(kept)),
                   key=lambda i: (sortable[i], medians[i]))
    win = dict(kept[order[0]])
    win["selection"] = {
        "metric": "median_normalised_regret",
        "coverage_threshold": COVERAGE_THRESHOLD,
        "n_shared_cells": len(shared),
        "n_candidates_in_pool": len(kept),
        "winner_median_regret":     float(point[order[0]]) if np.isfinite(point[order[0]]) else None,
        "winner_median_on_shared":  float(medians[order[0]]),
        "tiebreak": "median_on_shared",
    }
    return win, metrics


def main() -> int:
    """run phase-1 over each cand_*/, then phase-2 robust winner pick."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="dotted StudyConfig module")
    p.add_argument("--method", required=True, help="method name")
    p.add_argument("--output-root", default=None,
                   help="default $DPE_DATA_ROOT/holdout")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.output_root is not None:
        out_root = Path(args.output_root)
    else:
        if "DPE_DATA_ROOT" not in os.environ:
            logging.error("DPE_DATA_ROOT not set and --output-root not given")
            return 1
        out_root = Path(os.environ["DPE_DATA_ROOT"]) / "holdout"
    root = out_root / cfg.experiment / args.method
    if not root.exists():
        logging.error(f"no holdout output directory: {root}")
        return 1

    summaries: list[dict] = []
    rows = []
    for cand_dir in sorted(root.glob("cand_*")):
        s = aggregate_candidate(cand_dir)
        if s is None:
            continue
        summaries.append(s)
        rows.append({
            "trial_number": s["candidate_trial_number"],
            "pool_idx": s["candidate_pool_idx"],
            "best_step": s["best_step"],
            "best_value_median": s["best_value_median"],
            "best_value_mean": s["best_value_mean"],
            "best_value_std": s["best_value_std"],
            "best_n_finite": s["best_n_finite"],
            "n_cells_total": s["n_cells_total"],
            "n_cells_failed": s["n_cells_failed"],
            "cand_dir": str(cand_dir),
        })
    if not rows:
        logging.error(f"no cand_*/cell_*.json data under {root}")
        return 1

    win, per_cand_metrics = pick_winner(summaries)
    if win is None:
        logging.error("no candidate produced a finite best_value_median")
        return 1

    df = pd.DataFrame(rows)
    for col in ("median_regret", "median_on_shared"):
        df[col] = df["trial_number"].map(
            lambda tn, c=col: per_cand_metrics.get(int(tn), {}).get(c)
            if tn is not None else None
        )
    df = df.sort_values(
        ["median_regret", "median_on_shared", "best_value_median"],
        na_position="last",
    ).reset_index(drop=True)
    summary_csv = root / "aggregate_summary.csv"
    df.to_csv(summary_csv, index=False)

    best = {
        "experiment": cfg.experiment,
        "method": args.method,
        "winner_trial_number": int(win["candidate_trial_number"]),
        "winner_pool_idx": (int(win["candidate_pool_idx"])
                             if win.get("candidate_pool_idx") is not None
                             else None),
        "best_step": int(win["best_step"]),
        "best_value_median": float(win["best_value_median"]),
        "best_value_mean": float(win["best_value_mean"]),
        "best_value_std": (float(win["best_value_std"])
                           if win.get("best_value_std") is not None else None),
        "best_n_finite": int(win["best_n_finite"]),
        "best_hp": win.get("hp_dict"),
        "selection": win["selection"],
        "n_candidates_evaluated": int(len(rows)),
        "n_candidates_finite": int(
            df.dropna(subset=["best_value_median"]).shape[0]
        ),
    }
    (root / "best_hp.json").write_text(
        json.dumps(best, indent=2, default=str)
    )
    sel = best["selection"]
    if sel["metric"] == "borda_meanrank":
        logging.info(
            f"winner: trial {best['winner_trial_number']} "
            f"best_step={best['best_step']} "
            f"median={best['best_value_median']:.5f} "
            f"mean_rank={sel['winner_mean_rank']:.2f} "
            f"on {sel['n_shared_cells']} shared cells "
            f"({sel['n_candidates_in_borda']}/{len(rows)} candidates in borda)."
        )
    else:
        logging.info(
            f"winner (fallback={sel['metric']}): trial {best['winner_trial_number']} "
            f"best_step={best['best_step']} "
            f"median={best['best_value_median']:.5f}."
        )
    logging.info(f"best_hp -> {root / 'best_hp.json'}")
    logging.info(f"aggregate_summary -> {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
