"""two-phase aggregation of per-(candidate, cell) holdout outputs.

phase 1 (per candidate):
    read every cell_*.json under cand_<n>/, aggregate trajectories per step
    across cells (mean +/- std, n_finite), find best_step, write
    candidate_summary.json.

phase 2 (per study):
    rank cand_*/candidate_summary.json by best_value_mean ascending, write
    best_hp.json (winner) and aggregate_summary.csv (full table).

usage:
    python -m ex.utils.hpo.optuna.aggregate_holdout \
        --config <dotted> --method <name>
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ex.utils.hpo.optuna.study_config import load_config


def aggregate_candidate(cand_dir: Path) -> dict | None:
    """phase 1: read all cell_*.json under cand_dir, write candidate_summary.json.

    returns the summary dict, or None if no cells succeeded.
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

    # aggregate per step across cells, finite-only.
    all_steps = sorted({s for traj in per_cell_traj.values() for s, _ in traj})
    mean, std, n_finite = {}, {}, {}
    for s in all_steps:
        vals = []
        for traj in per_cell_traj.values():
            for ts, tv in traj:
                if ts == s and tv is not None and np.isfinite(tv):
                    vals.append(tv)
                    break
        if vals:
            mean[s] = float(np.mean(vals))
            std[s] = float(np.std(vals, ddof=0))
            n_finite[s] = len(vals)

    if mean:
        best_step = min(mean.keys(), key=lambda s: mean[s])
        best_value = mean[best_step]
        best_std = std[best_step]
        best_n = n_finite[best_step]
    else:
        best_step = None
        best_value = None
        best_std = None
        best_n = 0

    summary = {
        "candidate_trial_number": trial_number,
        "candidate_pool_idx": pool_idx,
        "hp_dict": hp_dict,
        "n_cells_total": n_total,
        "n_cells_failed": n_failed,
        "per_step_mean": {str(k): v for k, v in mean.items()},
        "per_step_std": {str(k): v for k, v in std.items()},
        "per_step_n_finite": {str(k): v for k, v in n_finite.items()},
        "best_step": best_step,
        "best_value_mean": best_value,
        "best_value_std": best_std,
        "best_n_finite": best_n,
    }
    (cand_dir / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary


def main() -> int:
    """run phase-1 over each cand_*/, then phase-2 pick winner across them."""
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

    # phase 1: per-candidate aggregation.
    rows = []
    for cand_dir in sorted(root.glob("cand_*")):
        s = aggregate_candidate(cand_dir)
        if s is None:
            continue
        rows.append({
            "trial_number": s["candidate_trial_number"],
            "pool_idx": s["candidate_pool_idx"],
            "best_step": s["best_step"],
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

    df = pd.DataFrame(rows).sort_values(
        "best_value_mean", na_position="last"
    ).reset_index(drop=True)
    summary_csv = root / "aggregate_summary.csv"
    df.to_csv(summary_csv, index=False)

    finite = df.dropna(subset=["best_value_mean"])
    if finite.empty:
        logging.error("no candidate produced a finite best_value_mean")
        return 1

    win = finite.iloc[0]
    win_summary_path = Path(win["cand_dir"]) / "candidate_summary.json"
    win_summary = json.loads(win_summary_path.read_text())
    best = {
        "experiment": cfg.experiment,
        "method": args.method,
        "winner_trial_number": int(win["trial_number"]),
        "winner_pool_idx": int(win["pool_idx"]),
        "best_step": int(win["best_step"]),
        "best_value_mean": float(win["best_value_mean"]),
        "best_value_std": (
            float(win["best_value_std"])
            if win["best_value_std"] is not None
            and not pd.isna(win["best_value_std"]) else None
        ),
        "best_n_finite": int(win["best_n_finite"]),
        "best_hp": win_summary.get("hp_dict"),
        "n_candidates_evaluated": int(len(df)),
        "n_candidates_finite": int(len(finite)),
    }
    (root / "best_hp.json").write_text(
        json.dumps(best, indent=2, default=str)
    )
    logging.info(
        f"winner: trial {best['winner_trial_number']} best_step={best['best_step']} "
        f"best_value={best['best_value_mean']:.5f} "
        f"({best['best_n_finite']}/{int(win['n_cells_total'])} cells finite). "
        f"{best['n_candidates_finite']}/{best['n_candidates_evaluated']} "
        f"candidates finite."
    )
    logging.info(f"best_hp -> {root / 'best_hp.json'}")
    logging.info(f"aggregate_summary -> {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
