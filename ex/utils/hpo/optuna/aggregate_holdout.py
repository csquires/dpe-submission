"""aggregate per-candidate holdout summaries into a single winner per study.

reads $DPE_DATA_ROOT/holdout/<experiment>/<method>/cand_*/candidate_summary.json
written by run_holdout.py, ranks candidates by `best_value_mean` ascending, and
writes:
    best_hp.json           winner's hp_dict + best_step + best_value
    aggregate_summary.csv  one row per candidate (sorted by best_value_mean)

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

import pandas as pd

from ex.utils.hpo.optuna.study_config import load_config


def main() -> int:
    """pick winner across per-candidate summaries; write best_hp + aggregate csv."""
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

    rows = []
    for cand_dir in sorted(root.glob("cand_*")):
        path = cand_dir / "candidate_summary.json"
        if not path.exists():
            logging.warning(f"missing summary: {path}; skipping")
            continue
        try:
            s = json.loads(path.read_text())
        except Exception as e:
            logging.warning(f"cannot parse {path}: {e}")
            continue
        rows.append({
            "trial_number": s.get("candidate_trial_number"),
            "pool_idx": s.get("candidate_pool_idx"),
            "best_step": s.get("best_step"),
            "best_value_mean": s.get("best_value_mean"),
            "best_value_std": s.get("best_value_std"),
            "best_n_finite": s.get("best_n_finite"),
            "n_cells_total": s.get("n_cells_total"),
            "n_cells_failed": s.get("n_cells_failed"),
            "summary_path": str(path),
        })

    if not rows:
        logging.error(f"no candidate_summary.json files under {root}")
        return 1

    df = pd.DataFrame(rows).sort_values(
        "best_value_mean", na_position="last"
    ).reset_index(drop=True)
    summary_path = root / "aggregate_summary.csv"
    df.to_csv(summary_path, index=False)

    finite = df.dropna(subset=["best_value_mean"])
    if finite.empty:
        logging.error(
            f"no candidate produced a finite best_value_mean "
            f"(check per-cell logs under {root})"
        )
        return 1

    win = finite.iloc[0]
    win_summary_path = Path(win["summary_path"])
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
    (root / "best_hp.json").write_text(json.dumps(best, indent=2, default=str))

    logging.info(
        f"winner: trial {best['winner_trial_number']} "
        f"best_step={best['best_step']} "
        f"best_value={best['best_value_mean']:.5f} "
        f"({best['best_n_finite']}/{int(win['n_cells_total'])} cells finite). "
        f"{best['n_candidates_finite']}/{best['n_candidates_evaluated']} "
        f"candidates finite."
    )
    logging.info(f"best_hp -> {root / 'best_hp.json'}")
    logging.info(f"aggregate_summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
