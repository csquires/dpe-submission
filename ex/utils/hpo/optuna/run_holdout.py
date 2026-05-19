"""one-shot driver: holdout for ONE candidate from the per-budget-pooled top-K.

the pool is `probe.top_k_at_each_budget(study, bands, k=K_PER_BUDGET)` -- the
union of top-K observed trials at each Hyperband band. for each candidate, this
driver runs the candidate's full hp dict at full budget on every holdout cell,
recording a holdout trajectory every `--eval-interval` steps via the adapter's
`step_cb`. then aggregates the trajectory across cells (mean per step) and
records the best (step, value).

modes:
    # list pool size (used by submit_holdout.sh to fan out per-candidate jobs):
    python -m ex.utils.hpo.optuna.run_holdout \
        --config <dotted> --method <name> --list-candidates

    # run a single candidate:
    python -m ex.utils.hpo.optuna.run_holdout \
        --config <dotted> --method <name> --candidate-idx N

per-candidate output under
  $DPE_DATA_ROOT/holdout/<experiment>/<method>/cand_<trial.number>/:
    cell_<...>.json          per-(cell) raw trajectory
    candidate_summary.json   aggregate-per-step + best_step + best_value
the per-study winner is selected later by `aggregate_holdout.py`.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from optuna.trial import FixedTrial

from ex.utils.hpo.adapters import get_adapter
from ex.utils.hpo.builders import BUILDERS_REGISTRY
from ex.utils.hpo.optuna import probe
from ex.utils.hpo.optuna.storage import create_or_load
from ex.utils.hpo.optuna.study_config import load_config
from ex.utils.hpo.suggest_hp import get_metadata, suggest_hp as _suggest_hp


HYPERBAND_BANDS = [100, 200, 400, 800, 1600]   # standard 5-band Hyperband (rf=2, min=100, max=1600)
POOL_K_PER_BUDGET = 10                          # top-K per band before dedup
EVAL_INTERVAL = 50                              # holdout eval frequency


def _resolve_output_root(override: str | None) -> Path:
    """resolve the holdout output root (env-var by default; --output-root override)."""
    if override:
        return Path(override)
    if "DPE_DATA_ROOT" not in os.environ:
        raise RuntimeError("DPE_DATA_ROOT not set and --output-root not given")
    return Path(os.environ["DPE_DATA_ROOT"]) / "holdout"


def _full_hp(trial, method: str, fixed_hp: dict | None) -> dict:
    """reconstruct the full hp dict objective.py built at HPO time.

    trial.params has only the trial.suggest_* values; module-level constants
    (e.g. num_epochs=2000 in mh_tdre.py) and config.fixed_hp are not in
    trial.params and must be re-applied:
      1. replay trial.params via FixedTrial through suggest_hp -> adds constants
         + handles conditional branches deterministically.
      2. overlay fixed_hp on top, same order as objective.py:
           hp = {**suggest_hp_result, **fixed_hp}
    """
    full = _suggest_hp(FixedTrial(trial.params), method)
    if fixed_hp:
        full = {**full, **fixed_hp}
    return full


def _json_safe(obj):
    """coerce numpy/torch scalars to native python so json.dumps works."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def _aggregate_per_step(per_cell_trajectories: dict) -> tuple[dict, dict, dict]:
    """mean/std/n_finite per step, across all holdout cells, finite-only."""
    all_steps = sorted({s for traj in per_cell_trajectories.values() for s, _ in traj})
    mean, std, n = {}, {}, {}
    for s in all_steps:
        vals = []
        for traj in per_cell_trajectories.values():
            for ts, tv in traj:
                if ts == s and tv is not None and np.isfinite(tv):
                    vals.append(tv)
                    break
        if vals:
            mean[s] = float(np.mean(vals))
            std[s] = float(np.std(vals, ddof=0))
            n[s] = len(vals)
    return mean, std, n


def main() -> int:
    """parse cli, fetch pool, run one candidate's holdout trajectory."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="dotted StudyConfig module")
    p.add_argument("--method", required=True, help="method name in config.methods")
    p.add_argument("--candidate-idx", type=int, default=None,
                   help="0-based index into the pool")
    p.add_argument("--list-candidates", action="store_true",
                   help="print pool size and exit (for the launcher)")
    p.add_argument("--bands", default=",".join(map(str, HYPERBAND_BANDS)),
                   help="comma-separated Hyperband band steps")
    p.add_argument("--k-per-budget", type=int, default=POOL_K_PER_BUDGET,
                   help="top-K per band before dedup")
    p.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL,
                   help="holdout eval every N steps")
    p.add_argument("--output-root", default=None,
                   help="default $DPE_DATA_ROOT/holdout")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.method not in cfg.methods:
        logging.error(f"method '{args.method}' not in {cfg.methods}")
        return 1
    bands = [int(b) for b in args.bands.split(",")]

    study = create_or_load(cfg.experiment, args.method)
    pool = probe.top_k_at_each_budget(study, bands, k=args.k_per_budget)
    if not pool:
        logging.error("empty pool (no finite intermediate values at any band)")
        return 1

    if args.list_candidates:
        # used by submit_holdout.sh to count fan-out
        print(len(pool))
        return 0

    if args.candidate_idx is None:
        logging.error("--candidate-idx required (or use --list-candidates)")
        return 2
    if not 0 <= args.candidate_idx < len(pool):
        logging.error(f"--candidate-idx {args.candidate_idx} out of range "
                      f"[0, {len(pool)})")
        return 1

    trial = pool[args.candidate_idx]
    full_hp = _full_hp(trial, args.method, cfg.fixed_hp)

    out_root = _resolve_output_root(args.output_root)
    out = out_root / cfg.experiment / args.method / f"cand_{trial.number}"
    out.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter(cfg.experiment)
    metadata = get_metadata(args.method)
    builder = BUILDERS_REGISTRY[metadata["builder"]]
    requires_pstar = metadata["requires_pstar"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    holdout_cells = adapter.holdout_pool()
    if not holdout_cells:
        logging.error("adapter.holdout_pool() is empty")
        return 1

    logging.info(
        f"holdout candidate idx={args.candidate_idx} (trial {trial.number}) "
        f"{cfg.experiment}/{args.method}: {len(holdout_cells)} cells, "
        f"eval every {args.eval_interval} steps, max budget {cfg.max_resource}, "
        f"device={device}, out={out}"
    )

    per_cell_trajectories = {}
    n_failed = 0
    for cell in holdout_cells:
        cell_id = "_".join(map(str, cell))
        traj: list[tuple[int, float | None]] = []

        def step_cb(step, val, _traj=traj):
            try:
                v = float(val)
                _traj.append((int(step), v if np.isfinite(v) else None))
            except Exception:
                _traj.append((int(step), None))

        try:
            data = adapter.load_cell_data(cell, device=device)
            final = adapter.eval_cell(
                cell, args.method, builder, full_hp,
                requires_pstar=requires_pstar, device=device,
                step_cb=step_cb, step_cb_interval=args.eval_interval,
                data=data,
            )
            if final is None or not np.isfinite(final):
                status = "failed_nonfinite"
                final = None
            else:
                status = "success"
                final = float(final)
        except Exception as e:
            logging.warning(f"cell {cell}: {type(e).__name__}: {e}")
            final = None
            status = f"failed:{type(e).__name__}"
            n_failed += 1

        per_cell_trajectories[cell_id] = traj
        (out / f"cell_{cell_id}.json").write_text(json.dumps({
            "candidate_trial_number": int(trial.number),
            "candidate_pool_idx": int(args.candidate_idx),
            "cell": list(cell),
            "trajectory": traj,
            "final_value": final,
            "status": status,
        }, indent=2, default=str))

    mean, std, n_finite = _aggregate_per_step(per_cell_trajectories)
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
        "candidate_pool_idx": int(args.candidate_idx),
        "candidate_trial_number": int(trial.number),
        "experiment": cfg.experiment,
        "method": args.method,
        "hp_dict": _json_safe(full_hp),
        "bands": bands,
        "k_per_budget": args.k_per_budget,
        "eval_interval": args.eval_interval,
        "max_resource": cfg.max_resource,
        "n_cells_total": len(holdout_cells),
        "n_cells_failed": n_failed,
        "per_step_mean": {str(k): v for k, v in mean.items()},
        "per_step_std": {str(k): v for k, v in std.items()},
        "per_step_n_finite": {str(k): v for k, v in n_finite.items()},
        "best_step": best_step,
        "best_value_mean": best_value,
        "best_value_std": best_std,
        "best_n_finite": best_n,
    }
    (out / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    logging.info(
        f"candidate idx={args.candidate_idx} (trial {trial.number}): "
        f"best_step={best_step} best_value={best_value} "
        f"({best_n}/{len(holdout_cells)} cells finite) -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
