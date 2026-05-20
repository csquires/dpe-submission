"""one-shot driver: holdout for ONE (candidate, cell) element.

per-element granularity keeps each slurm job tiny (~5-20 min) so preempt
preemption is unlikely. the launcher (submit_holdout.sh) sbatches a single
array job whose element index encodes (candidate_idx, cell_idx) via
element_idx = candidate_idx * n_cells + cell_idx.

modes:
    # total element count (used by launcher to size the array):
    python -m ex.utils.hpo.optuna.run_holdout \
        --config <dotted> --method <name> --list-elements

    # run one element:
    python -m ex.utils.hpo.optuna.run_holdout \
        --config <dotted> --method <name> --element-idx E

per-element output:
    $DPE_DATA_ROOT/holdout/<exp>/<method>/cand_<trial.number>/cell_<id>.json

`aggregate_holdout.py` runs after all elements: phase-1 builds
candidate_summary.json per cand_<n>/, phase-2 picks the per-study winner.
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


HYPERBAND_BANDS = [100, 200, 400, 800, 1600]   # standard 5-band Hyperband (rf=2)
POOL_K_PER_BUDGET = 10                          # top-K per band before dedup
EVAL_INTERVAL = 50                              # holdout eval frequency (steps)


def _resolve_output_root(override: str | None) -> Path:
    if override:
        return Path(override)
    if "DPE_DATA_ROOT" not in os.environ:
        raise RuntimeError("DPE_DATA_ROOT not set and --output-root not given")
    return Path(os.environ["DPE_DATA_ROOT"]) / "holdout"


def _full_hp(trial, method: str, fixed_hp: dict | None) -> dict:
    """reconstruct the full hp dict objective.py built at HPO time.

    trial.params has only the trial.suggest_* values; module-level constants
    (e.g. num_epochs=2000 in mh_tdre.py) and config.fixed_hp are not present
    and must be re-applied:
      1. replay trial.params via FixedTrial through suggest_hp -> adds constants
         and handles conditional branches.
      2. overlay fixed_hp on top, same order as objective.py.
    """
    full = _suggest_hp(FixedTrial(trial.params), method)
    if fixed_hp:
        full = {**full, **fixed_hp}
    return full


def _json_safe(obj):
    """coerce numpy scalars to native python so json.dumps works."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def main() -> int:
    """parse cli; either list element count or run one element."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="dotted StudyConfig module")
    p.add_argument("--method", required=True, help="method name in config.methods")
    p.add_argument("--element-idx", type=int, default=None,
                   help="0-based index into (candidate, cell) cross product")
    p.add_argument("--list-elements", action="store_true",
                   help="print N_candidates * N_cells and exit")
    p.add_argument("--bands", default=",".join(map(str, HYPERBAND_BANDS)),
                   help="comma-separated Hyperband bands for the pool")
    p.add_argument("--k-per-budget", type=int, default=POOL_K_PER_BUDGET)
    p.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL,
                   help="holdout eval every N training steps")
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

    adapter = get_adapter(cfg.experiment)
    cells = adapter.holdout_pool()
    if not cells:
        logging.error("adapter.holdout_pool() is empty")
        return 1
    n_cells = len(cells)
    total = len(pool) * n_cells

    if args.list_elements:
        print(total)
        return 0
    if args.element_idx is None:
        logging.error("--element-idx required (or --list-elements)")
        return 2
    if not 0 <= args.element_idx < total:
        logging.error(f"--element-idx {args.element_idx} out of range [0, {total})")
        return 1

    cand_idx, cell_idx = divmod(args.element_idx, n_cells)
    trial = pool[cand_idx]
    cell = cells[cell_idx]

    out_root = _resolve_output_root(args.output_root)
    cand_dir = out_root / cfg.experiment / args.method / f"cand_{trial.number}"
    cand_dir.mkdir(parents=True, exist_ok=True)
    cell_id = "_".join(map(str, cell))
    out_path = cand_dir / f"cell_{cell_id}.json"

    # idempotent: if a successful cell file already exists, skip. lets a
    # preempt requeue re-run only the missing cells.
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if existing.get("status") == "success":
                logging.info(f"cell already done: {out_path}; skipping")
                return 0
        except Exception:
            pass

    full_hp = _full_hp(trial, args.method, cfg.fixed_hp)
    metadata = get_metadata(args.method)
    builder = BUILDERS_REGISTRY[metadata["builder"]]
    requires_pstar = metadata["requires_pstar"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(
        f"element {args.element_idx} -> cand_idx={cand_idx} (trial {trial.number}) "
        f"cell={cell} {cfg.experiment}/{args.method} eval_interval={args.eval_interval} "
        f"device={device}"
    )

    traj: list[tuple[int, float | None]] = []

    def step_cb(step, val):
        try:
            v = float(val)
            traj.append((int(step), v if np.isfinite(v) else None))
        except Exception:
            traj.append((int(step), None))

    try:
        data = adapter.load_cell_data(cell, device=device)
        # trial_number must be non-None so base.eval_cell builds an eval_data
        # split; without that, the trainer's step_cb path is gated off and the
        # trajectory comes back empty. use the source trial's number -- it just
        # seeds the (cell, trial)-deterministic eval split.
        final = adapter.eval_cell(
            cell, args.method, builder, full_hp,
            requires_pstar=requires_pstar, device=device,
            step_cb=step_cb, step_cb_interval=args.eval_interval,
            data=data, trial_number=int(trial.number),
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

    out_path.write_text(json.dumps({
        "candidate_trial_number": int(trial.number),
        "candidate_pool_idx": int(cand_idx),
        "cell": list(cell),
        "trajectory": traj,
        "final_value": final,
        "status": status,
        "hp_dict": _json_safe(full_hp),
        "eval_interval": args.eval_interval,
        "max_resource": cfg.max_resource,
    }, indent=2, default=str))
    logging.info(
        f"cell done: status={status} final={final} traj_len={len(traj)} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
