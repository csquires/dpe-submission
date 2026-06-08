"""holdout driver: (candidate, cell) elements; chunk mode with loky fanout
matching the HPO array-worker resource schema.

modes:
    # total element count (used by launcher to size the array of chunks):
    python -m ex.utils.hpo.optuna.run_holdout \
        --config <dotted> --method <name> --list-elements

    # run ONE element (single training):
    python -m ex.utils.hpo.optuna.run_holdout \
        --config <dotted> --method <name> --element-idx E

    # run a CHUNK of B elements in parallel via loky (HPO-style fanout). a
    # slurm array element gets `--chunk-idx I --chunk-size B` and dispatches
    # B loky workers on [I*B, I*B+B). B should match the HPO worker fanout
    # (cpus_per_task // cores_per_trial; both = 16//4 = 4 on the array lane).
    python -m ex.utils.hpo.optuna.run_holdout \
        --config <dotted> --method <name> --chunk-idx I --chunk-size B

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


HYPERBAND_BANDS = [400, 800, 1600, 3200, 6400]       # 5-band Hyperband (rf=2, max_resource=6400)
POOL_K_PER_BUDGET = 10                          # top-K per band before dedup
EVAL_INTERVAL = 500                             # holdout eval frequency (steps)
DEFAULT_CHUNK_SIZE = 4                          # matches HPO B = 16 // 4 cores_per_trial


def _resolve_output_root(override: str | None) -> Path:
    if override:
        return Path(override)
    if "DPE_DATA_ROOT" not in os.environ:
        raise RuntimeError("DPE_DATA_ROOT not set and --output-root not given")
    return Path(os.environ["DPE_DATA_ROOT"]) / "holdout"


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


def _run_element(
    config_dotted: str,
    method: str,
    element_idx: int,
    bands_csv: str,
    k_per_budget: int,
    eval_interval: int,
    output_root_str: str,
    cores_per_trial: int,
) -> tuple[int, str]:
    """fresh-load everything in this (possibly loky-child) process and run
    one (candidate, cell) training. matches the HPO worker bootstrap order
    (BLAS env BEFORE torch import) so within-element multiprocessing does
    not over-saturate the cpu allocation.

    returns (element_idx, status_string) for parent-side logging.
    """
    # BLAS threads BEFORE any torch import (loky uses spawn -> torch is fresh)
    os.environ["OMP_NUM_THREADS"] = str(cores_per_trial)
    os.environ["MKL_NUM_THREADS"] = str(cores_per_trial)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores_per_trial)

    import torch
    torch.set_num_threads(cores_per_trial)

    from optuna.trial import FixedTrial

    from ex.utils.hpo.adapters import get_adapter
    from ex.utils.hpo.builders import BUILDERS_REGISTRY
    from ex.utils.hpo.optuna import probe
    from ex.utils.hpo.optuna.storage import create_or_load
    from ex.utils.hpo.optuna.study_config import load_config
    from ex.utils.hpo.suggest_hp import get_metadata, suggest_hp as _suggest_hp

    cfg = load_config(config_dotted)
    bands = [int(b) for b in bands_csv.split(",")]

    study = create_or_load(cfg.experiment, method)
    pool = probe.top_k_at_each_budget(study, bands, k=k_per_budget)
    adapter = get_adapter(cfg.experiment)
    cells = adapter.holdout_pool()
    n_cells = len(cells)

    cand_idx, cell_idx = divmod(element_idx, n_cells)
    trial = pool[cand_idx]
    cell = cells[cell_idx]

    out_root = Path(output_root_str)
    cand_dir = out_root / cfg.experiment / method / f"cand_{trial.number}"
    cand_dir.mkdir(parents=True, exist_ok=True)
    cell_id = "_".join(map(str, cell))
    out_path = cand_dir / f"cell_{cell_id}.json"

    # idempotent: skip if a successful cell file already exists.
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if existing.get("status") == "success":
                return element_idx, "skipped_already_success"
        except Exception:
            pass

    # reconstruct full hp the way objective.py does at HPO time.
    full_hp = _suggest_hp(FixedTrial(trial.params), method)
    if cfg.fixed_hp:
        full_hp = {**full_hp, **cfg.fixed_hp}

    metadata = get_metadata(method)
    builder = BUILDERS_REGISTRY[metadata["builder"]]
    requires_pstar = metadata["requires_pstar"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        # split; without it, the trainer's step_cb path is gated off and the
        # trajectory comes back empty.
        final = adapter.eval_cell(
            cell, method, builder, full_hp,
            requires_pstar=requires_pstar, device=device,
            step_cb=step_cb, step_cb_interval=eval_interval,
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
        "eval_interval": eval_interval,
        "max_resource": cfg.max_resource,
    }, indent=2, default=str))
    return element_idx, status


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="dotted StudyConfig module")
    p.add_argument("--method", required=True, help="method name in config.methods")
    p.add_argument("--element-idx", type=int, default=None,
                   help="single-element mode: 0-based (candidate, cell) index")
    p.add_argument("--chunk-idx", type=int, default=None,
                   help="chunk mode: run B elements [I*B, I*B+B) via loky")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help="elements per chunk (loky fanout). default matches HPO B")
    p.add_argument("--list-elements", action="store_true")
    p.add_argument("--bands", default=",".join(map(str, HYPERBAND_BANDS)))
    p.add_argument("--k-per-budget", type=int, default=POOL_K_PER_BUDGET)
    p.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    p.add_argument("--cores-per-trial", type=int, default=None,
                   help="BLAS threads per loky worker. required for "
                        "--element-idx / --chunk-idx; the launcher sets it to "
                        "cpus_per_task // chunk_size so workers don't "
                        "oversubscribe the slurm allocation.")
    p.add_argument("--output-root", default=None,
                   help="default $DPE_DATA_ROOT/holdout")
    args = p.parse_args()

    # cheap import here: only needed to count total elements.
    from ex.utils.hpo.optuna.study_config import load_config
    from ex.utils.hpo.optuna.storage import create_or_load
    from ex.utils.hpo.optuna import probe
    from ex.utils.hpo.adapters import get_adapter

    cfg = load_config(args.config)
    if args.method not in cfg.methods:
        logging.error(f"method '{args.method}' not in {cfg.methods}")
        return 1

    out_root = _resolve_output_root(args.output_root)
    # the launcher owns the (cpus, chunk_size, cores) shape; we just consume it.
    # only --list-elements may omit cores_per_trial (no training is launched).
    if (args.element_idx is not None or args.chunk_idx is not None) \
            and args.cores_per_trial is None:
        logging.error("--cores-per-trial required for --element-idx / --chunk-idx")
        return 2
    cores_per_trial = args.cores_per_trial

    # count total elements without spinning up training subsystems.
    bands = [int(b) for b in args.bands.split(",")]
    study = create_or_load(cfg.experiment, args.method)
    pool = probe.top_k_at_each_budget(study, bands, k=args.k_per_budget)
    if not pool:
        logging.error("empty pool")
        return 1
    adapter = get_adapter(cfg.experiment)
    cells = adapter.holdout_pool()
    n_cells = len(cells)
    total = len(pool) * n_cells

    if args.list_elements:
        print(total)
        return 0

    # --- single-element mode (used by the smoke + by aggregator fan-in tests) ---
    if args.element_idx is not None:
        if args.chunk_idx is not None:
            logging.error("pass --element-idx or --chunk-idx, not both")
            return 2
        if not 0 <= args.element_idx < total:
            logging.error(f"--element-idx {args.element_idx} out of range [0, {total})")
            return 1
        logging.info(f"single element {args.element_idx} / {total}")
        e, status = _run_element(
            args.config, args.method, args.element_idx,
            args.bands, args.k_per_budget, args.eval_interval,
            str(out_root), cores_per_trial,
        )
        logging.info(f"element {e}: {status}")
        return 0

    # --- chunk mode (sbatch array element runs B elements via loky) ---
    if args.chunk_idx is None:
        logging.error("--element-idx, --chunk-idx, or --list-elements required")
        return 2

    B = max(1, args.chunk_size)
    start = args.chunk_idx * B
    end = min(start + B, total)
    if start >= total:
        logging.info(f"chunk {args.chunk_idx} (start={start}) >= total {total}; nothing to do")
        return 0
    elements = list(range(start, end))
    logging.info(
        f"chunk {args.chunk_idx} (B={B}): {len(elements)} elements "
        f"{start}..{end - 1} of total {total}; cores_per_trial={cores_per_trial}"
    )

    import joblib
    results = joblib.Parallel(
        n_jobs=len(elements),
        backend="loky",
        batch_size=1,
        return_as="generator_unordered",
    )(
        joblib.delayed(_run_element)(
            args.config, args.method, e,
            args.bands, args.k_per_budget, args.eval_interval,
            str(out_root), cores_per_trial,
        )
        for e in elements
    )
    for e, status in results:
        logging.info(f"  element {e}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
