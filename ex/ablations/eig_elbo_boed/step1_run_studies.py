"""thin entry point for eig_elbo_boed ablation: parse slurm task, enumerate grid, dispatch study."""

import argparse
import os
import sys
import traceback
import hashlib
import json
from pathlib import Path

import yaml
import torch

from ex.ablations.eig_elbo_boed import study
from ex.utils.step2_runner.load_winners import load_winners, list_methods


def config_hash(cfg: dict) -> str:
    """
    stable hash of config dict. used to stamp shards so reruns with mismatched
    config fail loudly. sha256 digest (64-char hex).

    `exclude_methods`/`gather_ignore_config_hash`/`n_seeds` are dropped before
    hashing: they change WHICH cells run (methods, gather policy, seed-rep count),
    not HOW any single cell computes (seed_rep is derived per-cell), so they must
    not invalidate the shards of the cells that remain; this lets a seed
    backfill (n_seeds 2 -> 5) reuse the existing shards instead of re-running them.
    """
    cfg = {k: v for k, v in cfg.items()
           if k not in ("exclude_methods", "gather_ignore_config_hash", "n_seeds")}
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()


def enumerate_cells(cfg: dict, winners: dict, arms=None, geometries=None, methods=None) -> list[tuple]:
    """
    stable enumeration of full study grid via itertools.product order.
    order: arm x geometry x prior_idx x method x seed_rep.
    each tuple is (arm, geometry, prior_idx, method, seed_rep).

    if arms/geometries/methods are None, derive from cfg/winners (full grid);
    if provided, use them as filters. MUST sort methods before passing.
    """
    # derive defaults when not provided
    arms = arms if arms is not None else cfg.get("arms", ["method"])
    geometries = geometries if geometries is not None else cfg["geometries"]
    methods = methods if methods is not None else sorted(list_methods(winners))
    # drop campaign-excluded methods (applies whether methods is a filter or the
    # winners default) so compute AND gather/plots omit them uniformly.
    excl = set(cfg.get("exclude_methods", []))
    if excl:
        methods = [m for m in methods if m not in excl]

    # stable enumeration via nested product; no oracle arm in this ablation.
    cells = []
    for arm in arms:
        arm_methods = methods
        for geom in geometries:
            for prior_idx in range(cfg["num_priors"]):
                for m in arm_methods:
                    for seed_rep in range(cfg["n_seeds"]):
                        cells.append((arm, geom, prior_idx, m, seed_rep))
    return cells


def parse_args() -> argparse.Namespace:
    """parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="run one eig_elbo_boed study cell via slurm array task id"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ex/ablations/eig_elbo_boed/config.yaml",
        help="path to config yaml",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="override SLURM_ARRAY_TASK_ID for local testing",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="method name filter (e.g. TSM CTSM); ad-hoc use only",
    )
    parser.add_argument(
        "--geometries",
        nargs="+",
        default=None,
        help="geometry filter (e.g. diag rot); ad-hoc use only",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help="arm filter; ad-hoc use only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="skip skip-if-done check; rerun study",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    return parser.parse_args()


def main():
    """main entry point: parse, enumerate, dispatch."""
    args = parse_args()

    # load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # validate required keys (including alpha-BO machinery keys)
    required = [
        "num_priors",
        "n_seeds",
        "geometries",
        "raw_results_dir",
        "data_dim",
        "sigma2",
        "winners_path",
        "n_rounds",
        "n_trials",
        "n_trials_alpha",
        "alpha_lo",
        "alpha_hi",
        "design_channel",
    ]
    for key in required:
        if key not in cfg:
            print(f"ERROR: config missing required key '{key}'")
            sys.exit(1)

    # resolve task_id
    task_id = args.task_id
    if task_id is None:
        task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id_str is None:
            print("ERROR: --task-id not given and SLURM_ARRAY_TASK_ID not set")
            sys.exit(1)
        task_id = int(task_id_str)

    # load per-channel winners once (winners-per-channel): winners_elbo drives the
    # DRE-ELBO posterior search (always); winners_eig drives the DRE-EIG design search
    # (only for the dre_eig channel; absent -> analytic).
    winners_elbo = load_winners(cfg["winners_path"])
    winners_eig = load_winners(cfg["eig_winners_path"]) if cfg.get("eig_winners_path") else None

    # enumerate grid (pass arg overrides as filters, or None to use cfg/winners defaults)
    methods_override = sorted(args.methods) if args.methods else None
    cells = enumerate_cells(
        cfg,
        winners_elbo,
        arms=args.arms,
        geometries=args.geometries,
        methods=methods_override
    )

    # bounds check
    if task_id >= len(cells):
        print(f"task_id {task_id} >= grid size {len(cells)}, exit 0")
        sys.exit(0)

    # resolve cell
    cell = cells[task_id]

    # optional skip-if-filtered (for ad-hoc runs with filters)
    if args.geometries is not None or args.methods is not None or args.arms is not None:
        arm, geom, prior_idx, method, seed_rep = cell
        skip = False
        if args.geometries is not None and geom not in args.geometries:
            skip = True
        if args.methods is not None and method not in args.methods:
            skip = True
        if args.arms is not None and arm not in args.arms:
            skip = True
        if skip:
            print(f"task_id {task_id} filtered out (cell={cell}), exit 0")
            sys.exit(0)

    # compute config hash
    h = config_hash(cfg)

    # create output dir
    os.makedirs(cfg["raw_results_dir"], exist_ok=True)

    # resolve device: cuda -> cpu fallback when unavailable (repo convention).
    # lets a CPU-lane task (--device cpu) or a GPU-lane task that lands on a
    # GPU-less/failed node run without crashing; run_study moves every tensor
    # via .to(device), so cpu is a first-class path.
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"task_id {task_id}: cuda requested but unavailable -> using cpu")
        device = "cpu"

    # pin torch intra-op threads to the SLURM cpu allocation so multiple CPU-lane
    # tasks packed on one node don't oversubscribe its cores (in addition to
    # OMP_NUM_THREADS in the sbatch).
    if device == "cpu":
        n_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
        torch.set_num_threads(n_threads)

    # dispatch study
    try:
        result = study.run_study(
            cell=cell,
            cfg=cfg,
            out_dir=cfg["raw_results_dir"],
            device=device,
            winners_elbo=winners_elbo,
            winners_eig=winners_eig,
            config_hash=h,
            force=args.force,
        )
        print(f"task_id {task_id}: cell={cell}, result={result}")
    except Exception as e:
        print(f"task_id {task_id}: cell={cell}, FAILED")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
