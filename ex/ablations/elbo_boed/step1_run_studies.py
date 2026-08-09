"""thin entry point for elbo_boed ablation: parse slurm task, enumerate grid, dispatch study.

analytic-design variant of eig_elbo_boed; reuses its enumerate_cells/config_hash
(generic, channel-agnostic) and shared study.run_study orchestrator.
"""

import argparse
import os
import sys
import traceback

import yaml
import torch

from ex.ablations.eig_elbo_boed import study
from ex.ablations.eig_elbo_boed.step1_run_studies import enumerate_cells, config_hash
from ex.utils.step2_runner.load_winners import load_winners


def parse_args() -> argparse.Namespace:
    """parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="run one elbo_boed study cell via slurm array task id"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ex/ablations/elbo_boed/config.yaml",
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

    # load ELBO winners once (posterior channel). analytic design -> no design channel,
    # so no EIG winners are needed (winners_eig stays None in the run_study call).
    winners_elbo = load_winners(cfg["winners_path"])

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

    # dispatch study (design_channel=analytic -> 0 design trials, alpha-BO only)
    try:
        result = study.run_study(
            cell=cell,
            cfg=cfg,
            out_dir=cfg["raw_results_dir"],
            device=device,
            winners_elbo=winners_elbo,
            winners_eig=None,
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
