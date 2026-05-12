"""slurm array task entrypoint for optuna hpo.

per-task orchestration: parse config, resolve (experiment, method) combo,
compute parallelism from affinity, cleanup zombie trials, spawn M loky
workers via joblib.Parallel with SIGTERM graceful shutdown.
"""
import argparse
import logging
import os
import signal
import sys
from pathlib import Path

import joblib

from experiments.utils.hpo.optuna.study_config import load_config
from experiments.utils.hpo.optuna.cores_registry import get_cores_for_method
from experiments.utils.hpo.optuna.storage import (
    create_or_load,
    cleanup_zombies,
    study_path,
)
from experiments.utils.hpo.optuna import worker


# module-level logger and sentinel
logger = logging.getLogger(__name__)


class _SigtermReceived(Exception):
    """sentinel exception to break parallel iteration on SIGTERM."""
    pass


def _sigterm_handler(signum, frame):
    """raise sentinel exception to gracefully exit parallel.

    loky subprocesses do not auto-forward SIGTERM; this handler breaks
    the parent's task generator iteration, triggering Parallel.__exit__.
    """
    logger.warning("SIGTERM received, shutting down workers...")
    raise _SigtermReceived()


def _run_parallel(
    experiment: str,
    method: str,
    study_seed: int,
    n_workers: int,
    cores_per_trial: int,
    walltime_minutes: int,
    walltime_margin_minutes: int,
) -> int:
    """spawn M loky workers and consume task generator with SIGTERM protection.

    args:
      experiment: experiment name (e.g., "mnist")
      method: method name (e.g., "BDRE")
      study_seed: seed for sampler
      n_workers: number of loky workers to spawn
      cores_per_trial: cores per trial (passed to each worker)
      walltime_minutes: total wall time budget
      walltime_margin_minutes: buffer before timeout

    returns: 0 on success or SIGTERM, 1 on error
    """
    try:
        # load study; cleanup zombies idempotently
        study = create_or_load(experiment, method)
        count = cleanup_zombies(study)
        logger.info(f"cleaned up {count} zombie trials")

        # compute timeout from budget minus margin
        timeout_seconds = (walltime_minutes - walltime_margin_minutes) * 60
        study_url = str(study_path(experiment, method))
        study_name = f"{experiment}_{method}"
        logger.info(
            f"spawning {n_workers} workers for {experiment}/{method}, "
            f"timeout={timeout_seconds}s, cores={cores_per_trial}"
        )

        # build task generator
        def make_tasks():
            """yield M delayed worker tasks."""
            # objective_factory is constructed per-worker from study_url, not in orchestrator
            def objective_factory(trial):
                raise NotImplementedError("use worker-local objective construction")

            for i in range(n_workers):
                yield joblib.delayed(worker.run_worker)(
                    study_url=study_url,
                    study_name=study_name,
                    study_seed=study_seed,
                    timeout_seconds=timeout_seconds,
                    objective_factory=objective_factory,
                    worker_id=i,
                    cores_per_trial=cores_per_trial,
                )

        # spawn parallel executor with SIGTERM handler
        signal.signal(signal.SIGTERM, _sigterm_handler)

        parallel = joblib.Parallel(
            n_jobs=n_workers,
            backend='loky',
            batch_size=1,
            return_as='generator_unordered',
            timeout=None,
        )

        # consume generator; break on SIGTERM
        try:
            for result in parallel(make_tasks()):
                # ignore return values; workers log independently
                pass
            logger.info("all workers completed normally")
            return 0

        except _SigtermReceived:
            # explicit shutdown via context manager
            logger.warning("breaking task iteration, shutting down workers...")
            parallel.__exit__(None, None, None)
            logger.info("workers shutdown complete")
            return 0

    except Exception as e:
        logger.error(f"parallel execution failed: {e}")
        return 1


def main() -> int:
    """entrypoint for python -m experiments.utils.hpo.optuna.submit.

    cli signature:
      python -m experiments.utils.hpo.optuna.submit \\
        --config <dotted.module.path> \\
        [--combo-index <int>]

    returns: 0 on success or SIGTERM, 1 on validation/runtime error
    """
    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    )

    # parse cli args
    parser = argparse.ArgumentParser(
        description="slurm array task entrypoint for optuna hpo"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="dotted module path to StudyConfig (e.g., experiments.utils.hpo.optuna.configs.bdre_pilot)",
    )
    parser.add_argument(
        "--combo-index",
        type=int,
        default=None,
        help="index into combo list (default: SLURM_ARRAY_TASK_ID)",
    )
    args = parser.parse_args()

    # validate environment
    data_root = os.environ.get("DPE_DATA_ROOT")
    if not data_root:
        logger.error("DPE_DATA_ROOT environment variable not set")
        return 1

    # resolve combo index
    if args.combo_index is not None:
        combo_index = args.combo_index
    else:
        slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if not slurm_id:
            logger.error("--combo-index not provided and SLURM_ARRAY_TASK_ID unset")
            return 1
        try:
            combo_index = int(slurm_id)
        except ValueError:
            logger.error(f"invalid SLURM_ARRAY_TASK_ID: {slurm_id}")
            return 1

    # load config
    try:
        config = load_config(args.config)
        logger.info(
            f"loaded config from {args.config}: "
            f"experiment={config.experiment}, methods={config.methods}"
        )
    except (ModuleNotFoundError, AttributeError, TypeError) as e:
        logger.error(f"failed to load config {args.config}: {e}")
        return 1

    # resolve combo
    combos = [(config.experiment, method) for method in config.methods]
    if not combos:
        logger.error("no methods in config")
        return 1

    combo_idx = combo_index % len(combos)
    experiment, method = combos[combo_idx]
    logger.info(f"resolved combo {combo_idx}: ({experiment}, {method})")

    # compute parallelism
    try:
        cores_per_trial = get_cores_for_method(
            method, config.cores_per_trial_overrides
        )
    except KeyError as e:
        logger.error(f"failed to get cores for method {method}: {e}")
        return 1

    cores_available = len(os.sched_getaffinity(0))
    logger.info(f"cores available: {cores_available}, cores_per_trial: {cores_per_trial}")

    if config.n_jobs_per_task is not None:
        n_workers = config.n_jobs_per_task
    else:
        n_workers = cores_available // cores_per_trial

    if n_workers < 1:
        logger.error(
            f"computed n_workers={n_workers} < 1 "
            f"(cores_available={cores_available}, cores_per_trial={cores_per_trial})"
        )
        return 1

    logger.info(f"computed n_workers={n_workers}")

    # spawn workers
    return _run_parallel(
        experiment=experiment,
        method=method,
        study_seed=config.study_seed,
        n_workers=n_workers,
        cores_per_trial=cores_per_trial,
        walltime_minutes=config.walltime_minutes,
        walltime_margin_minutes=config.walltime_margin_minutes,
    )


if __name__ == "__main__":
    sys.exit(main())
