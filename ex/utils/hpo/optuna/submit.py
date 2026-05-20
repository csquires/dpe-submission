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

from ex.utils.hpo.optuna.study_config import load_config, resolve_combo
from ex.utils.hpo.optuna.cores_registry import get_cores_for_method
from ex.utils.hpo.optuna.storage import create_or_load
from ex.utils.hpo.optuna import worker


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


def _walltime_seconds(walltime_str: str) -> float:
    """parse HH:MM:SS string to seconds.

    args:
        walltime_str: time string, e.g. "00:30:00"

    returns: total seconds as float

    raises: ValueError if format invalid
    """
    parts = walltime_str.split(':')
    if len(parts) != 3:
        raise ValueError(
            f"walltime must be HH:MM:SS, got '{walltime_str}'"
        )
    try:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    except ValueError as e:
        raise ValueError(
            f"invalid walltime format '{walltime_str}': {e}"
        )


def _run_parallel(
    experiment: str,
    method: str,
    study_seed: int,
    n_workers: int,
    cores_per_trial: int,
    walltime_minutes: int,
    walltime_margin_minutes: int,
    min_resource: int = 100,
    max_resource: int = 10000,
    reduction_factor: int = 3,
    target_trials: int = 320,
    fixed_hp: dict | None = None,
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
        # stale-trial reaping is done once in main() before dispatch; the
        # loky workers each create_or_load the study themselves.
        # compute timeout from budget minus margin
        timeout_seconds = (walltime_minutes - walltime_margin_minutes) * 60
        logger.info(
            f"spawning {n_workers} workers for {experiment}/{method}, "
            f"timeout={timeout_seconds}s, cores={cores_per_trial}"
        )

        # build task generator
        def make_tasks():
            """yield M delayed worker.run_worker tasks, one per loky worker."""
            for i in range(n_workers):
                yield joblib.delayed(worker.run_worker)(
                    experiment=experiment,
                    method=method,
                    study_seed=study_seed,
                    timeout_seconds=timeout_seconds,
                    worker_id=i,
                    cores_per_trial=cores_per_trial,
                    min_resource=min_resource,
                    max_resource=max_resource,
                    reduction_factor=reduction_factor,
                    target_trials=target_trials,
                    fixed_hp=fixed_hp,
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
    """entrypoint for python -m ex.utils.hpo.optuna.submit.

    cli signature:
      python -m ex.utils.hpo.optuna.submit \\
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
        help="dotted module path to StudyConfig (e.g., ex.utils.hpo.optuna.configs.bdre_pilot)",
    )
    parser.add_argument(
        "--combo-index",
        type=int,
        default=None,
        help="index into combo list (default: SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--lane",
        required=True,
        help="lane profile name (e.g., 'gpu', 'cpu', 'short')",
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

    # resolve (experiment, method) from config and combo_index
    try:
        experiment, method = resolve_combo(config, combo_index)
        logger.info(f"resolved combo {combo_index}: ({experiment}, {method})")
    except (ValueError, IndexError) as e:
        logger.error(f"failed to resolve combo {combo_index}: {e}")
        return 1

    # get lane profile
    try:
        from ex.utils.hpo.optuna.lanes import get_lane
        lane = get_lane(args.lane)
        logger.info(
            f"loaded lane '{args.lane}': "
            f"partition={lane.partition}, cpus_per_task={lane.cpus_per_task}, "
            f"batch_size={lane.batch_size}, worker_walltime={lane.worker_walltime}"
        )
    except (KeyError, Exception) as e:
        logger.error(f"failed to load lane '{args.lane}': {e}")
        return 1

    # compute cores per trial: lane override wins, else per-method registry.
    try:
        if lane.cores_per_trial is not None:
            cores_per_trial = lane.cores_per_trial
            logger.info(
                f"cores_per_trial pinned by lane '{args.lane}': {cores_per_trial}"
            )
        else:
            cores_per_trial = get_cores_for_method(method, config.cores_per_trial)
    except KeyError as e:
        logger.error(f"failed to get cores for method {method}: {e}")
        return 1

    cores_available = len(os.sched_getaffinity(0))
    logger.info(f"cores available: {cores_available}, cores_per_trial: {cores_per_trial}")

    # compute batch size from lane
    if lane.batch_size is not None:
        B = lane.batch_size
        logger.info(f"batch_size from lane: {B}")
    else:
        B = max(1, cores_available // cores_per_trial)
        logger.info(
            f"batch_size computed from cores: "
            f"max(1, {cores_available} // {cores_per_trial}) = {B}"
        )

    if B < 1:
        logger.error(f"computed batch_size={B} < 1")
        return 1

    # parse walltime from lane and call reaper at startup
    try:
        walltime_seconds = _walltime_seconds(lane.worker_walltime)
    except ValueError as e:
        logger.error(f"invalid lane walltime '{lane.worker_walltime}': {e}")
        return 1

    # create study and reap stale trials
    try:
        study = create_or_load(experiment, method)
        margin_seconds = 600  # 10-minute buffer
        max_age = walltime_seconds + margin_seconds
        from ex.utils.hpo.optuna.storage import reap_stale_trials
        count = reap_stale_trials(study, max_age_seconds=max_age)
        logger.info(
            f"reaped {count} stale trials (max_age={max_age}s = {walltime_seconds}s + {margin_seconds}s margin)"
        )
    except Exception as e:
        logger.error(f"failed to reap stale trials: {e}")
        return 1

    # dispatch: B=1 -> direct in-process call; B>1 -> loky fanout
    if B == 1:
        # direct call in-process (no loky, CUDA-clean)
        logger.info(f"B=1: calling run_worker directly (in-process)")
        try:
            # optimize() soft timeout from StudyConfig -- intentionally separate
            # from lane.worker_walltime (the slurm hard --time / reaper basis).
            # both the B==1 and B>1 paths use config.walltime_minutes here.
            timeout_seconds = (
                config.walltime_minutes - config.walltime_margin_minutes
            ) * 60
            worker.run_worker(
                experiment=experiment,
                method=method,
                study_seed=config.study_seed,
                timeout_seconds=timeout_seconds,
                worker_id=0,
                cores_per_trial=cores_per_trial,
                min_resource=config.min_resource,
                max_resource=config.max_resource,
                reduction_factor=config.reduction_factor,
                target_trials=config.target_trials,
                fixed_hp=config.fixed_hp,
            )
            logger.info("worker completed normally")
            return 0
        except SystemExit as e:
            # worker calls sys.exit() on failure
            logger.info(f"worker exited with code {e.code}")
            return e.code if e.code is not None else 1
        except Exception as e:
            logger.error(f"worker execution failed: {e}")
            return 1
    else:
        # loky fanout via joblib.Parallel (B > 1)
        logger.info(f"B={B}: spawning loky workers via Parallel")
        return _run_parallel(
            experiment=experiment,
            method=method,
            study_seed=config.study_seed,
            n_workers=B,
            cores_per_trial=cores_per_trial,
            walltime_minutes=config.walltime_minutes,
            walltime_margin_minutes=config.walltime_margin_minutes,
            min_resource=config.min_resource,
            max_resource=config.max_resource,
            reduction_factor=config.reduction_factor,
            target_trials=config.target_trials,
            fixed_hp=config.fixed_hp,
        )


if __name__ == "__main__":
    sys.exit(main())
