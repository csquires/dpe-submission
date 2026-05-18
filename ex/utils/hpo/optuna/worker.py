"""
per-loky-worker optuna study optimizer.

run_worker(experiment, method, study_seed, timeout_seconds, worker_id, cores_per_trial, max_retry)
bootstraps BLAS env, loads study from storage, registers retry callback, and calls study.optimize()
until timeout. designed to run in isolated loky subprocess.
"""

import os
import logging
import sys
import signal

import optuna
import optuna.storages
import optuna.pruners
import optuna.samplers
import optuna.exceptions

from pathlib import Path
from ex.utils.hpo.optuna.storage import create_or_load
from ex.utils.hpo.suggest_hp import get_metadata
from ex.utils.hpo.adapters import get_adapter
from ex.utils.hpo.optuna.objective import make_objective
from ex.utils.hpo.builders import BUILDERS_REGISTRY


def run_worker(
    experiment: str,
    method: str,
    study_seed: int,
    timeout_seconds: float,
    worker_id: int,
    cores_per_trial: int,
    max_retry: int = 2,
) -> None:
    """
    run optuna.study.optimize() in isolated loky subprocess.

    bootstrap order:
      1. set BLAS env vars (OMP, MKL, OPENBLAS) before torch import
      2. import torch; set num_threads
      3. configure logging with worker_id prefix
      4. load/create study via storage.create_or_load(experiment, method)
      5. get adapter; resolve builder from METADATA['builder'] in BUILDERS_REGISTRY
      6. make_objective via objective module
      7. instantiate callbacks = [RetryFailedTrialCallback(...)]
      8. study.optimize(...) with timeout, gc_after_trial=True, catch=(RuntimeError, ValueError)
      9. log completion

    thread config must precede torch import (env vars only take effect on first load).
    sampler seed = hash((study_seed, slurm_job_id, slurm_array_task_id, worker_id)) & 0xFFFFFFFF.
    pruner determined by method.uses_pruning from suggest_hp.get_metadata(method).

    args:
      experiment: str, e.g. "mnist", "dbpedia"
      method: str, e.g. "BDRE", "FMDRE"
      study_seed: int, base random seed for sampler
      timeout_seconds: float, wall-clock timeout for this worker's optimize() call
      worker_id: int, [0, n_jobs); used to derive unique sampler seed per worker
      cores_per_trial: int, BLAS threads to allocate
      max_retry: int, retries on trial failure (default 2 = 3 total attempts)

    returns: None (logs errors and exits nonzero on failure)
    """

    # bootstrap 1: BLAS env setup (BEFORE torch import)
    os.environ["OMP_NUM_THREADS"] = str(cores_per_trial)
    os.environ["MKL_NUM_THREADS"] = str(cores_per_trial)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores_per_trial)

    if "torch" in sys.modules:
        logging.warning(
            "torch already imported before BLAS config; thread counts may not take effect"
        )

    # bootstrap 2: torch import and config
    import torch

    torch.set_num_threads(cores_per_trial)

    # bootstrap 3: logging setup
    logger_name = f"{experiment}.{method}.worker{worker_id}"
    logger = logging.getLogger(logger_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info(
        f"worker {worker_id} starting for {experiment}/{method}, "
        f"timeout={timeout_seconds}s, cores={cores_per_trial}"
    )

    # bootstrap 4: load or create study
    try:
        slurm_job_id = int(os.environ.get("SLURM_JOB_ID", 0))
        slurm_array_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        seed_for_worker = hash((study_seed, slurm_job_id, slurm_array_task_id, worker_id)) & 0xFFFFFFFF
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=10, multivariate=True, group=True, constant_liar=True, seed=seed_for_worker
        )

        metadata = get_metadata(method)
        uses_pruning = metadata.get("uses_pruning", False)
        if uses_pruning:
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=100, max_resource=10000, reduction_factor=3
            )
        else:
            pruner = optuna.pruners.NopPruner()

        study = create_or_load(experiment, method, sampler=sampler, pruner=pruner)
    except Exception as e:
        logger.error(f"failed to load/create study: {e}")
        sys.exit(1)

    # bootstrap 5: resolve builder and adapter
    try:
        adapter = get_adapter(experiment)
    except Exception:
        logger.error(f"adapter not found for {experiment}")
        sys.exit(1)

    try:
        builder_name = get_metadata(method)["builder"]
        builder = BUILDERS_REGISTRY.get(builder_name)
        if builder is None:
            raise KeyError(builder_name)
    except (KeyError, Exception):
        logger.error(f"builder {builder_name} not in registry")
        sys.exit(1)

    # bootstrap 6: construct objective
    try:
        objective_fn = make_objective(adapter, method, builder, study_seed)
    except Exception as e:
        logger.error(f"objective factory failed: {e}")
        sys.exit(1)

    # bootstrap 7: build callbacks list
    callbacks = [
        optuna.storages.RetryFailedTrialCallback(
            max_retry=max_retry, inherit_intermediate_values=False
        )
    ]
    logger.info(f"registered RetryFailedTrialCallback(max_retry={max_retry})")

    # bootstrap 8: optimize study
    try:
        study.optimize(
            objective_fn,
            timeout=timeout_seconds,
            gc_after_trial=True,
            callbacks=callbacks,
            catch=(RuntimeError, ValueError),
        )
        logger.info("optimize() completed or timed out")
    except KeyboardInterrupt:
        logger.info("interrupted")
        sys.exit(130)
    except SystemExit:
        raise

    # report results
    logger.info(
        f"n_trials={len(study.trials)}, "
        f"best_value={study.best_value}, "
        f"best_params={study.best_params}"
    )
    logger.info(f"worker {worker_id} exit normally")
