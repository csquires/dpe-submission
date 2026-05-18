"""optuna JournalStorage wrapper for NFS-safe study lifecycle.

provides canonical journal paths, idempotent study creation, and zombie trial cleanup.
"""

from pathlib import Path
from os import environ
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState
import optuna
import datetime


def study_path(experiment: str, method: str) -> Path:
    """resolve canonical journal path from DPE_DATA_ROOT env var.

    args:
        experiment: experiment identifier.
        method: method identifier.

    returns:
        Path: `$DPE_DATA_ROOT / experiment / "hpo_optuna" / f"{method}.journal"`

    raises:
        RuntimeError: if DPE_DATA_ROOT environment variable is not set.
    """
    if "DPE_DATA_ROOT" not in environ:
        raise RuntimeError("DPE_DATA_ROOT environment variable not set")

    root = Path(environ["DPE_DATA_ROOT"])
    return root / experiment / "hpo_optuna" / f"{method}.journal"


def _get_storage(path: Path) -> JournalStorage:
    """build JournalStorage with lock.

    creates parent directories if missing; avoids duplication across
    create_or_load calls.

    args:
        path: journal file path.

    returns:
        JournalStorage: configured with JournalFileBackend and 30s lock grace period.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = JournalFileOpenLock(str(path) + ".lock", grace_period=30)
    backend = JournalFileBackend(str(path), lock_obj=lock)
    return JournalStorage(backend)


def create_or_load(
    experiment: str,
    method: str,
    *,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    direction: str = "minimize",
    n_startup_trials: int = 10,
) -> optuna.Study:
    """create study if not exists; load if exists (idempotent).

    concurrent access from multiple workers is serialized via JournalFileOpenLock
    (grace_period=30s). reuses on-disk config on existing studies; passed sampler/
    pruner args are ignored if study already exists.

    args:
        experiment: experiment identifier.
        method: method identifier.
        sampler: optuna sampler; defaults to TPESampler(n_startup_trials,
            multivariate=True, group=True, seed=42).
        pruner: optuna pruner; defaults to HyperbandPruner(min_resource=100,
            max_resource=10000, reduction_factor=3).
        direction: "minimize" or "maximize"; default "minimize".
        n_startup_trials: warmup trials for TPESampler; default 10.

    returns:
        optuna.Study: created or loaded from canonical journal path.

    note:
        existing study with different sampler config reuses on-disk config.
        callers cannot override config mid-stream; sampler/pruner args are
        passed only on first creation.
    """
    path = study_path(experiment, method)
    storage = _get_storage(path)

    if sampler is None:
        sampler = TPESampler(
            n_startup_trials=n_startup_trials,
            multivariate=True,
            group=True,
            seed=42,
        )

    if pruner is None:
        pruner = HyperbandPruner(
            min_resource=100,
            max_resource=10000,
            reduction_factor=3,
        )

    study_name = f"{experiment}_{method}"
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    return study


def reap_stale_trials(study: optuna.Study, max_age_seconds: float) -> int:
    """fail trials in RUNNING state older than max_age_seconds.

    a RUNNING trial older than (worker walltime + margin) cannot be alive —
    its slurm job was killed at walltime — so this is safe for any worker to
    call concurrently; it never touches a genuinely-live trial. the journal
    storage automatically re-reads the latest state, and skip_if_finished=True
    ensures concurrent reapers idempotently skip already-finished trials.

    args:
        study: optuna Study instance.
        max_age_seconds: age threshold (seconds). trials with
            state == RUNNING and age > max_age_seconds are marked FAIL.

    returns:
        int: count of trials marked FAIL.
    """
    count = 0
    now = datetime.datetime.now()
    for trial in study.trials:
        if trial.state == TrialState.RUNNING and trial.datetime_start is not None:
            age = (now - trial.datetime_start).total_seconds()
            if age > max_age_seconds:
                study.tell(trial.number, state=TrialState.FAIL, skip_if_finished=True)
                count += 1
    return count
