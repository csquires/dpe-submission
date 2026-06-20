"""optuna storage backed by a shared Redis journal for distributed hpo.

hpo workers run on many nodes. a single long-lived redis-server (launched by
redis_server.sh) is the only process that ever writes to disk -- its own
append-only file. workers reach it over tcp, discovering its address from an
endpoint file the server publishes on $DPE_DATA_ROOT.

this replaces the file-journal backend: concurrent cross-node appends to one
NFS file are not torn-write safe (a killed/slow writer can poison the whole
journal). routing every write through one atomic redis server removes that
failure class entirely.

provides redis url resolution, per-study key namespacing, idempotent study
creation, and age-thresholded stale-trial reaping.
"""

import datetime
import re
from os import environ
from pathlib import Path
from typing import Hashable

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalRedisBackend
from optuna.trial import TrialState


def redis_url() -> str:
    """resolve the redis url from the endpoint file the server publishes.

    redis_server.sh writes ``<host>:<port>`` to
    ``$DPE_DATA_ROOT/redis/endpoint`` on every (re)start.

    returns:
        str: ``redis://<host>:<port>``.

    raises:
        RuntimeError: if DPE_DATA_ROOT is unset or the endpoint file is
            absent (the redis-server job is not running).
    """
    if "DPE_DATA_ROOT" not in environ:
        raise RuntimeError("DPE_DATA_ROOT environment variable not set")
    endpoint = Path(environ["DPE_DATA_ROOT"]) / "redis" / "endpoint"
    if not endpoint.is_file():
        raise RuntimeError(
            f"redis endpoint file not found at {endpoint}; "
            f"is the redis-server job running? (see redis_server.sh)"
        )
    return f"redis://{endpoint.read_text().strip()}"


def _serialize_slice(s: Hashable) -> str:
    """serialize a slice value to redis-prefix-safe string.

    codomain: [A-Za-z0-9_-]+ (simultaneously redis-prefix-safe and
    slurm-job-name-safe). used by both storage prefix and keeper job names.

    args:
        s: slice identifier (int, str, or tuple of ints).
            tuple branch is restricted to int-only elements because the "_"
            separator would otherwise be ambiguous: with str elements you can
            construct distinct tuples that serialize identically (e.g.,
            ("a", "b") and ("a_b",) both produce "a_b"), breaking the
            _match_slice inverse.

    returns:
        str: serialized slice in [A-Za-z0-9_-]+.

    raises:
        ValueError: if int/str contains forbidden chars or tuple is empty or
            tuple contains a non-int element.
        TypeError: if s is not int, str, or tuple.
    """
    if isinstance(s, int) and not isinstance(s, bool):
        result = str(s)
    elif isinstance(s, str):
        result = s
    elif isinstance(s, tuple):
        if not s:
            raise ValueError("cannot serialize empty tuple slice")
        for i, x in enumerate(s):
            if not isinstance(x, int) or isinstance(x, bool):
                raise ValueError(
                    f"tuple slice element {i} = {x!r} is not int; "
                    f"_serialize_slice only accepts int-only tuples to keep "
                    f"the encoding injective"
                )
        result = "_".join(str(x) for x in s)
    else:
        raise TypeError(f"cannot serialize slice of type {type(s).__name__}")

    # validate codomain [A-Za-z0-9_-]+
    if not re.fullmatch(r"[A-Za-z0-9_-]+", result):
        forbidden_chars = [c for c in result if not re.match(r"[A-Za-z0-9_-]", c)]
        first_forbidden = forbidden_chars[0] if forbidden_chars else "?"
        raise ValueError(
            f"slice serialization contains forbidden char {repr(first_forbidden)}; "
            f"allowed: [A-Za-z0-9_-]"
        )

    return result


def _match_slice(cfg, serialized: str) -> Hashable:
    """find the original slice value whose serialization equals the given string.

    cli entrypoints (run_holdout.py / aggregate_holdout.py) receive --slice as
    a string (output of _serialize_slice); this helper maps it back to the
    original hashable in cfg.slices, enabling adapter.cells_for_slice(...) calls
    that need exact type match against stratify_key(cell).

    args:
        cfg: loaded StudyConfig with .slices list (or None).
        serialized: the --slice CLI argument (must equal _serialize_slice(s)
            for some s in cfg.slices).

    returns:
        the original hashable slice from cfg.slices.

    raises:
        ValueError: cfg.slices is None or empty, or no match found.
    """
    if cfg.slices is None:
        raise ValueError(
            f"cannot match slice '{serialized}': cfg.slices is None "
            f"(this config has no slice fan-out)"
        )
    for s in cfg.slices:
        if _serialize_slice(s) == serialized:
            return s
    raise ValueError(
        f"no slice in cfg.slices matches '{serialized}'; "
        f"available: {[_serialize_slice(x) for x in cfg.slices]}"
    )


def study_prefix(experiment: str, method: str, slice: Hashable | None = None) -> str:
    """redis key namespace for one (experiment, method[, slice]) study's journal.

    slice=None branch MUST equal the pre-slice format byte-for-byte.
    don't refactor the f-string.
    """
    if slice is None:
        return f"{experiment}:{method}"
    return f"{experiment}:{_serialize_slice(slice)}:{method}"


def _get_storage(
    experiment: str, method: str, slice: Hashable | None = None
) -> JournalStorage:
    """build a JournalStorage over the shared redis-server.

    use_cluster=False: every append is a single atomic redis incr+set (a
    server-side lua script), so concurrent cross-node writers stay consistent
    with no client-side file lock.

    args:
        experiment: experiment identifier.
        method: method identifier.
        slice: optional slice identifier for stratified studies.

    returns:
        JournalStorage: configured with a JournalRedisBackend namespaced to
        this study.
    """
    backend = JournalRedisBackend(
        redis_url(), use_cluster=False, prefix=study_prefix(experiment, method, slice)
    )
    return JournalStorage(backend)


def create_or_load(
    experiment: str,
    method: str,
    *,
    slice: Hashable | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    direction: str = "minimize",
    n_startup_trials: int = 10,
) -> optuna.Study:
    """create study if not exists; load if exists (idempotent).

    concurrent access from multiple workers is serialized by the single redis
    server (atomic appends). reuses on-disk config on existing studies; passed
    sampler/pruner args are ignored if the study already exists.

    args:
        experiment: experiment identifier.
        method: method identifier.
        slice: optional slice identifier for stratified studies.
        sampler: optuna sampler; defaults to TPESampler(n_startup_trials,
            multivariate=True, group=True, seed=42).
        pruner: optuna pruner; defaults to HyperbandPruner(min_resource=100,
            max_resource=10000, reduction_factor=3).
        direction: "minimize" or "maximize"; default "minimize".
        n_startup_trials: warmup trials for TPESampler; default 10.

    returns:
        optuna.Study: created or loaded from the shared redis journal.

    note:
        existing study with different sampler config reuses on-disk config.
        callers cannot override config mid-stream; sampler/pruner args are
        passed only on first creation.
    """
    storage = _get_storage(experiment, method, slice)

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
    if slice is not None:
        study_name = f"{study_name}_slice_{_serialize_slice(slice)}"

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    return study


def reap_stale_trials(
    study: optuna.Study, max_age_seconds: float, max_retry: int = 2
) -> int:
    """fail RUNNING trials older than max_age_seconds and re-enqueue their config.

    a RUNNING trial older than (worker walltime + margin) cannot be alive --
    its slurm job was killed at walltime/preemption -- so this is safe for any
    worker to call concurrently; it never touches a genuinely-live trial.
    skip_if_finished=True ensures concurrent reapers idempotently skip
    already-finished trials.

    option-B preemption recovery: a reaped trial's hyperparameters would
    otherwise be lost (a FAIL contributes nothing to TPE). so each reaped
    trial's params are re-enqueued as a fresh WAITING trial for a clean retry,
    bounded by max_retry via a `reap_retry_count` user-attr. skip_if_exists
    keeps concurrent reapers from enqueueing duplicate retries.

    args:
        study: optuna Study instance.
        max_age_seconds: age threshold (seconds). trials with
            state == RUNNING and age > max_age_seconds are marked FAIL.
        max_retry: max number of times a given config is re-enqueued after
            being reaped (tracked per-lineage via the reap_retry_count
            user-attr); default 2.

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
                # re-enqueue the preempted config for a clean retry.
                n_retry = trial.user_attrs.get("reap_retry_count", 0)
                if n_retry < max_retry and trial.params:
                    study.enqueue_trial(
                        trial.params,
                        user_attrs={"reap_retry_count": n_retry + 1},
                        skip_if_exists=True,
                    )
    return count
