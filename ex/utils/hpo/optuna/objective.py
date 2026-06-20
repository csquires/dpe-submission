"""optuna trial objective closure with stratified cell pooling and pruning instrumentation."""
import logging
from collections import defaultdict
from math import isfinite
from random import Random
from typing import Callable, Hashable, Optional

import numpy as np
import optuna

from ex.utils.hpo.optuna.storage import _serialize_slice
from ex.utils.hpo.suggest_hp import get_metadata, suggest_hp


def _hyperband_rungs(pruner) -> frozenset[int]:
    """geometric rung set {min*eta^k} clipped to [min, max] for HyperbandPruner.

    union of all bracket rungs, since the estimator's step counter has no
    knowledge of which bracket the trial belongs to. firing at non-rung
    steps would still load eval_fn (the expensive part); firing only on
    rungs is the right granularity for `should_prune`.

    falls back to interval=50 (returned as a sentinel empty set means caller
    should keep the int default) when pruner is not HyperbandPruner.
    """
    if not isinstance(pruner, optuna.pruners.HyperbandPruner):
        return frozenset()
    mn = pruner._min_resource
    mx = pruner._max_resource
    eta = pruner._reduction_factor
    rungs = []
    r = mn
    while r <= mx:
        rungs.append(int(r))
        r *= eta
    return frozenset(rungs)


def stratified_pick(
    pool: list[tuple],
    stratify_fn: Optional[Callable],
    study_seed: int,
    trial_number: int,
) -> tuple:
    """deterministically select cell from pool via stratification and hashing.

    pool: list of cell tuples.
    stratify_fn: optional function mapping cell -> hashable key (or None). if None,
        unstratified random selection via hash. if provided, groups pool by key
        and round-robins over strata.
    study_seed: study-level seed for determinism.
    trial_number: trial index for deterministic per-trial seeding.

    returns: single cell tuple from pool.

    raises ValueError if pool is empty.
    """
    if not pool:
        raise ValueError("pool cannot be empty")

    if stratify_fn is None:
        # unstratified: hash-pick from full pool
        rng = Random(hash((study_seed, trial_number)) & 0xFFFFFFFF)
        return pool[rng.randint(0, len(pool) - 1)]

    # stratified: group pool by key
    strata = defaultdict(list)
    for cell in pool:
        key = stratify_fn(cell)
        strata[key].append(cell)

    # collect non-None keys in sorted order
    sorted_keys = sorted([k for k in strata.keys() if k is not None])

    # edge case: all keys are None (degenerate single stratum)
    if not sorted_keys:
        rng = Random(hash((study_seed, trial_number)) & 0xFFFFFFFF)
        return strata[None][rng.randint(0, len(strata[None]) - 1)]

    # deterministic stratum selection via round-robin
    stratum_idx = trial_number % len(sorted_keys)
    chosen_key = sorted_keys[stratum_idx]

    # pick within chosen stratum via hash
    rng = Random(hash((study_seed, trial_number, chosen_key)) & 0xFFFFFFFF)
    stratum = strata[chosen_key]
    return stratum[rng.randint(0, len(stratum) - 1)]


def make_objective(adapter, method: str, builder, study_seed: int,
                   fixed_hp: Optional[dict] = None, slice: Optional[Hashable] = None) -> Callable:
    """build trial objective closure with stratified cell pooling and pruning.

    adapter: ExperimentAdapter instance with train_pool(), stratify_key, eval_cell(),
        device() methods.
    method: method name string (e.g., "BDRE").
    builder: estimator builder callable.
    study_seed: study-level seed for deterministic cell selection.
    slice: optional hashable stratification key to restrict pool to single stratum.
        if None, full pool + round-robin stratification (default).
        if set, pool restricted to cells matching slice + uniform random selection.

    returns: callable accepting optuna.Trial and returning float metric.

    closure behavior (per trial):
      1. suggest hyperparams via suggest_hp(trial, method).
      2. fetch metadata: uses_pruning, requires_pstar.
      3. select cell:
         - if slice is None:
           - pool = adapter.train_pool(), stratify_fn = adapter.stratify_key.
           - round-robin stratification over all strata (backward-compatible).
         - else:
           - pool = adapter.cells_for_slice(slice, pool='train').
           - stratify_fn = None (single stratum; uniform random within slice).
           - call stratified_pick(pool, None, study_seed, trial.number).
      4. store cell in trial.user_attr('cell').
      5. if slice is not None, store slice in trial.user_attr('slice') for traceability.
      6. derive trial-local rng via PCG64(hash((study_seed, trial.number))).
      7. construct step_cb if uses_pruning else None.
      8. call adapter.eval_cell(..., trial_number=trial.number, step_cb_interval=50).
      9. catch RuntimeError, ValueError, AttributeError; return float('inf') on
         a bad-hyperparameter failure, but re-raise an OOM RuntimeError (FAIL).
      10. validate metric is finite; return float('inf') if not.
      11. return metric.
    """

    def objective(trial: optuna.Trial) -> float:
        # 1. suggest hyperparams; overlay experiment-level fixed pins
        hp = suggest_hp(trial, method)
        if fixed_hp:
            hp = {**hp, **fixed_hp}

        # 2. fetch metadata
        metadata = get_metadata(method)
        uses_pruning = metadata["uses_pruning"]
        requires_pstar = metadata["requires_pstar"]

        # 3. prepare pool and stratify function based on slice
        if slice is None:
            pool = adapter.train_pool()
            stratify_fn = adapter.stratify_key
        else:
            pool = adapter.cells_for_slice(slice, pool='train')
            stratify_fn = None  # single stratum; skip round-robin stratification

        # 3b. stratified cell selection
        cell = stratified_pick(pool, stratify_fn, study_seed, trial.number)

        # 4. store cell in trial
        trial.set_user_attr("cell", tuple(cell))
        if slice is not None:
            # emit slice via _serialize_slice so the user_attr value matches
            # the output-dir name AND the aggregator's winners_by_slice.csv
            # 'slice' column (which both use _serialize_slice). using str(slice)
            # here would diverge for tuple slices: str((0,1)) = '(0, 1)' but
            # _serialize_slice((0,1)) = '0_1'. downstream joins would then
            # silently miss every tuple-valued slice (pendulum / occupancy).
            trial.set_user_attr("slice", _serialize_slice(slice))

        # 5. derive trial-local rng
        rng = np.random.Generator(
            np.random.PCG64(hash((study_seed, trial.number)) & 0xFFFFFFFF)
        )

        # 6. construct step_cb
        if uses_pruning:

            def step_cb(step: int, score: float) -> None:
                trial.report(score, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        else:
            step_cb = None

        # 6b. only fire step_cb on Hyperband rung boundaries (eval_fn is
        # expensive; non-rung reports are stored but never acted on).
        # fall back to interval=50 if pruner is not HyperbandPruner.
        rungs = _hyperband_rungs(trial.study.pruner)
        step_cb_interval = rungs if rungs else 50

        # 7. call adapter.eval_cell
        try:
            metric = adapter.eval_cell(
                cell,
                method,
                builder,
                hp,
                requires_pstar,
                adapter.device(),
                step_cb=step_cb,
                trial_number=trial.number,
                step_cb_interval=step_cb_interval,
            )
        except (RuntimeError, ValueError, AttributeError) as e:
            # an OOM or missing-device crash is an infrastructure fault, not a
            # verdict on the hyperparameters: re-raise so optuna marks the trial
            # FAIL (kept out of the TPE model and the target-trial count) instead
            # of recording a COMPLETE inf that silently pollutes the study.
            msg = str(e).lower()
            infra = (
                "out of memory" in msg
                or "can't allocate" in msg
                or "not enough memory" in msg
                or "no cuda gpus are available" in msg
                or "cuda error" in msg
            )
            if isinstance(e, RuntimeError) and infra:
                logging.error(f"trial {trial.number} infrastructure fault: {e}")
                raise
            logging.warning(f"trial {trial.number} eval failed: {e}")
            return float("inf")

        # 9. validate metric is finite
        if not isfinite(metric):
            logging.warning(f"trial {trial.number} returned non-finite: {metric}")
            return float("inf")

        # 10. return metric
        return metric

    return objective
