"""optuna trial objective closure with stratified cell pooling and pruning instrumentation."""
import logging
from collections import defaultdict
from math import isfinite
from random import Random
from typing import Callable, Optional

import numpy as np
import optuna

from ex.utils.hpo.suggest_hp import get_metadata, suggest_hp


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


def make_objective(adapter, method: str, builder, study_seed: int) -> Callable:
    """build trial objective closure with stratified cell pooling and pruning.

    adapter: ExperimentAdapter instance with train_pool(), stratify_key, eval_cell(),
        device() methods.
    method: method name string (e.g., "BDRE").
    builder: estimator builder callable.
    study_seed: study-level seed for deterministic cell selection.

    returns: callable accepting optuna.Trial and returning float metric.

    closure behavior (per trial):
      1. suggest hyperparams via suggest_hp(trial, method).
      2. fetch metadata: uses_pruning, requires_pstar.
      3. select cell via stratified_pick(adapter.train_pool(), adapter.stratify_key,
         study_seed, trial.number).
      4. store cell in trial.user_attr('cell').
      5. derive trial-local rng via PCG64(hash((study_seed, trial.number))).
      6. construct step_cb if uses_pruning else None.
      7. call adapter.eval_cell(..., trial_number=trial.number, step_cb_interval=50).
      8. catch RuntimeError, ValueError, AttributeError; return float('inf') on failure.
      9. validate metric is finite; return float('inf') if not.
      10. return metric.
    """

    def objective(trial: optuna.Trial) -> float:
        # 1. suggest hyperparams
        hp = suggest_hp(trial, method)

        # 2. fetch metadata
        metadata = get_metadata(method)
        uses_pruning = metadata["uses_pruning"]
        requires_pstar = metadata["requires_pstar"]

        # 3. stratified cell selection
        cell = stratified_pick(
            adapter.train_pool(),
            adapter.stratify_key,
            study_seed,
            trial.number,
        )

        # 4. store cell in trial
        trial.set_user_attr("cell", tuple(cell))

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
                step_cb_interval=50,
            )
        except (RuntimeError, ValueError, AttributeError) as e:
            logging.warning(f"trial {trial.number} eval failed: {e}")
            return float("inf")

        # 9. validate metric is finite
        if not isfinite(metric):
            logging.warning(f"trial {trial.number} returned non-finite: {metric}")
            return float("inf")

        # 10. return metric
        return metric

    return objective
