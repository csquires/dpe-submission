"""
query completed optuna studies for top-k observed trials' hyperparameters.

uses only public optuna APIs (trial.value, trial.params, trial.state).
"""
import logging
import math
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState

log = logging.getLogger(__name__)


def best_at_budget(
    study: optuna.Study, budget_step: int, k: int = 5, n_grid_samples: int = 10000
) -> list[dict]:
    """top-K observed COMPLETE trials by trial.value, returning user-facing params.

    a COMPLETE trial has run to its bracket's max resource, so trial.value is the
    value at (or beyond) budget_step. PRUNED, FAIL, and RUNNING trials are skipped,
    as are completed trials whose value is not finite (a non-finite COMPLETE means
    the objective returned inf -- treated as unusable).

    user-facing semantics: trial.params returns categorical choices as strings
    (e.g. "stiff"/"bridge"), numeric distributions as float/int. callers can hand
    each returned dict to a builder directly (after overlaying any fixed_hp the
    objective applied during HPO).

    args:
        study: an optuna Study.
        budget_step: kept for API stability; trial.value already corresponds to
            the full budget for COMPLETE trials.
        k: number of top trials to return.
        n_grid_samples: legacy, ignored (was used by the previous Parzen-sample
            implementation).

    returns: list of up to k dicts of {param_name: user_facing_value}, sorted by
    trial.value ascending (best first).
    """
    _validate_tpe_sampler(study)
    if k <= 0:
        raise ValueError(f"k must be > 0; got {k}")
    _ = budget_step, n_grid_samples  # accepted for API stability

    finite = [
        t for t in study.trials
        if t.state == TrialState.COMPLETE
        and t.value is not None
        and math.isfinite(t.value)
    ]
    if not finite:
        raise RuntimeError("no COMPLETE finite trials in study")
    if len(finite) < k:
        log.warning(
            f"only {len(finite)} COMPLETE finite trial(s); requested top_k={k}"
        )
    finite.sort(key=lambda t: t.value)
    return [dict(t.params) for t in finite[:k]]


def top_k_at_each_budget(
    study: optuna.Study,
    budgets: list[int],
    k: int,
) -> list[Any]:
    """union of top-K observed trials by intermediate_values[b] over each budget.

    rationale: top-K-by-final misses early peakers (a trial that was best at
    step 200 but overfit by step 1600). by pooling top-K at each Hyperband
    band, the pool covers winners at every stage of training.

    for each budget b in `budgets`, rank trials whose intermediate_values[b]
    is finite ascending by that value, keep the top-k, then dedup across
    budgets by trial.number (first-seen order preserved).

    args:
        study: optuna Study.
        budgets: iterable of step values (e.g. Hyperband bands
            [100, 200, 400, 800, 1600]).
        k: per-budget top count.

    returns: list of FrozenTrial in deterministic dedup order. callers can
    read trial.params, trial.intermediate_values, trial.number, etc.
    """
    seen: dict[int, Any] = {}
    for b in budgets:
        ranked = []
        for t in study.trials:
            if t.state not in (TrialState.COMPLETE, TrialState.PRUNED):
                continue
            v = t.intermediate_values.get(b)
            if v is None or not math.isfinite(v):
                continue
            ranked.append((v, t))
        ranked.sort(key=lambda vt: vt[0])
        for _, t in ranked[:k]:
            if t.number not in seen:
                seen[t.number] = t
    return list(seen.values())


def trial_intermediate_values_at_budget(
    study: optuna.Study, budget_step: int
) -> pd.DataFrame:
    """
    extract trial-level values at or below budget_step from study.
    returns dataframe with cols: trial_number, value_at_budget, step_at_budget, params, state.
    sorted by trial_number ascending.

    input: optuna study, budget step
    action: iterate trials -> find max step <= budget_step -> extract value and params
    output: dataframe with trial-level metrics sorted by trial_number
    """
    rows = []
    for trial in study.trials:
        if not trial.intermediate_values:
            continue

        # find max step s <= budget_step
        valid_steps = [s for s in trial.intermediate_values.keys() if s <= budget_step]
        if not valid_steps:
            # no step <= budget_step; use max available and warn
            max_step = max(trial.intermediate_values.keys())
            log.info(
                f"trial {trial.number}: budget_step {budget_step} exceeds all steps; "
                f"using max available step {max_step}"
            )
            s = max_step
        else:
            s = max(valid_steps)

        val = trial.intermediate_values[s]
        row = {
            "trial_number": trial.number,
            "value_at_budget": val,
            "step_at_budget": s,
            "params": trial.params,
            "state": trial.state.name,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("trial_number").reset_index(drop=True)
    return df


def _validate_tpe_sampler(study: optuna.Study) -> None:
    """
    assert study uses TPESampler; raise TypeError if not.
    """
    if not isinstance(study.sampler, optuna.samplers.TPESampler):
        raise TypeError("study.sampler must be TPESampler")
    if len(study.trials) == 0:
        raise RuntimeError("study has no trials")
