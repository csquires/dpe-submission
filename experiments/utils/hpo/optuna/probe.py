"""
query completed optuna tpe studies for top-k hyperparameters at given budget
by reconstructing tpe parzen estimator posterior and evaluating log-density on grid.

CRITICAL: uses optuna 4.8.0 private APIs. signatures not guaranteed across versions.
verify compatibility if optuna is upgraded.
"""
import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers._tpe.sampler import _split_trials
from optuna.search_space import intersection_search_space

log = logging.getLogger(__name__)


def best_at_budget(
    study: optuna.Study, budget_step: int, k: int = 5, n_grid_samples: int = 10000
) -> list[dict]:
    """
    reconstruct tpe parzen estimator from below-median trials and return top-k hps
    by posterior density at a given budget. study must use TPESampler.
    Returns list of k dicts; each dict maps HP names to sampled values.

    input: optuna study (tpe sampler), budget step, k (num top), grid size
    action: validate -> filter trials at budget -> reconstruct parzen from below-trials
            -> sample grid -> rank by log-density -> extract top-k
    output: list of k dicts mapping param name -> sampled value
    """
    # validate inputs
    _validate_tpe_sampler(study)
    if k <= 0:
        raise ValueError(f"k must be > 0; got {k}")
    if k > n_grid_samples:
        raise ValueError(f"k={k} > n_grid_samples={n_grid_samples}")

    # filter trials with intermediate values at or below budget_step
    trials = []
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
            trials.append(trial)
        else:
            trials.append(trial)

    if len(trials) < 10:
        raise RuntimeError(f"insufficient completed trials for parzen estimator; got {len(trials)}")

    # reconstruct parzen estimator
    sampler = study.sampler
    search_space = sampler._search_space
    if search_space is None:
        search_space = intersection_search_space(study.trials)
        if search_space is None:
            raise RuntimeError("search space not yet determined")

    # compute split threshold and split trials
    n_below = sampler._gamma(len(trials))
    below_trials, _ = _split_trials(
        study, trials, n_below, sampler._constraints_func is not None
    )

    # build parzen estimator from below-trials
    try:
        mpe_below = sampler._build_parzen_estimator(
            study, search_space, below_trials, handle_below=True
        )
    except Exception as e:
        raise RuntimeError(
            f"parzen estimator query failed; check optuna version. {e}"
        ) from e

    # sample grid and evaluate log-density
    rng = np.random.RandomState(study.trials[0].number or 0)
    try:
        samples_dict = mpe_below.sample(rng, n_grid_samples)
        log_pdfs = mpe_below.log_pdf(samples_dict)
    except Exception as e:
        raise RuntimeError(
            f"parzen estimator query failed; check optuna version. {e}"
        ) from e

    # extract top-k by log-density
    top_idx = np.argsort(-log_pdfs)[:k]
    result = [{name: samples_dict[name][i] for name in samples_dict} for i in top_idx]

    return result


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
