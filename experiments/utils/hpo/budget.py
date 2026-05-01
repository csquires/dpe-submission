"""fixed trial budget allocation and convergence detection for hpo workflow."""

TOTAL_BUDGET: int = 250

STAGE_SPLIT: dict[str, int] = {
    "recalibrate": 0,
    "broad": 200,
    "refined": 49,
    "holdout": 1,
    "persist": 0,
}


def stage_budget(stage: str, total: int = TOTAL_BUDGET) -> int:
    """return trial count for one stage under fixed 250-trial budget.

    args:
      stage: one of "broad", "refined", "holdout", "recalibrate", "persist".
      total: total budget (must equal TOTAL_BUDGET=250 in v1; no scaling).

    returns: integer trial count for this stage.
    raises ValueError if stage not in STAGE_SPLIT.
    raises ValueError if total != TOTAL_BUDGET.
    """
    if total != TOTAL_BUDGET:
        raise ValueError(
            f"total must equal TOTAL_BUDGET ({TOTAL_BUDGET}); no scaling in v1"
        )

    if stage not in STAGE_SPLIT:
        raise ValueError(
            f"unknown stage: {stage}. must be one of {list(STAGE_SPLIT.keys())}"
        )

    return STAGE_SPLIT[stage]


def is_converged(study, *, patience: int = 20, min_trials: int = 30) -> bool:
    """detect convergence via best-value plateau in completed trials.

    input: optuna.Study object, patience (default 20), min_trials (default 30).
    process:
      - filter to COMPLETE trials only (exclude PRUNED, etc.)
      - if fewer than min_trials completed, return False
      - sort by trial.number ascending
      - compute running_min: running_min[i] = min(values[:i+1]) for each i
      - check plateau: return True iff running_min[-1] >= running_min[-patience-1]
        (no improvement over last patience completed trials)
    output: bool.
    """
    from optuna.trial import TrialState

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(completed) < min_trials:
        return False

    sorted_completed = sorted(completed, key=lambda t: t.number)
    values = [t.value for t in sorted_completed]

    running_min = [min(values[: i + 1]) for i in range(len(values))]

    if len(values) < patience + 1:
        return False

    return running_min[-1] >= running_min[-patience - 1]
