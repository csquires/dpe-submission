"""fixed trial budget allocation and convergence detection for hpo workflow.

env-var driven profile selector
    DPE_BUDGET_PROFILE=<name> picks an alternate (TOTAL_BUDGET, STAGE_SPLIT) pair
    from BUDGET_PROFILES below and disables METHOD_BUDGET_OVERRIDES. each profile
    is self-consistent: its STAGE_SPLIT sums to its TOTAL_BUDGET, so the
    workflow.main() startup assert keeps passing as long as launcher --budget
    matches the profile's total.

    profiles
      default       broad=200 refined=49  holdout=1  total=250  (with method overrides)
      refined24     broad=24  refined=0   holdout=1  total=25   (overrides disabled;
                                                                 use this with a
                                                                 pre-populated
                                                                 recalibrated_spec)
"""
import os as _os

BUDGET_PROFILES: dict[str, dict[str, int]] = {
    "default": {"recalibrate": 0, "broad": 200, "refined": 49, "holdout": 1, "persist": 0},
    "refined24": {"recalibrate": 0, "broad": 24, "refined": 0, "holdout": 1, "persist": 0},
}

_PROFILE = _os.environ.get("DPE_BUDGET_PROFILE", "default")
if _PROFILE not in BUDGET_PROFILES:
    raise ValueError(f"unknown DPE_BUDGET_PROFILE={_PROFILE!r}; "
                     f"known: {list(BUDGET_PROFILES)}")

STAGE_SPLIT: dict[str, int] = dict(BUDGET_PROFILES[_PROFILE])
TOTAL_BUDGET: int = sum(STAGE_SPLIT.values())

# per-method overrides for stage budgets. methods listed here use a smaller
# trial budget for broad and/or refined; missing keys fall back to STAGE_SPLIT.
# rationale: FMDRE family (slow flow methods on `general` partition via lite
# watchdog) is bottlenecked on partition cap and per-trial wallclock; reduce
# their budget so the rest of the campaign can finish in a reasonable horizon.
# disabled when a non-default profile is active (the profile already encodes a
# tight budget and per-method tweaks would break the budget==total invariant).
_METHOD_BUDGET_OVERRIDES_DEFAULT: dict[str, dict[str, int]] = {
    "FMDRE":            {"broad": 40,  "refined": 10},
    "FMDRE_S2":         {"broad": 40,  "refined": 10},
    "TriangularFMDRE":  {"broad": 40,  "refined": 10},
    "VFM":              {"broad": 100, "refined": 25},
    "TriangularVFM_V1": {"broad": 100, "refined": 25},
    "TriangularVFM_V2": {"broad": 100, "refined": 25},
    "TriangularVFM_V3": {"broad": 100, "refined": 25},
}
METHOD_BUDGET_OVERRIDES: dict[str, dict[str, int]] = (
    _METHOD_BUDGET_OVERRIDES_DEFAULT if _PROFILE == "default" else {}
)


def stage_budget(stage: str, method: str | None = None,
                 total: int = TOTAL_BUDGET) -> int:
    """return trial count for one stage; honors per-method overrides.

    args:
      stage: one of "broad", "refined", "holdout", "recalibrate", "persist".
      method: optional method name; if listed in METHOD_BUDGET_OVERRIDES with
        an entry for this stage, that override is returned. otherwise the
        global STAGE_SPLIT value is returned.
      total: total budget (must equal TOTAL_BUDGET=250 in v1; no scaling).

    returns: integer trial count for this stage.
    raises ValueError if stage not in STAGE_SPLIT or total != TOTAL_BUDGET.
    """
    if total != TOTAL_BUDGET:
        raise ValueError(
            f"total must equal TOTAL_BUDGET ({TOTAL_BUDGET}); no scaling in v1"
        )

    if stage not in STAGE_SPLIT:
        raise ValueError(
            f"unknown stage: {stage}. must be one of {list(STAGE_SPLIT.keys())}"
        )

    if method and method in METHOD_BUDGET_OVERRIDES:
        overrides = METHOD_BUDGET_OVERRIDES[method]
        if stage in overrides:
            return overrides[stage]
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
