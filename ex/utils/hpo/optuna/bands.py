"""derive hyperband rung sequence from a studyconfig.

single source of truth for the (min_resource, max_resource, reduction_factor)
triple that defines optuna's hyperband pruner rungs. used by run_holdout.py
to align probe.top_k_at_each_budget's bands arg with the pruner's actual
rungs.
"""
from ex.utils.hpo.optuna.study_config import StudyConfig


def compute_bands(cfg: StudyConfig) -> list[int]:
    """return the rung resource sequence [min*rf**k for k in 0..K] where
    min * rf**K == max.

    raises:
        ValueError: if max_resource is not exactly min_resource * rf**K
            for any integer K.
    """
    min_r, max_r, rf = cfg.min_resource, cfg.max_resource, cfg.reduction_factor
    bands = []
    r = min_r
    while r <= max_r:
        bands.append(r)
        r *= rf
    if not bands or bands[-1] != max_r:
        raise ValueError(
            f"max_resource={max_r} is not min_resource={min_r} times "
            f"reduction_factor={rf} raised to an integer power; "
            f"computed bands={bands}. fix the config so they align "
            f"with hyperband rungs (e.g. min=400/max=6400/rf=2 or "
            f"min=900/max=24300/rf=3)."
        )
    return bands
