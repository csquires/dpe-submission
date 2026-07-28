"""Base factory for dokls HPO configs.

builds StudyConfig for any (variant_tag, tier, route) triple. shared settings
extracted from dokls requirements. n_steps pinned 6400 per ablation spec. route
is a first-class axis: two_leg and direct are fully separate experiments (own
StudyConfig, own optuna study, own holdout/winners), never a per-trial hp
choice within a shared experiment.

intended use: thin client config modules import make() and set
CONFIG = make(tag=..., tier=..., route=...).
"""

from ex.utils.hpo.optuna.study_config import StudyConfig


CPU_METHODS = [
    "BDRE",
    "BDRE_NWJ",
    "BDRE_DV",
    "MultiHeadTDRE",
    "MHT_NWJ",
    "MHT_DV",
    "CTSM",
    "TSM",
]

GPU_METHODS = [
    "VFM",
    "FMDRE",
]

# invariant: keeper done-markers keyed by config module path, not experiment
# name. cpu and gpu tiers share experiment=f"dokls_{route}_{tag}" safely:
# method lists are disjoint.
assert not (set(CPU_METHODS) & set(GPU_METHODS)), \
    "cpu and gpu method lists must be disjoint; isolation scheme depends on it"


TIER_DEFAULTS = {
    "cpu": {
        "methods": CPU_METHODS,
        "lanes": ["array"],
    },
    "gpu": {
        "methods": GPU_METHODS,
        "lanes": ["array_gpu_wide", "general", "preempt"],
        # holdout defaults to the cpu "holdout" lane (gpus=0); pin the gpu tier
        # to a gpu lane or VFM/FMDRE holdout silently grinds on cpu. preempt
        # gives the widest fanout (B=32 x max_concurrent=24).
        "holdout_lane": "preempt",
    },
}

SHARED_SETTINGS = {
    "study_seed": 1729,
    "target_trials": 256,
    "slices": None,
    "walltime_minutes": 120,
    "walltime_margin_minutes": 10,
    "min_resource": 400,
    "max_resource": 6400,
    "reduction_factor": 2,
    "holdout_top_k": 5,
    "fixed_hp": {"n_hidden_layers": 5},
    "gate_gpu_methods": False,
    "resume_existing": True,
    "include_tabular": False,
}


def make(tag: str, tier: str, route: str, lanes: list[str] | None = None) -> StudyConfig:
    """Build StudyConfig for dokls HPO.

    Args:
        tag: variant tag (validated against ex.ablations.dokls.variants.VARIANTS).
        tier: "cpu" or "gpu"; selects method list and default lanes.
        route: "two_leg" or "direct" (validated against variants.ROUTES); baked
            into the experiment name so the two routes never share a study.
        lanes: override default lanes for this tier. if None, uses tier default.

    Returns:
        StudyConfig instance with experiment=f"dokls_{route}_{tag}".

    Raises:
        ValueError: if tier not in ("cpu", "gpu"), route not in variants.ROUTES,
            or tag not in VARIANTS.
    """
    # validate tier
    if tier not in TIER_DEFAULTS:
        raise ValueError(f"tier must be 'cpu' or 'gpu', got {tier!r}")

    # validate tag and route against variants registry
    from ex.ablations.dokls.variants import VARIANTS, ROUTES, experiment_name
    if tag not in VARIANTS:
        raise ValueError(
            f"tag {tag!r} not in VARIANTS; known tags: {list(VARIANTS.keys())}"
        )
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, got {route!r}")

    # select methods and default lanes by tier
    tier_cfg = TIER_DEFAULTS[tier]
    methods = tier_cfg["methods"]
    lanes_to_use = lanes if lanes is not None else tier_cfg["lanes"]

    # build StudyConfig with shared settings + tier-specific overrides
    config_dict = {
        "experiment": experiment_name(tag, route),
        "methods": methods,
        "lanes": lanes_to_use,
        "holdout_lane": tier_cfg.get("holdout_lane", "holdout"),
        **SHARED_SETTINGS,
    }

    return StudyConfig(**config_dict)
