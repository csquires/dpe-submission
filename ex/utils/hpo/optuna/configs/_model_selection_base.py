"""Base factory for model_selection HPO configs; replaces 8 thin modules.

builds StudyConfig for any (variant_tag, tier) pair.

intended use: thin client config modules import make() and set
CONFIG = make(tag=..., tier=...).
"""

from ex.utils.hpo.optuna.study_config import StudyConfig


CPU_METHODS = [
    "BDRE",
    "MDRE",
    "MultiHeadTDRE",
    "TriangularMDRE",
    "MultiHeadTriangularTDRE",
    "CTSM",
    "TSM",
    "TriangularCTSM_V1",
    "TriangularCTSM_V2",
    "TriangularCTSM_V3",
    "TriangularTSM",
]

GPU_METHODS = [
    "VFM",
    "TriangularVFM_V1",
    "TriangularVFM_V2",
    "TriangularVFM_V3",
    "FMDRE",
    "FMDRE_S2",
    "TriangularFMDRE",
]

# invariant: keeper done-markers are keyed by config module path, not the
# experiment name. therefore both cpu and gpu tiers share experiment="model_selection_{tag}"
# safely: their method lists are disjoint, keeper will never mis-route a cell.
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
    },
}

SHARED_SETTINGS = {
    "study_seed": 1729,
    "target_trials": 512,
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


def make(tag: str, tier: str, lanes: list[str] | None = None) -> StudyConfig:
    """Build StudyConfig for model_selection HPO.

    Args:
        tag: variant tag (validated against ex.synth.model_selection.variants.VARIANTS).
        tier: "cpu" or "gpu"; selects method list and default lanes.
        lanes: override default lanes for this tier. if None, uses tier default.

    Returns:
        StudyConfig instance with experiment=f"model_selection_{tag}".

    Raises:
        ValueError: if tier not in ("cpu", "gpu") or tag not in VARIANTS.
    """
    # validate tier
    if tier not in TIER_DEFAULTS:
        raise ValueError(f"tier must be 'cpu' or 'gpu', got {tier!r}")

    # validate tag against VARIANTS registry
    from ex.synth.model_selection.variants import VARIANTS
    if tag not in VARIANTS:
        raise ValueError(
            f"tag {tag!r} not in VARIANTS; known tags: {list(VARIANTS.keys())}"
        )

    # select methods and default lanes by tier
    tier_cfg = TIER_DEFAULTS[tier]
    methods = tier_cfg["methods"]
    lanes_to_use = lanes if lanes is not None else tier_cfg["lanes"]

    # build StudyConfig with shared settings + tier-specific overrides
    config_dict = {
        "experiment": f"model_selection_{tag}",
        "methods": methods,
        "lanes": lanes_to_use,
        **SHARED_SETTINGS,
    }

    return StudyConfig(**config_dict)
