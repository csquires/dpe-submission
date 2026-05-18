from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import importlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class StudyConfig:
    """Per-study HPO configuration with validation.

    Holds hyperparameter search metadata, resource budgets, and I/O paths.
    Validation in __post_init__ ensures all constraints are satisfied.

    Attributes:
        study_seed: random seed for optuna sampler and trial determinism.
        experiment: experiment name, must be non-empty; not validated against
            registry here.
        methods: method names to optimize; validated against SUGGEST_HP_REGISTRY.
        walltime_minutes: wall-clock timeout per task (minutes); must >
            walltime_margin_minutes.
        min_resource: hyperband min budget steps; must <= max_resource.
        max_resource: hyperband max budget steps; used as step count in
            pruner config.
        reduction_factor: hyperband reduction factor; must > 1.
        holdout_top_k: number of trials to evaluate on holdout pool; must > 0.
        target_trials: per-study trial-count goal for the preempt keeper; the
            keeper stops dispatching once a study's COMPLETE+PRUNED count
            reaches it. must > 0.
        walltime_margin_minutes: buffer before hard timeout; must <
            walltime_minutes.
        nfs_base: root for journal files; defaults to $DPE_DATA_ROOT/optuna at
            runtime. If set, overrides env var lookup; must be absolute path
            string.
        cores_per_trial: per-method core count; if set, method names must be
            in registry. If None, uses cores_registry defaults.
        fixed_hp: optional dict of hyperparameter pins overlaid onto every
            trial's suggested hp (e.g. {"n_hidden_layers": 3}); experiment-level
            overrides for params removed from the per-method search space.
        resume_existing: load existing study journal if present, else start
            fresh.
        include_tabular: if True, methods may include tabular; warn if True
            but methods has no tabular.
        lanes: lane names the multi-lane keeper drains for this study; each must be a
            key in ex.utils.hpo.optuna.lanes.LANES. default ['preempt', 'array'].
        schema_version: config schema version; mismatch logs warning, does not
            error.
    """

    study_seed: int
    experiment: str
    methods: List[str]
    walltime_minutes: int

    min_resource: int = 100
    max_resource: int = 10000
    reduction_factor: int = 3
    holdout_top_k: int = 5
    target_trials: int = 200

    walltime_margin_minutes: int = 10
    nfs_base: Optional[str] = None
    cores_per_trial: Optional[Dict[str, int]] = None
    fixed_hp: Optional[Dict[str, Any]] = None

    resume_existing: bool = True
    include_tabular: bool = False
    lanes: List[str] = field(default_factory=lambda: ["preempt", "array"])
    schema_version: str = "1.0"

    def __post_init__(self):
        """Validate all fields after initialization.

        Coerce methods to list[str] if string. Cross-check method names against
        SUGGEST_HP_REGISTRY. Validate resource and timeout constraints.
        Raises ValueError if any constraint violated.
        """
        # coerce methods to list if string
        if isinstance(self.methods, str):
            self.methods = [self.methods]

        # assert experiment non-empty
        if not self.experiment or not self.experiment.strip():
            raise ValueError("experiment must be non-empty string")

        # validate method names exist in registry
        from ex.utils.hpo.suggest_hp import SUGGEST_HP_REGISTRY
        missing = [m for m in self.methods if m not in SUGGEST_HP_REGISTRY]
        if missing:
            raise ValueError(f"methods {missing} not in SUGGEST_HP_REGISTRY")

        # validate timeouts
        if self.walltime_minutes <= self.walltime_margin_minutes:
            raise ValueError(
                f"walltime_minutes ({self.walltime_minutes}) must > margin "
                f"({self.walltime_margin_minutes})"
            )

        # validate resource budget
        if self.min_resource > self.max_resource:
            raise ValueError(
                f"min_resource ({self.min_resource}) must <= max_resource "
                f"({self.max_resource})"
            )

        # validate reduction factor
        if self.reduction_factor <= 1:
            raise ValueError(
                f"reduction_factor must > 1, got {self.reduction_factor}"
            )

        # validate holdout_top_k
        if self.holdout_top_k <= 0:
            raise ValueError(f"holdout_top_k must > 0, got {self.holdout_top_k}")

        # validate target_trials
        if self.target_trials <= 0:
            raise ValueError(f"target_trials must > 0, got {self.target_trials}")

        # validate cores_per_trial if provided
        if self.cores_per_trial is not None:
            for method, cores in self.cores_per_trial.items():
                if method not in SUGGEST_HP_REGISTRY:
                    raise ValueError(
                        f"cores_per_trial key '{method}' not in "
                        f"SUGGEST_HP_REGISTRY"
                    )
                if cores <= 0:
                    raise ValueError(
                        f"cores_per_trial['{method}'] must > 0, got {cores}"
                    )

        # validate lanes exist in LANES registry
        from ex.utils.hpo.optuna.lanes import LANES
        offending = [lane for lane in self.lanes if lane not in LANES]
        if offending:
            known = list(LANES.keys())
            raise ValueError(
                f"lanes {offending} not in LANES registry; known lanes: {known}"
            )


def load_config(module_path: str) -> StudyConfig:
    """Load StudyConfig from a config module at runtime.

    Dynamically imports module at module_path and extracts module.CONFIG.
    Validates CONFIG is a StudyConfig instance.

    Args:
        module_path: import path, e.g. 'ex.configs.hpo.bdre_pilot'

    Returns:
        StudyConfig instance

    Raises:
        ModuleNotFoundError: if module does not exist
        AttributeError: if module lacks CONFIG attribute
        TypeError: if CONFIG is not a StudyConfig instance
    """
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"config module not found: {module_path}"
        ) from e

    if not hasattr(mod, "CONFIG"):
        raise AttributeError(f"{module_path} missing CONFIG attribute")

    config = mod.CONFIG
    if not isinstance(config, StudyConfig):
        raise TypeError(
            f"{module_path}.CONFIG is {type(config)}, expected StudyConfig"
        )

    return config


def resolve_combo(config: StudyConfig, combo_index: int) -> tuple[str, str]:
    """Resolve (experiment, method) pair from config and combo index.

    Builds cartesian product of (experiment, method) where method is from
    config.methods. Returns the pair at combo_index with wraparound via modulo.

    Single shared resolver used by submit.py and keeper.py to avoid duplicating
    logic for experiment-method pair construction.

    Args:
        config: StudyConfig instance with experiment and methods fields.
        combo_index: index into (experiment, method) product; wraparound via %.

    Returns:
        tuple of (experiment_name, method_name).

    Raises:
        ValueError: if config.methods is empty.
    """
    if not config.methods:
        raise ValueError("config.methods must not be empty")

    combos = [(config.experiment, m) for m in config.methods]
    return combos[combo_index % len(combos)]
