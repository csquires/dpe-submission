from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Hashable
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
        target_trials: per-study budget of COMPLETE trials. scalar int (all methods
            share budget, default 320) or dict[str, int] (per-method budgets).
            dict form requires keys == set(methods) exactly. the keeper stops
            dispatching once a study's COMPLETE count reaches the per-method
            target, and workers self-stop via MaxTrialsCallback. scalar or dict
            values must > 0. accessor: target_for(method) -> int.
        slices: optional list of hashable stratify-key values for per-hardness-slice
            studies. one set of HPs per slice value; study names and redis prefixes
            include slice suffix when set. if provided, must be non-empty with
            no duplicates. defaults to None (full pool, no stratification).
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
        max_in_flight: cap on concurrent trials per (experiment, method) summed
            across lanes (per-lane B * n_active_jobs). keeper-enforced; prevents
            journal-lock contention when the array lane (B=32) is fully loaded
            on a slow-load journal. default 256.
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
    target_trials: int | Dict[str, int] = 320
    slices: Optional[List[Hashable]] = None

    walltime_margin_minutes: int = 10
    nfs_base: Optional[str] = None
    cores_per_trial: Optional[Dict[str, int]] = None
    fixed_hp: Optional[Dict[str, Any]] = None

    resume_existing: bool = True
    include_tabular: bool = False
    lanes: List[str] = field(default_factory=lambda: ["preempt", "array"])
    max_in_flight: int = 256
    schema_version: str = "1.0"

    def target_for(self, method: str) -> int:
        """Return the target trial count for a given method.

        If target_trials is an int, returns it as-is (all methods share the same
        budget). If target_trials is a dict, returns target_trials[method] (per-method
        budget). Raises KeyError with a helpful message if the method is not in the
        dict.

        Args:
            method: method name (must be a key in self.methods).

        Returns:
            int: target trial count for this method.

        Raises:
            KeyError: if target_trials is a dict and method is not a key.
        """
        if isinstance(self.target_trials, int):
            return self.target_trials
        else:
            # target_trials is a dict[str, int]
            try:
                return self.target_trials[method]
            except KeyError:
                known = list(self.target_trials.keys())
                raise KeyError(
                    f"method '{method}' not in target_trials dict; "
                    f"known methods: {known}"
                ) from None

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
        if isinstance(self.target_trials, int):
            # scalar form: must be > 0
            if self.target_trials <= 0:
                raise ValueError(
                    f"target_trials (scalar) must > 0, got {self.target_trials}"
                )
        else:
            # dict form: keys must be exactly self.methods, all values > 0
            if not isinstance(self.target_trials, dict):
                raise ValueError(
                    f"target_trials must be int or dict[str, int], "
                    f"got {type(self.target_trials)}"
                )

            target_keys = set(self.target_trials.keys())
            method_keys = set(self.methods)

            if target_keys != method_keys:
                missing = method_keys - target_keys
                extra = target_keys - method_keys
                msg = "target_trials dict keys must equal set(self.methods) exactly; "
                if missing:
                    msg += f"missing: {missing}; "
                if extra:
                    msg += f"extra: {extra}"
                raise ValueError(msg)

            for method, count in self.target_trials.items():
                if not isinstance(count, int) or count <= 0:
                    raise ValueError(
                        f"target_trials['{method}'] must be int > 0, "
                        f"got {count} (type {type(count).__name__})"
                    )

        # validate max_in_flight
        if self.max_in_flight <= 0:
            raise ValueError(
                f"max_in_flight must > 0, got {self.max_in_flight}"
            )

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

        # validate slices
        if self.slices is not None:
            # slices must be a non-empty list with no duplicates, all hashable
            if not isinstance(self.slices, list):
                raise ValueError(
                    f"slices must be None or list, got {type(self.slices)}"
                )

            if len(self.slices) == 0:
                raise ValueError("slices must be non-empty if provided (not None)")

            # check all elements are hashable (try to hash them)
            try:
                hashed = set(self.slices)
            except TypeError as e:
                raise ValueError(
                    f"all slices elements must be hashable; error: {e}"
                ) from e

            # check no duplicates
            if len(hashed) != len(self.slices):
                raise ValueError(
                    f"slices contains duplicates; must be unique. "
                    f"got {len(self.slices)} items, {len(hashed)} unique"
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


def resolve_combo(
    config: StudyConfig, combo_index: int
) -> tuple[str, str, Optional[Hashable]]:
    """Resolve (experiment, method, slice) from config and combo index.

    method-major cross product:
      [(config.experiment, m, s) for m in config.methods
       for s in (config.slices or [None])]
    returned at combo_index with wraparound via modulo.

    when config.slices is None, slice is always None and the product
    reduces to today's (experiment, method) list with an appended None.

    keeper and submit.py both call this with the same combo_index to
    agree on (method, slice) selection.

    Args:
        config: StudyConfig instance with experiment, methods, slices fields.
        combo_index: index into the product; wraparound via %.

    Returns:
        tuple of (experiment_name, method_name, slice_value_or_None).

    Raises:
        ValueError: if config.methods is empty.
    """
    if not config.methods:
        raise ValueError("config.methods must not be empty")

    slices = config.slices if config.slices is not None else [None]
    combos = [
        (config.experiment, m, s)
        for m in config.methods
        for s in slices
    ]
    return combos[combo_index % len(combos)]
