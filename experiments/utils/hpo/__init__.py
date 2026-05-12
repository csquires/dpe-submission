"""hpo: hyperparameter optimization for DRE experiments.

active subpackages:
- experiments.utils.hpo.adapters: per-experiment data and metric adapters.
- experiments.utils.hpo.optuna: optuna-based HPO driver (storage, objective,
  worker, submit, study_config, cores_registry, probe, holdout, figures, configs).
- experiments.utils.hpo.suggest_hp: per-method optuna suggest_hp functions and
  METADATA registry.

the legacy random-search infrastructure (sample, trial, trial_runner, registry,
narrow, cell_schema, budget, cpu_*, launcher*, cli, workflow*) has been removed.
import the optuna stack directly via `from experiments.utils.hpo.optuna import ...`.
"""
