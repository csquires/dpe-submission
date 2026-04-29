"""hpo: shared hyperparameter optimization core for DRE experiments.

provides utilities for sampling hyperparameter configurations from search spaces
and a generic per-trial harness loop. per-experiment hpo_trial.py wires these
together with experiment-specific data loading, estimator construction, and
metric computation.
"""

from experiments.utils.hpo.sample import sample_param, gen_config
from experiments.utils.hpo.trial import parse_cells, cell_id, run_trial

__all__ = ["sample_param", "gen_config", "parse_cells", "cell_id", "run_trial"]
