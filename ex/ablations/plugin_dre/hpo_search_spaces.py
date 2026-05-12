"""
HPO registry for plugin_dre.

This experiment uses exactly the same DRE methods and hyperparameter search
spaces as dre_sample_complexity; only the evaluation target differs
(uniform-grid plugin estimation instead of sample-based LDR prediction).
"""

from ex.dre_sample_complexity.hpo_search_spaces import SEARCH_SPACES

