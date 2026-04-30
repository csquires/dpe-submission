"""
registry-driven HPO method specifications for pendulum trajectory ELDR.
18 canonical continuous methods + 5 legacy aliases (23 total):
baselines (BDRE, MDRE_15, TSM, CTSM) and triangular/flow variants.
each entry is a dict keyed by method name with shape {search_space, builder, requires_pstar}.
search_space is param_name -> spec tuple (log_uniform/uniform/choice with bounds).
builder is a callable that constructs an estimator from hyperparams. requires_pstar
is bool indicating whether data loading/fitting must include p* samples.

NOTE: Pendulum is continuous-state (theta, theta_dot, action). NO encoding axis;
TabularPluginDRE and SmoothedTabularPluginDRE are excluded (apply only to smodice
tile-coded encoding). input_dim (= 3) and device are passed by hpo_trial, not in registry.

Eval cells: (k1_idx, k2_idx, seed) triplets from config.yaml::kl_targets.
Winner key: (k1_idx, k2_idx).
"""

from experiments.utils.hpo.registry import build_search_spaces

# tabular methods (TabularPluginDRE, SmoothedTabularPluginDRE) excluded:
# pendulum step1_create_data produces continuous (theta, theta_dot, action).
# no discrete tile-coded representation. tabular methods apply only to smodice.

SEARCH_SPACES = build_search_spaces(include_tabular=False)
