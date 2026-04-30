"""
HPO search space registry for ELBO estimation experiment. all 18 continuous
methods plus 5 legacy aliases available for expansion. each entry holds
{search_space, builder, requires_pstar, num_waypoints} with canonical
ranges from experiments.utils.hpo.method_specs.METHOD_SPECS.
"""

from experiments.utils.hpo.registry import build_search_spaces


SEARCH_SPACES = build_search_spaces(include_tabular=False)
