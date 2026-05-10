"""
re-export canonical search spaces for mnist_eldr via registry.

SEARCH_SPACES is built once from METHOD_SPECS with no experiment-specific
overrides. all 18 continuous methods + 5 legacy aliases are available.
"""

from experiments.utils.hpo.registry import build_search_spaces


SEARCH_SPACES = build_search_spaces(include_tabular=False)
