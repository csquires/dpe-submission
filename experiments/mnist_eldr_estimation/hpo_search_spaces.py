"""
registry of HPO method specifications: consolidated from local builders into
universal registry-driven pattern. all 18 continuous methods (baselines +
triangular variants, including 5 newly added) and their search spaces are
defined in experiments.utils.hpo.method_specs.METHOD_SPECS. MHTTDRE is
renamed to MultiHeadTriangularTDRE; legacy alias MHTTDRE is auto-injected
by build_search_spaces() for transition compatibility.
"""

from experiments.utils.hpo.registry import build_search_spaces


SEARCH_SPACES = build_search_spaces(include_tabular=False)
