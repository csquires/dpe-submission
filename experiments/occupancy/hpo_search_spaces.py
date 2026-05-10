"""hpo search spaces for smodice eldr estimation.

universal registry includes all 20 methods (18 continuous + 2 tabular).
each entry is a dict with keys:
  search_space: param_name -> (spec_type, *bounds) tuple for hpo tuning.
  builder: callable(input_dim, device, num_waypoints, **flat_hp) -> estimator.
  requires_pstar: bool, whether fit() receives p* samples.
  num_waypoints: int or None.

legacy aliases auto-injected by registry:
  MHTTDRE -> MultiHeadTriangularTDRE
  MDRE -> MDRE_15
  TDRE -> TDRE_5
  TriangularCTSM -> TriangularCTSM_V1
  TriangularVFM -> TriangularVFM_V1

total: 25 keys (20 canonical + 5 aliases).
"""

from experiments.utils.hpo.registry import build_search_spaces


SEARCH_SPACES = build_search_spaces(include_tabular=True)
