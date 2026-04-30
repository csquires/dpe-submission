"""
registry of HPO method specifications for dbpedia text embeddings (64-d SBERT→PCA).
overrides MultiHeadTriangularTDRE.hidden_dim to [64, 128, 256] for high-dimensional
latent codes; other 17 continuous methods and 5 aliases inherit canonical ranges
from experiments.utils.hpo.method_specs.METHOD_SPECS.
"""

from experiments.utils.hpo.registry import build_search_spaces


SEARCH_SPACES = build_search_spaces(
    include_tabular=False,
    overrides={
        "MultiHeadTriangularTDRE": {"hidden_dim": ("choice", [64, 128, 256])},
    },
)
