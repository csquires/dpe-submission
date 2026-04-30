"""
MNIST ELDR Cond-Flow HPO search spaces: delegates to mnist_eldr_estimation
for canonical method registry and search space definitions.
"""

from experiments.mnist_eldr_estimation.hpo_search_spaces import SEARCH_SPACES

__all__ = ["SEARCH_SPACES"]
