"""
registry of 18+5-alias HPO method specifications for model_selection (data_dim=3).

18 canonical continuous methods from METHOD_SPECS with overrides for BDRE/MDRE_15/TDRE_5
(tightened search ranges: latent_dim=[64,128,256], num_epochs=[100,500]) tuned for
small Gaussian pair data. also includes 5 legacy aliases from LEGACY_ALIASES.

each entry: {search_space: dict, builder: callable, requires_pstar: bool,
num_waypoints: int|None}.
"""

from experiments.utils.hpo.registry import build_search_spaces

SEARCH_SPACES = build_search_spaces(
    include_tabular=False,
    overrides={
        "BDRE": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
        "MDRE_15": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
        "TDRE_5": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
    },
)
