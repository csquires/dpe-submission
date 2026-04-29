"""
registry of 8 HPO method specifications for pendulum trajectory ELDR: baselines
(BDRE, MDRE, TSM, CTSM) and triangular variants (TriangularMDRE, MultiHeadTriangularTDRE,
TriangularCTSM, TriangularVFM). each entry is a dict keyed by method name with shape
{search_space, builder, requires_pstar}. search_space is param_name -> spec tuple
(log_uniform/uniform/choice with bounds). builder is a callable that constructs an
estimator from hyperparams. requires_pstar is bool indicating whether data loading/fitting
must include p* samples.

NOTE: Pendulum is continuous-state (theta, theta_dot, action). NO encoding axis;
TabularPluginDRE and SmoothedTabularPluginDRE are excluded (apply only to smodice
tile-coded encoding). input_dim (= 3) and device are passed by hpo_trial, not in registry.

Eval cells: (k1_idx, k2_idx, seed) triplets from config.yaml::kl_targets.
Winner key: (k1_idx, k2_idx).
"""

from experiments.utils.hpo.builders import (
    build_BDRE,
    build_MDRE,
    build_TriangularMDRE,
    build_MHTTDRE,
    build_TSM,
    build_CTSM,
    build_TriangularCTSM_V2,
    build_TriangularVFM_V2,
)

# tabular methods (TabularPluginDRE, SmoothedTabularPluginDRE) excluded:
# pendulum step1_create_data produces continuous (theta, theta_dot, action).
# no discrete tile-coded representation. tabular methods apply only to smodice.

SEARCH_SPACES = {
    "BDRE": {
        "search_space": {
            "latent_dim": ("choice", [64, 128, 256, 512]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 1000),
        },
        "builder": build_BDRE,
        "requires_pstar": False,
    },
    "MDRE": {
        "search_space": {
            "latent_dim": ("choice", [64, 128, 256, 512]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 1000),
        },
        "builder": build_MDRE,
        "requires_pstar": False,
    },
    "TriangularMDRE": {
        "search_space": {
            "latent_dim": ("choice", [64, 128, 256, 512]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 1000),
            "midpoint_oversample": ("choice", [1.0, 1.5, 2.0, 3.0]),
            "gamma_power": ("uniform", 0.5, 2.0),
        },
        "builder": build_TriangularMDRE,
        "requires_pstar": True,
    },
    "MultiHeadTriangularTDRE": {
        "search_space": {
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 1000),
            "hidden_dim": ("choice", [16, 32, 64, 128]),
            "head_dim": ("choice", [10, 20, 40]),
            "num_shared_layers": ("choice", [1, 2, 3]),
        },
        "builder": build_MHTTDRE,
        "requires_pstar": True,
    },
    "TSM": {
        "search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "eps": ("log_uniform", 1e-6, 1e-4),
        },
        "builder": build_TSM,
        "requires_pstar": False,
    },
    "CTSM": {
        "search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "eps": ("log_uniform", 3e-4, 3e-3),
        },
        "builder": build_CTSM,
        "requires_pstar": False,
    },
    "TriangularCTSM": {
        "search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "eps": ("log_uniform", 3e-4, 3e-3),
        },
        "builder": build_TriangularCTSM_V2,
        "requires_pstar": True,
    },
    "TriangularVFM": {
        "search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "k": ("choice", [10, 20, 40]),
            "eps": ("log_uniform", 1e-3, 5e-3),
            "integration_steps": ("uniform_int", 500, 5000),
        },
        "builder": build_TriangularVFM_V2,
        "requires_pstar": True,
    },
}
