"""
canonical method specifications for HPO registry.

consolidates all 20 hyperparameter optimization method definitions (18 continuous
DRE/score-based + 2 tabular oracle plug-in) from distributed experiments into a
single source of truth. each entry maps a method name to its builder function,
oracle requirement, num_waypoints (if applicable), search space, and tabular flag.

consumed by experiments/utils/hpo/registry.py, which exposes both canonical names
and alias shortcuts for convenience.
"""

from experiments.utils.hpo.builders import (
    build_TSM,
    build_CTSM,
    build_VFM,
    build_BDRE,
    build_MDRE,
    build_TDRE,
    build_FMDRE,
    build_FMDRE_S2,
    build_TriangularFMDRE,
    build_TriangularTSM,
    build_MHTTDRE,
    build_TriangularMDRE,
    build_TriangularCTSM_V1,
    build_TriangularCTSM_V2,
    build_TriangularCTSM_V3,
    build_TriangularVFM_V1,
    build_TriangularVFM_V2,
    build_TriangularVFM_V3,
    build_TabularPluginDRE,
    build_SmoothedTabularPluginDRE,
)


METHOD_SPECS = {
    # baseline: time score matching (unconditional)
    "TSM": {
        "builder": build_TSM,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "eps": ("log_uniform", 1e-6, 1e-4),
        },
        "tabular_only": False,
    },
    # baseline: conditional time score matching
    "CTSM": {
        "builder": build_CTSM,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "eps": ("log_uniform", 3e-4, 3e-3),
        },
        "tabular_only": False,
    },
    # baseline: variational flow matching
    "VFM": {
        "builder": build_VFM,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "k": ("choice", [10, 20, 40]),
            "eps": ("log_uniform", 1e-3, 5e-3),
            "integration_steps": ("uniform_int", 500, 3000),
        },
        "tabular_only": False,
    },
    # baseline: binary density ratio estimation
    "BDRE": {
        "builder": build_BDRE,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
        "tabular_only": False,
    },
    # baseline: multiclass DRE (15 waypoints fixed)
    "MDRE_15": {
        "builder": build_MDRE,
        "requires_pstar": False,
        "num_waypoints": 15,
        "base_search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
        "tabular_only": False,
    },
    # baseline: triangular DRE (5 waypoints fixed)
    "TDRE_5": {
        "builder": build_TDRE,
        "requires_pstar": False,
        "num_waypoints": 5,
        "base_search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
        "tabular_only": False,
    },
    # baseline: flow-based multivariate DRE
    "FMDRE": {
        "builder": build_FMDRE,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            "n_epochs": ("log_uniform_int", 500, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [128, 256, 512]),
            "eps": ("log_uniform", 1e-3, 5e-2),
            "integration_steps": ("uniform_int", 1000, 3000),
            "hidden_dim": ("choice", [128, 256, 512]),
            "score_weight": ("log_uniform", 0.1, 10.0),
        },
        "tabular_only": False,
    },
    # baseline: FMDRE with unconditional score weighting
    "FMDRE_S2": {
        "builder": build_FMDRE_S2,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            "n_epochs": ("log_uniform_int", 500, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [128, 256, 512]),
            "eps": ("log_uniform", 1e-3, 5e-2),
            "integration_steps": ("uniform_int", 1000, 3000),
            "hidden_dim": ("choice", [128, 256, 512]),
            "score_weight": ("log_uniform", 0.1, 10.0),
            "p_uncond": ("uniform", 0.1, 0.9),
        },
        "tabular_only": False,
    },
    # baseline: oracle empirical tabular plug-in (state-action counting)
    "TabularPluginDRE": {
        "builder": build_TabularPluginDRE,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {},
        "tabular_only": True,
    },
    # baseline: oracle tabular plug-in with gaussian smoothing
    "SmoothedTabularPluginDRE": {
        "builder": build_SmoothedTabularPluginDRE,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {},
        "tabular_only": True,
    },
    # triangular: FMDRE with p0 -> p* -> p1 path
    "TriangularFMDRE": {
        "builder": build_TriangularFMDRE,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_epochs": ("log_uniform_int", 500, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [128, 256, 512]),
            "eps": ("log_uniform", 1e-3, 5e-2),
            "integration_steps": ("uniform_int", 1000, 3000),
            "hidden_dim": ("choice", [128, 256, 512]),
            "score_weight": ("log_uniform", 0.1, 10.0),
        },
        "tabular_only": False,
    },
    # triangular: TSM with triangular path
    "TriangularTSM": {
        "builder": build_TriangularTSM,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "hidden_dim": ("choice", [128, 256, 512]),
            "eps": ("log_uniform", 1e-5, 1e-3),
            "vertex": ("uniform", 0.2, 0.8),
            "peak_max": ("uniform", 0.5, 1.0),
        },
        "tabular_only": False,
    },
    # triangular: multi-head binary classifiers per waypoint edge
    "MultiHeadTriangularTDRE": {
        "builder": build_MHTTDRE,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 1000),
            "hidden_dim": ("choice", [16, 32, 64, 128]),
            "head_dim": ("choice", [10, 20, 40]),
            "num_shared_layers": ("choice", [1, 2, 3]),
        },
        "tabular_only": False,
    },
    # triangular: multiclass classifier with triangular weighting
    "TriangularMDRE": {
        "builder": build_TriangularMDRE,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
            "midpoint_oversample": ("choice", [3, 5, 7]),
            "gamma_power": ("uniform", 1.0, 5.0),
        },
        "tabular_only": False,
    },
    # triangular: piecewise-SB path with vertex kink point
    "TriangularCTSM_V1": {
        "builder": build_TriangularCTSM_V1,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "vertex": ("uniform", 0.2, 0.8),
            "eps": ("log_uniform", 1e-3, 3e-3),
        },
        "tabular_only": False,
    },
    # triangular: barycentric path (fixed vertex = 0.5)
    "TriangularCTSM_V2": {
        "builder": build_TriangularCTSM_V2,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "eps": ("log_uniform", 3e-4, 3e-3),
        },
        "tabular_only": False,
    },
    # triangular: 2D stacked path with curvature params
    "TriangularCTSM_V3": {
        "builder": build_TriangularCTSM_V3,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "gamma_schedule": ("choice", ["sqrt", "linear-stiff"]),
            "k": ("choice", [12, 24, 48]),
            "t2_max": ("uniform", 0.6, 0.9),
            "eps": ("log_uniform", 1e-3, 3e-3),
            "path_height": ("uniform", 1.0, 2.0),
        },
        "tabular_only": False,
    },
    # triangular: piecewise-SB path with gamma_min for hamiltonian weighting
    "TriangularVFM_V1": {
        "builder": build_TriangularVFM_V1,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "vertex": ("uniform", 0.2, 0.8),
            "gamma_min": ("log_uniform", 1e-2, 1e-1),
            "eps": ("log_uniform", 1e-3, 5e-3),
            "integration_steps": ("uniform_int", 500, 3000),
        },
        "tabular_only": False,
    },
    # triangular: barycentric path (fixed vertex = 0.5)
    "TriangularVFM_V2": {
        "builder": build_TriangularVFM_V2,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "k": ("choice", [10, 20, 40]),
            "eps": ("log_uniform", 1e-3, 5e-3),
            "integration_steps": ("uniform_int", 500, 3000),
        },
        "tabular_only": False,
    },
    # triangular: 2D stacked path with curvature params
    "TriangularVFM_V3": {
        "builder": build_TriangularVFM_V3,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "k": ("choice", [10, 20, 40]),
            "gamma_schedule": ("choice", ["linear-stiff"]),
            "t2_max": ("uniform", 0.6, 0.9),
            "eps": ("log_uniform", 1e-3, 5e-3),
            "path_height": ("uniform", 1.0, 2.0),
            "integration_steps": ("uniform_int", 500, 3000),
        },
        "tabular_only": False,
    },
}

# alias pairs for registry: (short_name, canonical_name)
ALIAS_PAIRS = [
    ("MHTTDRE", "MultiHeadTriangularTDRE"),
    ("MDRE", "MDRE_15"),
    ("TDRE", "TDRE_5"),
    ("TriangularCTSM", "TriangularCTSM_V1"),
    ("TriangularVFM", "TriangularVFM_V1"),
]
