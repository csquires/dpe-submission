"""
registry of 12 HPO method specifications: baselines (TSM, CTSM, VFM, FMDRE,
FMDRE_S2, MHTTDRE) and triangular variants (TriangularCTSM_V1-V3,
TriangularVFM_V1-V3). each entry is a dict keyed by method name with shape
{search_space, builder, requires_pstar}. search_space is param_name -> spec tuple
(log_uniform/uniform/choice with bounds). builder is a callable that constructs
an estimator from hyperparams. requires_pstar is bool indicating whether data
loading/fitting must include p* samples.
"""

from experiments.mnist_eldr_estimation.hpo_builders import (
    build_TSM,
    build_CTSM,
    build_VFM,
    build_FMDRE,
    build_FMDRE_S2,
    build_TriangularFMDRE,
    build_MHTTDRE,
    build_TriangularCTSM_V1,
    build_TriangularCTSM_V2,
    build_TriangularCTSM_V3,
    build_TriangularVFM_V1,
    build_TriangularVFM_V2,
    build_TriangularVFM_V3,
)


SEARCH_SPACES = {
    # baseline methods (do not require p*)
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
    "VFM": {
        "search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "k": ("choice", [10, 20, 40]),
            "eps": ("log_uniform", 1e-3, 5e-3),
            "integration_steps": ("uniform_int", 500, 5000),
        },
        "builder": build_VFM,
        "requires_pstar": False,
    },
    "FMDRE": {
        "search_space": {
            "n_epochs": ("log_uniform_int", 500, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [128, 256, 512]),
            "eps": ("log_uniform", 1e-3, 5e-2),
            "integration_steps": ("uniform_int", 1000, 10000),
            "hidden_dim": ("choice", [128, 256, 512]),
            "score_weight": ("log_uniform", 0.1, 10.0),
        },
        "builder": build_FMDRE,
        "requires_pstar": False,
    },
    "FMDRE_S2": {
        "search_space": {
            "n_epochs": ("log_uniform_int", 500, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [128, 256, 512]),
            "eps": ("log_uniform", 1e-3, 5e-2),
            "integration_steps": ("uniform_int", 1000, 10000),
            "hidden_dim": ("choice", [128, 256, 512]),
            "score_weight": ("log_uniform", 0.1, 10.0),
            "p_uncond": ("uniform", 0.1, 0.9),
        },
        "builder": build_FMDRE_S2,
        "requires_pstar": False,
    },
    "TriangularFMDRE": {
        "search_space": {
            "n_epochs": ("log_uniform_int", 500, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [128, 256, 512]),
            "eps": ("log_uniform", 1e-3, 5e-2),
            "integration_steps": ("uniform_int", 1000, 10000),
            "hidden_dim": ("choice", [128, 256, 512]),
            "score_weight": ("log_uniform", 0.1, 10.0),
        },
        "builder": build_TriangularFMDRE,
        "requires_pstar": True,
    },
    "MHTTDRE": {
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
    # triangular variants (require p*)
    "TriangularCTSM_V1": {
        "search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "vertex": ("uniform", 0.2, 0.8),
            "eps": ("log_uniform", 1e-3, 3e-3),
        },
        "builder": build_TriangularCTSM_V1,
        "requires_pstar": True,
    },
    "TriangularCTSM_V2": {
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
    "TriangularCTSM_V3": {
        "search_space": {
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
        "builder": build_TriangularCTSM_V3,
        "requires_pstar": True,
    },
    "TriangularVFM_V1": {
        "search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "vertex": ("uniform", 0.2, 0.8),
            "gamma_min": ("log_uniform", 1e-2, 1e-1),
            "eps": ("log_uniform", 1e-3, 5e-3),
            "integration_steps": ("uniform_int", 500, 5000),
        },
        "builder": build_TriangularVFM_V1,
        "requires_pstar": True,
    },
    "TriangularVFM_V2": {
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
    "TriangularVFM_V3": {
        "search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "k": ("choice", [10, 20, 40]),
            "gamma_schedule": ("choice", ["linear-stiff"]),
            "t2_max": ("uniform", 0.6, 0.9),
            "eps": ("log_uniform", 1e-3, 5e-3),
            "path_height": ("uniform", 1.0, 2.0),
            "integration_steps": ("uniform_int", 500, 5000),
        },
        "builder": build_TriangularVFM_V3,
        "requires_pstar": True,
    },
}
