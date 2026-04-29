"""registry of 6 HPO method specifications for EIG estimation.

methods: BDRE, TDRE, MDRE, TriangularMDRE, TSM, VFM. each entry is keyed by
method name with shape {search_space, builder, requires_pstar}.

builders are imported from experiments.utils.hpo.builders (bare DRE estimators
returning .fit/.predict_ldr-shaped objects). hpo_trial.py wraps them with
EIGPlugin (and a triangular-fit adapter where needed) at metric-evaluation time.
this keeps the registry generic and reusable.

requires_pstar is False for all entries: EIGPlugin owns its own fit call and
does not load p* samples; the triangular adapter re-routes p0 as a stand-in
pstar for triangular methods inside hpo_trial.py.
"""

from experiments.utils.hpo.builders import (
    build_BDRE,
    build_TDRE,
    build_MDRE,
    build_TriangularMDRE,
    build_TSM,
    build_VFM,
)


SEARCH_SPACES = {
    # baseline: single binary classifier on (theta, y) joint vs marginal.
    "BDRE": {
        "search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
        "builder": build_BDRE,
        "requires_pstar": False,
    },

    # baseline: pairwise binary classifiers along the bridge path.
    "TDRE": {
        "search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
        "builder": build_TDRE,
        "requires_pstar": False,
    },

    # baseline: multi-class classifier; num_classes = num_waypoints.
    "MDRE": {
        "search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
        "builder": build_MDRE,
        "requires_pstar": False,
    },

    # proposed: triangular mdre with midpoint oversampling and power scaling.
    # under EIGPlugin, the triangular-fit adapter forwards p0 in the pstar slot.
    "TriangularMDRE": {
        "search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
            "midpoint_oversample": ("choice", [3, 5, 7]),
            "gamma_power": ("uniform", 1.0, 5.0),
        },
        "builder": build_TriangularMDRE,
        "requires_pstar": False,
    },

    # score-matching baseline; ranges shared with mnist registry.
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

    # variational flow matching baseline; ranges shared with mnist registry.
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
}
