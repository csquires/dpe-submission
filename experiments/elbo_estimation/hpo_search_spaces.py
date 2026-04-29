"""
registry of six HPO method specifications (BDRE, TDRE_5, MDRE_15, TSM,
TriangularMDRE, VFM) for elbo estimation experiment. eval cell shape:
(alpha_idx, design_idx); winner-key is alpha_idx. step2_run_algorithms.py
hardcodes these six algorithms (lines 74-81) and dispatches fit(p0, p1) for
non-triangular and fit(p0, p1, pstar) for TriangularMDRE (lines 86-87). each
entry is a dict with shape {search_space, builder, requires_pstar,
[num_waypoints]}. search_space is param_name -> spec tuple (type, bounds).
builder is a callable that constructs an estimator from flat hyperparams.
requires_pstar is bool indicating whether fit() takes 3 args (pstar samples).
num_waypoints is elbo-specific extension (TDRE_5=5, MDRE_15=15, TriangularMDRE=15).
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
    # dre methods: shared base hyperparameter space (latent_dim, learning_rate,
    # num_epochs); ranges span shallow to moderate network capacity and typical
    # dre tuning regimes. num_epochs [100, 500] sufficient for convergence on
    # modest datasets.
    "BDRE": {
        "search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
        "builder": build_BDRE,
        "requires_pstar": False,
    },
    # TDRE with fixed num_waypoints=5 (step2 line 45: TDRE_WAYPOINTS=5). search
    # space identical to BDRE; num_waypoints supplied by registry (not hpo'd).
    "TDRE_5": {
        "search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
        "builder": build_TDRE,
        "requires_pstar": False,
        "num_waypoints": 5,
    },
    # MDRE with fixed num_waypoints=15 (step2 line 46: MDRE_WAYPOINTS=15).
    # search space identical to BDRE; num_waypoints supplied by registry (not
    # hpo'd).
    "MDRE_15": {
        "search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
        "builder": build_MDRE,
        "requires_pstar": False,
        "num_waypoints": 15,
    },
    # score-based method; higher epoch/batch_size ranges reflect different
    # train scaling vs dre methods. copied from mnist canonical registry.
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
    # triangular mdre: extends BDRE base params with midpoint_oversample (density
    # of intermediate waypoints; [3, 5, 7]) and gamma_power (curvature/schedule
    # of density ratio along path; [1.0, 5.0] uniform to allow linear schedules).
    # num_waypoints=15 (step2 line 61). requires_pstar=True (step2 fit dispatch
    # lines 86-87 calls fit(p0, p1, pstar)).
    "TriangularMDRE": {
        "search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
            "midpoint_oversample": ("choice", [3, 5, 7]),
            "gamma_power": ("uniform", 1.0, 5.0),
        },
        "builder": build_TriangularMDRE,
        "requires_pstar": True,
        "num_waypoints": 15,
    },
    # variational flow method; k: number of flow steps, integration_steps: ode
    # solver granularity. copied from mnist canonical registry.
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
