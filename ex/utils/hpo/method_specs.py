"""
canonical method specifications for HPO registry.

consolidates all 20 hyperparameter optimization method definitions (18 continuous
DRE/score-based + 2 tabular oracle plug-in) from distributed experiments into a
single source of truth. each entry maps a method name to its builder function,
oracle requirement, num_waypoints (if applicable), search space, and tabular flag.

consumed by ex/utils/hpo/registry.py, which exposes both canonical names
and alias shortcuts for convenience.
"""

from ex.utils.hpo.builders import (
    build_TSM,
    build_CTSM,
    build_VFM,
    build_VFMOrthros,
    build_BDRE,
    build_MDRE,
    build_MHTDRE,
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
from src.methods.reg.common._time_samplers import TIME_DISTS


METHOD_SPECS = {
    # baseline: time score matching (unconditional)
    "TSM": {
        "builder": build_TSM,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            "n_steps": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "eps": ("log_uniform", 1e-6, 1e-4),
            "integration_steps": ("uniform_int", 300, 2600),
            # tier-1 estimator scalars
            "hidden_dim": ("choice", [128, 256, 512]),
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "activation": ("choice", ["elu", "gelu", "silu"]),
            "reweight": ("choice", [False, True]),
            # importance sampling time distribution knob
            "time_dist": ("choice", list(TIME_DISTS)),
            # ema / grad-clip pilot knobs
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
            "apply_iw": ("choice", [True, False]),
        },
        "tabular_only": False,
    },
    # baseline: conditional time score matching
    "CTSM": {
        "builder": build_CTSM,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            "n_steps": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            # eps lower bound raised to 1e-3: it now feeds the general direct_1d
            # path builder, whose construction enforces eps >= 1e-3.
            "eps": ("log_uniform", 1e-3, 3e-3),
            "integration_steps": ("uniform_int", 300, 2600),
            # pilot fix knobs (defaults None preserve current behavior)
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            # importance sampling time distribution knob
            "time_dist": ("choice", list(TIME_DISTS)),
            # activation knob for score network
            "activation": ("choice", ["elu", "gelu", "silu"]),
            # noise schedule: type + amplitude, train + test path
            "sched": ("choice", ["stiff", "bridge"]),
            "k": ("choice", [10, 20, 40]),
            "gamma_min": ("log_uniform", 1e-2, 2e-1),
            "inner_eps": ("choice", [0.0, 0.05, 0.1]),
            "test_eps": ("log_uniform", 1e-3, 1e-1),
            "test_sched": ("choice", ["stiff", "bridge"]),
            "test_sigma": ("log_uniform", 0.3, 3.0),
            "test_gamma_min": ("log_uniform", 1e-2, 2e-1),
            "test_inner_eps": ("choice", [0.0, 0.05, 0.1]),
            # tier-1 estimator scalars
            "hidden_dim": ("choice", [128, 256, 512]),
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "reweight": ("choice", [False, True]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
            "apply_iw": ("choice", [True, False]),
        },
        "tabular_only": False,
    },
    # baseline: velocity flow matching
    "VFM": {
        "builder": build_VFM,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            "n_steps": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "k": ("choice", [10, 20, 40]),
            "eps": ("log_uniform", 1e-3, 5e-3),
            "integration_steps": ("uniform_int", 300, 2600),
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            "n_hutch_samples": ("choice", [1, 4, 16]),
            # activation knob for MLP networks
            "activation": ("choice", ["gelu", "elu", "silu"]),
            # noise schedule: type + amplitude, train + test path
            "sched": ("choice", ["stiff", "bridge"]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "gamma_min": ("log_uniform", 1e-2, 2e-1),
            "inner_eps": ("choice", [0.0, 0.05, 0.1]),
            "test_eps": ("log_uniform", 1e-3, 1e-1),
            "test_sched": ("choice", ["stiff", "bridge"]),
            "test_sigma": ("log_uniform", 0.3, 3.0),
            "test_gamma_min": ("log_uniform", 1e-2, 2e-1),
            "test_inner_eps": ("choice", [0.0, 0.05, 0.1]),
            # tier-1 estimator scalars
            "hidden_dim": ("choice", [128, 256, 512]),
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "layernorm": ("choice", ["off", "pre", "post"]),
            "reweight": ("choice", [False, True]),
            "antithetic": ("choice", [False, True]),
            "div_noise": ("choice", ["rademacher", "gaussian"]),
            "div_method": ("choice", ["hutchinson", "exact"]),
            # EDM preconditioning toggle
            "precond": ("choice", [False, True]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
        },
        "tabular_only": False,
    },
    # baseline: velocity flow matching with orthogonal shared backbone
    "VFMOrthros": {
        "builder": build_VFMOrthros,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            "n_steps": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "k": ("choice", [10, 20, 40]),
            "eps": ("log_uniform", 1e-3, 5e-3),
            # gamma_min: standard VFM-style noise floor for the score term
            "gamma_min": ("log_uniform", 1e-2, 2e-1),
            # test_eps: clips the inference integration domain away from the
            # tau->0 corner where the derived x1 endpoint carries a 1/beta factor
            "test_eps": ("log_uniform", 2e-2, 1e-1),
            "integration_steps": ("uniform_int", 300, 2600),
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            "n_hutch_samples": ("choice", [1, 4, 16]),
            # activation knob for MLP networks
            "activation": ("choice", ["gelu", "elu", "silu"]),
            # VFMOrthros-specific: number of shared backbone layers. capped at 2
            # so n_shared_layers <= n_hidden_layers holds for the smallest
            # n_hidden_layers choice (OrthrosNet enforces this at build time).
            "n_shared_layers": ("choice", [1, 2]),
            # noise schedule: type + amplitude, train + test path
            "sched": ("choice", ["stiff", "bridge"]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "inner_eps": ("choice", [0.0, 0.05, 0.1]),
            "test_sched": ("choice", ["stiff", "bridge"]),
            "test_sigma": ("log_uniform", 0.3, 3.0),
            "test_gamma_min": ("log_uniform", 1e-2, 2e-1),
            "test_inner_eps": ("choice", [0.0, 0.05, 0.1]),
            # tier-1 estimator scalars
            "hidden_dim": ("choice", [128, 256, 512]),
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "layernorm": ("choice", ["off", "pre", "post"]),
            "reweight": ("choice", [False, True]),
            "antithetic": ("choice", [False, True]),
            "div_noise": ("choice", ["rademacher", "gaussian"]),
            "div_method": ("choice", ["hutchinson", "exact"]),
            # EDM preconditioning toggle
            "precond": ("choice", [False, True]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
        },
        "tabular_only": False,
    },
    # baseline: binary density ratio estimation
    "BDRE": {
        "builder": build_BDRE,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            # tier-1 architecture scalars
            "latent_dim": ("choice", [64, 128, 256]),
            "n_hidden_layers": ("choice", [1, 2, 3]),
            # tier-2 training dynamics
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "n_steps": ("log_uniform_int", 100, 500),
            "batch_size": ("choice", [None, 128, 256]),
        },
        "tabular_only": False,
    },
    # baseline: multiclass DRE (15 waypoints fixed)
    "MDRE_15": {
        "builder": build_MDRE,
        "requires_pstar": False,
        "num_waypoints": None,  # HP-sampled per trial
        "base_search_space": {
            # tier-1 architecture scalars
            "latent_dim": ("choice", [64, 128, 256]),
            "n_hidden_layers": ("choice", [1, 2, 3]),
            # tier-2 training dynamics
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "n_steps": ("log_uniform_int", 100, 500),
            "batch_size": ("choice", [None, 128, 256]),
            # classifier output dimension (= num_classes)
            "num_waypoints": ("choice", [5, 10, 15]),
        },
        "tabular_only": False,
    },
    # baseline: multi-head TDRE (standard)
    "MultiHeadTDRE": {
        "builder": build_MHTDRE,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "n_steps": ("log_uniform_int", 100, 1000),
            "hidden_dim": ("choice", [16, 32, 64, 128]),
            "head_dim": ("choice", [10, 20, 40]),
            "num_shared_layers": ("choice", [1, 2, 3]),
            "num_waypoints": ("choice", [5, 10, 15]),
        },
        "tabular_only": False,
    },
    # baseline: flow-based multivariate DRE
    "FMDRE": {
        "builder": build_FMDRE,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            "n_steps": ("log_uniform_int", 500, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [128, 256, 512]),
            "eps": ("log_uniform", 1e-3, 5e-2),
            "integration_steps": ("uniform_int", 1000, 3000),
            "hidden_dim": ("choice", [128, 256, 512]),
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "score_weight": ("log_uniform", 0.1, 10.0),
            "reweight": ("choice", [False, True]),
            # EDM preconditioning toggle
            "precond": ("choice", [False, True]),
            # importance sampling time distribution knob
            "time_dist": ("choice", list(TIME_DISTS)),
            # ema / grad-clip pilot knobs
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
            "apply_iw": ("choice", [True, False]),
            # not an HPO knob: FMDRE-family div_method/n_hutch_samples are pinned
            # to "hutch_rademacher"/4 (mirrors VFM/VFMOrthros)
        },
        "tabular_only": False,
    },
    # baseline: FMDRE with unconditional score weighting
    "FMDRE_S2": {
        "builder": build_FMDRE_S2,
        "requires_pstar": False,
        "num_waypoints": None,
        "base_search_space": {
            "n_steps": ("log_uniform_int", 500, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [128, 256, 512]),
            "eps": ("log_uniform", 1e-3, 5e-2),
            "integration_steps": ("uniform_int", 1000, 3000),
            "hidden_dim": ("choice", [128, 256, 512]),
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "score_weight": ("log_uniform", 0.1, 10.0),
            "p_uncond": ("uniform", 0.1, 0.9),
            "reweight": ("choice", [False, True]),
            # EDM preconditioning toggle
            "precond": ("choice", [False, True]),
            # importance sampling time distribution knob
            "time_dist": ("choice", list(TIME_DISTS)),
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
            "apply_iw": ("choice", [True, False]),
            # not an HPO knob: FMDRE-family div_method/n_hutch_samples are pinned
            # to "hutch_rademacher"/4 (mirrors VFM/VFMOrthros)
            # not an HPO knob: sentinel_cond is FMDRE_S2's internal CFG sentinel
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
            "n_steps": ("log_uniform_int", 500, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [128, 256, 512]),
            "eps": ("log_uniform", 1e-3, 5e-2),
            "integration_steps": ("uniform_int", 1000, 3000),
            "hidden_dim": ("choice", [128, 256, 512]),
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "score_weight": ("log_uniform", 0.1, 10.0),
            "triangular_p_uncond": ("uniform", 0.0, 0.5),
            "layernorm": ("choice", ["off", "pre", "post"]),
            "reweight": ("choice", [False, True]),
            # EDM preconditioning toggle
            "precond": ("choice", [False, True]),
            # importance sampling time distribution knob
            "time_dist": ("choice", list(TIME_DISTS)),
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
            "apply_iw": ("choice", [True, False]),
            # not an HPO knob: FMDRE-family div_method/n_hutch_samples are pinned
            # to "hutch_rademacher"/4 (mirrors VFM/VFMOrthros)
        },
        "tabular_only": False,
    },
    # triangular: TSM with triangular path
    "TriangularTSM": {
        "builder": build_TriangularTSM,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_steps": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "hidden_dim": ("choice", [128, 256, 512]),
            "eps": ("log_uniform", 1e-5, 1e-3),
            "vertex": ("uniform", 0.2, 0.8),
            # peak_max lower bound dropped from 0.5 to 0.05 -- the synthetic
            # A/B (D=14 Gaussian, 2 seeds) showed peak_max=0.05 matches plain
            # TSM's MAE while peak_max=0.5/1.0 is materially worse. the
            # constructor enforces peak_max in (0, 1], so 0.05 is the
            # smallest practical lower bound. allows hpo to discover that
            # weak/no anchor wins on score-matching-style triangular paths.
            "peak_max": ("uniform", 0.05, 1.0),
            # tier-1 estimator scalars
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "activation": ("choice", ["elu", "gelu", "silu"]),
            "reweight": ("choice", [False, True]),
            # importance sampling time distribution knob
            "time_dist": ("choice", list(TIME_DISTS)),
            # ema / grad-clip pilot knobs
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
            "apply_iw": ("choice", [True, False]),
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
            "n_steps": ("log_uniform_int", 100, 1000),
            "hidden_dim": ("choice", [16, 32, 64, 128]),
            "head_dim": ("choice", [10, 20, 40]),
            "num_shared_layers": ("choice", [1, 2, 3]),
            "num_waypoints": ("choice", [5, 10, 15]),
            "vertex": ("uniform", 0.2, 0.8),
        },
        "tabular_only": False,
    },
    # triangular: multiclass classifier with triangular weighting
    "TriangularMDRE": {
        "builder": build_TriangularMDRE,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            # tier-1 architecture scalars
            "latent_dim": ("choice", [64, 128, 256]),
            "n_hidden_layers": ("choice", [1, 2, 3]),
            # tier-2 training dynamics
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "n_steps": ("log_uniform_int", 100, 500),
            "batch_size": ("choice", [None, 128, 256]),
            # classifier output dimension (= num_classes)
            "num_waypoints": ("choice", [5, 10, 15]),
            # triangular waypoint-builder knobs
            "midpoint_oversample": ("choice", [0, 3, 5, 7]),
            "gamma_power": ("uniform", 1.0, 5.0),
            "vertex": ("uniform", 0.2, 0.8),
        },
        "tabular_only": False,
    },
    # triangular: piecewise-SB path with vertex kink point
    "TriangularCTSM_V1": {
        "builder": build_TriangularCTSM_V1,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_steps": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "vertex": ("uniform", 0.2, 0.8),
            "eps": ("log_uniform", 1e-3, 3e-3),
            "integration_steps": ("uniform_int", 300, 2600),
            # pilot fix knobs: inner_eps avoids the per-leg vertex-singularity
            # band; ema_decay + grad_clip_norm mirror the CTSM family.
            # post-pilot tightening: 0.10/0.20 plateau on KL=10 sweep, dropped.
            "inner_eps": ("choice", [0.0, 0.02, 0.05]),
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            # importance sampling time distribution knob
            "time_dist": ("choice", list(TIME_DISTS)),
            # activation knob for score network
            "activation": ("choice", ["elu", "gelu", "silu"]),
            # noise schedule: type, train + test path
            "sched": ("choice", ["stiff", "bridge"]),
            "k": ("choice", [10, 20, 40]),
            "gamma_min": ("log_uniform", 1e-2, 2e-1),
            "test_eps": ("log_uniform", 1e-3, 1e-1),
            "test_sched": ("choice", ["stiff", "bridge"]),
            "test_sigma": ("log_uniform", 0.3, 3.0),
            "test_gamma_min": ("log_uniform", 1e-2, 2e-1),
            "test_inner_eps": ("choice", [0.0, 0.02, 0.05]),
            # tier-1 estimator scalars
            "hidden_dim": ("choice", [128, 256, 512]),
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "reweight": ("choice", [False, True]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
            "apply_iw": ("choice", [True, False]),
        },
        "tabular_only": False,
    },
    # triangular: barycentric path (fixed vertex = 0.5)
    "TriangularCTSM_V2": {
        "builder": build_TriangularCTSM_V2,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_steps": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "eps": ("log_uniform", 3e-4, 3e-3),
            "integration_steps": ("uniform_int", 300, 2600),
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            # importance sampling time distribution knob
            "time_dist": ("choice", list(TIME_DISTS)),
            # activation knob for score network
            "activation": ("choice", ["elu", "gelu", "silu"]),
            # noise schedule: type, train + test path
            "sched": ("choice", ["stiff", "bridge"]),
            "k": ("choice", [10, 20, 40]),
            "gamma_min": ("log_uniform", 1e-2, 2e-1),
            "inner_eps": ("choice", [0.0, 0.05, 0.1]),
            "test_eps": ("log_uniform", 1e-3, 1e-1),
            "test_sched": ("choice", ["stiff", "bridge"]),
            "test_sigma": ("log_uniform", 0.3, 3.0),
            "test_gamma_min": ("log_uniform", 1e-2, 2e-1),
            "test_inner_eps": ("choice", [0.0, 0.05, 0.1]),
            # tier-1 estimator scalars
            "hidden_dim": ("choice", [128, 256, 512]),
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "reweight": ("choice", [False, True]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
            "apply_iw": ("choice", [True, False]),
        },
        "tabular_only": False,
    },
    # triangular: 2D stacked path with curvature params
    "TriangularCTSM_V3": {
        "builder": build_TriangularCTSM_V3,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_steps": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            # noise schedule: type ("stiff"/"bridge") replaces the old
            # gamma_schedule string; the 2d path builder reads sched/sigma/k.
            "sched": ("choice", ["stiff", "bridge"]),
            "k": ("choice", [12, 24, 48]),
            "t2_max": ("uniform", 0.6, 0.9),
            "eps": ("log_uniform", 1e-3, 3e-3),
            "path_height": ("uniform", 1.0, 2.0),
            "integration_steps": ("uniform_int", 300, 2600),
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            # activation knob for score network
            "activation": ("choice", ["elu", "gelu", "silu"]),
            # test path
            "gamma_min": ("log_uniform", 1e-2, 2e-1),
            "inner_eps": ("choice", [0.0, 0.05, 0.1]),
            "test_eps": ("log_uniform", 1e-3, 1e-1),
            "test_sched": ("choice", ["stiff", "bridge"]),
            "test_sigma": ("log_uniform", 0.3, 3.0),
            "test_gamma_min": ("log_uniform", 1e-2, 2e-1),
            "test_inner_eps": ("choice", [0.0, 0.05, 0.1]),
            # tier-1 estimator scalars
            "hidden_dim": ("choice", [128, 256, 512]),
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "reweight": ("choice", [False, True]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
        },
        "tabular_only": False,
    },
    # triangular: piecewise-SB path with gamma_min for hamiltonian weighting
    "TriangularVFM_V1": {
        "builder": build_TriangularVFM_V1,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_steps": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "vertex": ("uniform", 0.2, 0.8),
            "gamma_min": ("log_uniform", 1e-2, 1e-1),
            "eps": ("log_uniform", 1e-3, 5e-3),
            "integration_steps": ("uniform_int", 300, 2600),
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            "n_hutch_samples": ("choice", [1, 4, 16]),
            # activation knob for MLP networks
            "activation": ("choice", ["gelu", "elu", "silu"]),
            # inner_eps vertex-band guard. VFM V1's gamma_min floor already
            # protects from the singularity, so 0.0 is competitive; small/large
            # values can help on harder problems. wider grid than CTSM V1
            # because A/B sweep showed sweet-spot dataset-dependent.
            "inner_eps": ("choice", [0.0, 0.05, 0.1, 0.2]),
            # noise schedule: type + stiff sharpness, train + test path. k
            # parameterises stiff_noise(k, sigma) and is read by _sched_1d
            # whenever sched == "stiff" (inert under bridge). it is distinct
            # from the barycentric `k` constructor scalar -- which the psb path
            # genuinely ignores because the path is passed in directly.
            "sched": ("choice", ["stiff", "bridge"]),
            "k": ("choice", [10, 20, 40]),
            "test_eps": ("log_uniform", 1e-3, 1e-1),
            "test_sched": ("choice", ["stiff", "bridge"]),
            "test_sigma": ("log_uniform", 0.3, 3.0),
            "test_gamma_min": ("log_uniform", 1e-2, 1e-1),
            "test_inner_eps": ("choice", [0.0, 0.05, 0.1, 0.2]),
            # importance sampling time distribution knob (builder-consumed)
            "time_dist": ("choice", list(TIME_DISTS)),
            "apply_iw": ("choice", [True, False]),
            # tier-1 estimator scalars
            "hidden_dim": ("choice", [128, 256, 512]),
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "layernorm": ("choice", ["off", "pre", "post"]),
            "reweight": ("choice", [False, True]),
            "antithetic": ("choice", [False, True]),
            "div_noise": ("choice", ["rademacher", "gaussian"]),
            "div_method": ("choice", ["hutchinson", "exact"]),
            # EDM preconditioning toggle
            "precond": ("choice", [False, True]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
        },
        "tabular_only": False,
    },
    # triangular: barycentric path (fixed vertex = 0.5)
    "TriangularVFM_V2": {
        "builder": build_TriangularVFM_V2,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_steps": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "k": ("choice", [10, 20, 40]),
            "eps": ("log_uniform", 1e-3, 5e-3),
            "integration_steps": ("uniform_int", 300, 2600),
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            "n_hutch_samples": ("choice", [1, 4, 16]),
            # activation knob for MLP networks
            "activation": ("choice", ["gelu", "elu", "silu"]),
            # noise schedule: type + amplitude, train + test path
            "sched": ("choice", ["stiff", "bridge"]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "gamma_min": ("log_uniform", 1e-2, 2e-1),
            "inner_eps": ("choice", [0.0, 0.05, 0.1]),
            "vertex": ("uniform", 0.2, 0.8),
            "test_eps": ("log_uniform", 1e-3, 1e-1),
            "test_sched": ("choice", ["stiff", "bridge"]),
            "test_sigma": ("log_uniform", 0.3, 3.0),
            "test_gamma_min": ("log_uniform", 1e-2, 2e-1),
            "test_inner_eps": ("choice", [0.0, 0.05, 0.1]),
            # importance sampling time distribution knob (builder-consumed)
            "time_dist": ("choice", list(TIME_DISTS)),
            "apply_iw": ("choice", [True, False]),
            # tier-1 estimator scalars
            "hidden_dim": ("choice", [128, 256, 512]),
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "layernorm": ("choice", ["off", "pre", "post"]),
            "reweight": ("choice", [False, True]),
            "antithetic": ("choice", [False, True]),
            "div_noise": ("choice", ["rademacher", "gaussian"]),
            "div_method": ("choice", ["hutchinson", "exact"]),
            # EDM preconditioning toggle
            "precond": ("choice", [False, True]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
        },
        "tabular_only": False,
    },
    # triangular: 2D stacked path with curvature params
    "TriangularVFM_V3": {
        "builder": build_TriangularVFM_V3,
        "requires_pstar": True,
        "num_waypoints": None,
        "base_search_space": {
            "n_steps": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 5e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "k": ("choice", [10, 20, 40]),
            # noise schedule: type ("stiff"/"bridge") replaces gamma_schedule.
            "sched": ("choice", ["stiff", "bridge"]),
            "sigma": ("log_uniform", 0.3, 3.0),
            "t2_max": ("uniform", 0.6, 0.9),
            "eps": ("log_uniform", 1e-3, 5e-3),
            "path_height": ("uniform", 1.0, 2.0),
            "integration_steps": ("uniform_int", 300, 2600),
            "ema_decay": ("choice", [None, 0.999, 0.9999]),
            # activation knob for MLP networks
            "activation": ("choice", ["gelu", "elu", "silu"]),
            "grad_clip_norm": ("choice", [None, 1.0, 5.0]),
            "n_hutch_samples": ("choice", [1, 4, 16]),
            # test path
            "gamma_min": ("log_uniform", 1e-2, 2e-1),
            "inner_eps": ("choice", [0.0, 0.05, 0.1]),
            "test_eps": ("log_uniform", 1e-3, 1e-1),
            "test_sched": ("choice", ["stiff", "bridge"]),
            "test_sigma": ("log_uniform", 0.3, 3.0),
            "test_gamma_min": ("log_uniform", 1e-2, 2e-1),
            "test_inner_eps": ("choice", [0.0, 0.05, 0.1]),
            # tier-1 estimator scalars
            "hidden_dim": ("choice", [128, 256, 512]),
            "n_hidden_layers": ("choice", [2, 3, 4]),
            "layernorm": ("choice", ["off", "pre", "post"]),
            "reweight": ("choice", [False, True]),
            "antithetic": ("choice", [False, True]),
            "div_noise": ("choice", ["rademacher", "gaussian"]),
            "div_method": ("choice", ["hutchinson", "exact"]),
            # tier-3 cfg fields
            "weight_decay": ("choice", [0.0, 1e-5, 1e-4, 1e-3]),
            "cosine_min_factor": ("choice", [0.0, 0.01, 0.1]),
            # not an HPO knob: TriangularVFM2D has no precond support
        },
        "tabular_only": False,
    },
}

# alias pairs for registry: (short_name, canonical_name)
ALIAS_PAIRS = [
    ("MHTTDRE", "MultiHeadTriangularTDRE"),
    ("MDRE", "MDRE_15"),
    ("TriangularCTSM", "TriangularCTSM_V1"),
    ("TriangularVFM", "TriangularVFM_V1"),
]


# ---------------------------------------------------------------------------
# wave 3 overrides: isolated factorial.
# ---------------------------------------------------------------------------
# triggered by env var DPE_WAVE3=1. when active, every non-fix HP becomes a
# choice over a single representative value, while the new fix knobs (ema_decay,
# grad_clip_norm, time_dist, n_hutch_samples, inner_eps, activation) keep their
# full sweep ranges. effect: trials become a clean factorial over the fix knobs
# with everything else frozen, addressing the debate's "confounded attribution"
# critique.
#
# representative values are picked at the midpoint of each existing range.
# rationale: a single locked value lets the factorial expose true fix-knob
# effects without confounding from non-fix-knob variability.
import os as _os

_FIX_KNOBS = {
    "ema_decay", "grad_clip_norm", "time_dist", "n_hutch_samples",
    "inner_eps", "activation",
}

# unused after refactor; kept here as documentation of the canonical
# midpoint values that wave-3 will pick automatically via _midpoint().
_WAVE3_LOCKS = {
    "n_steps": 1100,
    "lr": 1e-3,
    "batch_size": 128,
    "sigma": 1.0,
    "eps": 2e-3,
    "integration_steps": 1500,
    "vertex": 0.5,
    "gamma_schedule": "sqrt",  # CTSM-V3 allowed; VFM-V3 will keep its single allowed value
    "k": 20,
    "t2_max": 0.75,
    "path_height": 1.5,
    "gamma_min": 5e-2,
    "hidden_dim": 256,
    "n_hidden_layers": 3,
}

def _midpoint(spec_tuple):
    """return a sensible single value at the center of a spec_tuple.

    spec_tuple forms (matching ex.utils.hpo.sample.sample_param):
      ("choice", [v1, v2, ...])          -> middle element
      ("uniform", lo, hi)                -> (lo + hi) / 2
      ("log_uniform", lo, hi)            -> sqrt(lo * hi)
      ("uniform_int", lo, hi)            -> (lo + hi) // 2
      ("log_uniform_int", lo, hi)        -> round(sqrt(lo * hi))
    """
    import math
    kind = spec_tuple[0]
    if kind == "choice":
        opts = spec_tuple[1]
        return opts[len(opts) // 2]
    if kind == "uniform":
        lo, hi = spec_tuple[1], spec_tuple[2]
        return 0.5 * (lo + hi)
    if kind == "log_uniform":
        lo, hi = spec_tuple[1], spec_tuple[2]
        return math.sqrt(lo * hi)
    if kind == "uniform_int":
        lo, hi = spec_tuple[1], spec_tuple[2]
        return (lo + hi) // 2
    if kind == "log_uniform_int":
        lo, hi = spec_tuple[1], spec_tuple[2]
        return int(round(math.sqrt(lo * hi)))
    raise ValueError(f"unsupported spec kind for wave3 lock: {spec_tuple!r}")


def _wave3_lock(spec_dict):
    """convert a base_search_space dict into its wave-3 locked variant.

    keeps fix-knob params unchanged; collapses every other param to a
    single-element choice at the midpoint of its existing spec. this
    preserves method-specific constraints (e.g. TriangularVFM_V3's
    `gamma_schedule` only allows "linear-stiff") because we never invent
    values that weren't already in the spec.
    """
    out = {}
    for k, v in spec_dict.items():
        if k in _FIX_KNOBS:
            out[k] = v
        else:
            mid = _midpoint(v)
            out[k] = ("choice", [mid])
    return out


if _os.environ.get("DPE_WAVE3") == "1":
    for _name, _spec in METHOD_SPECS.items():
        if _spec.get("tabular_only"):
            continue
        _spec["base_search_space"] = _wave3_lock(_spec["base_search_space"])
