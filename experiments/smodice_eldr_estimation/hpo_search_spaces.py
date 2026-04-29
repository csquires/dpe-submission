"""registry of 10 hpo method specifications for smodice eldr estimation.

baselines: TabularPluginDRE, SmoothedTabularPluginDRE, BDRE, MDRE, TSM, CTSM.
triangular variants: TriangularMDRE, MultiHeadTriangularTDRE, TriangularCTSM, TriangularVFM.

each entry is a dict with keys:
  search_space: param_name -> (spec_type, *bounds) tuple for hpo tuning.
  builder: callable(input_dim, device, num_waypoints, **flat_hp) -> estimator.
  requires_pstar: bool, whether fit() receives p* samples (triangular/tabular learners).
  needs_latent: bool, whether fit() receives latent indices (SmoothedTabularPluginDRE only).
  encoding_compat: set of encoding type strings supported by the method.

encoding_compat values: onehot_joint, onehot_concat, gaussian_blob, flow_pushforward.
spec types: log_uniform, uniform, log_uniform_int, uniform_int, choice.
"""

from experiments.utils.hpo.builders import (
    build_TabularPluginDRE,
    build_SmoothedTabularPluginDRE,
    build_BDRE,
    build_MDRE,
    build_TriangularMDRE,
    build_MHTTDRE,
    build_TSM,
    build_CTSM,
    build_TriangularCTSM_V1,
    build_TriangularVFM_V1,
)


SEARCH_SPACES = {
    # baseline: oracle tabular plugin decoder via exact state-action counts.
    # no hpo tuning; state/action counts problem-defined; decoding invariant across encodings.
    "TabularPluginDRE": {
        "search_space": {},
        "builder": build_TabularPluginDRE,
        "requires_pstar": False,
        "needs_latent": False,
        "encoding_compat": {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    },

    # baseline: smoothed tabular plugin with gaussian smoothing over latent state-action pairs.
    # sigma baked into constructor; no hpo tuning; requires latent indices for smoothing grid.
    "SmoothedTabularPluginDRE": {
        "search_space": {},
        "builder": build_SmoothedTabularPluginDRE,
        "requires_pstar": False,
        "needs_latent": True,
        "encoding_compat": {"gaussian_blob", "flow_pushforward"},
    },

    # baseline: binary dnn classifier (p0 vs p1); no p* needed.
    # hyperparams: latent_dim, learning_rate, num_epochs.
    "BDRE": {
        "search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
        "builder": build_BDRE,
        "requires_pstar": False,
        "needs_latent": False,
        "encoding_compat": {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    },

    # baseline: multiclass dnn classifier over num_waypoints; no p* needed.
    # hyperparams: same as BDRE (architecture + optimization).
    "MDRE": {
        "search_space": {
            "latent_dim": ("choice", [64, 128, 256]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs": ("log_uniform_int", 100, 500),
        },
        "builder": build_MDRE,
        "requires_pstar": False,
        "needs_latent": False,
        "encoding_compat": {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    },

    # proposed: triangular mdre with midpoint supervision from p*.
    # extends BDRE with oversample ratio and power law path spacing.
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
        "needs_latent": False,
        "encoding_compat": {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    },

    # proposed: multi-head triangular tdre with binary classification head per waypoint edge.
    # requires p* for midpoint supervision; shared lower layers with per-edge output heads.
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
        "needs_latent": False,
        "encoding_compat": {"onehot_joint", "onehot_concat", "gaussian_blob", "flow_pushforward"},
    },

    # baseline: time score matching (unconditional); no p*.
    # continuous-only method (requires continuous latent representation).
    # hyperparams tuned for efficient training on ~5k samples per distribution.
    "TSM": {
        "search_space": {
            "n_epochs": ("log_uniform_int", 750, 1500),
            "lr": ("log_uniform", 3e-4, 3e-3),
            "batch_size": ("choice", [64, 128, 256]),
            "eps": ("log_uniform", 1e-6, 1e-4),
        },
        "builder": build_TSM,
        "requires_pstar": False,
        "needs_latent": False,
        "encoding_compat": {"gaussian_blob", "flow_pushforward"},
    },

    # baseline: conditional time score matching; no p*.
    # sigma tunes smoothing bandwidth of conditional score. continuous-only.
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
        "needs_latent": False,
        "encoding_compat": {"gaussian_blob", "flow_pushforward"},
    },

    # proposed: triangular ctsm with piecewise-sb path through p0 -> p* -> p1.
    # vertex tunes kink position on [0, 1]; piecewise-sb supervised at vertex by p*.
    # v1 builder uses piecewise-sb path (differs from hardcoded barycentric in step2; both valid).
    "TriangularCTSM": {
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
        "needs_latent": False,
        "encoding_compat": {"gaussian_blob", "flow_pushforward"},
    },

    # proposed: triangular vfm (variational flow matching) with piecewise-sb path.
    # requires p* for path definition; gamma_min lower-bounds hamiltonian weighting.
    # integration_steps controls ode discretization for inference.
    # v1 builder uses piecewise-sb (differs from step2 barycentric; both valid).
    "TriangularVFM": {
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
        "needs_latent": False,
        "encoding_compat": {"gaussian_blob", "flow_pushforward"},
    },
}
