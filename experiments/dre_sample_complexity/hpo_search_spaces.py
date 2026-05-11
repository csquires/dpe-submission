"""
HPO search spaces and builder functions for dre_sample_complexity.

Each entry in SEARCH_SPACES maps a method name to:
  - search_space: dict of param_name -> (distribution_type, *args)
  - builder: callable (input_dim, device, config, **hyperparams) -> estimator

Only the most impactful hyperparameters are tuned per method to keep the
total trial count tractable. Batch sizes are fixed at GPU-efficient values;
architecture depth and integration settings are fixed at sensible defaults.

Distribution types (from experiments.utils.hpo.sample):
  ("log_uniform", lo, hi)      continuous, uniform in log space
  ("log_uniform_int", lo, hi)  same but rounded to int
  ("uniform", lo, hi)          continuous uniform
  ("choice", [v1, v2, ...])    discrete uniform over list
"""

import torch

from src.methods.cls.bdre import BDRE
from src.methods.cls.mdre import MDRE
from src.methods.cls.tdre.mh import MultiHeadTDRE
from src.methods.reg.tsm import TSM
from src.methods.reg.ctsm import CTSM
from src.methods.reg.vfm.spatial_adapters import make_spatial_velo_denoiser
from src.methods.reg.fmdre import FMDRE
from src.methods.reg.fmdre.s2 import FMDRE_S2

from src.models.binary_classification import make_binary_classifier
from src.models.binary_classification.multi_head_binary_classifier import make_multi_head_binary_classifier
from src.models.multiclass_classification import make_multiclass_classifier


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def build_BDRE(input_dim: int, device: str, config: dict, **hp) -> BDRE:
    classifier = make_binary_classifier(
        name="default",
        input_dim=input_dim,
        latent_dim=hp["latent_dim"],
        learning_rate=hp["learning_rate"],
        num_epochs=hp["num_epochs"],
    )
    return BDRE(classifier=classifier, device=device)


def build_MDRE(input_dim: int, device: str, config: dict, **hp) -> MDRE:
    num_waypoints = config["mdre_num_waypoints"]
    classifier = make_multiclass_classifier(
        name="default",
        input_dim=input_dim,
        num_classes=num_waypoints,
        latent_dim=hp["latent_dim"],
        learning_rate=hp["learning_rate"],
        num_epochs=hp["num_epochs"],
    )
    return MDRE(classifier=classifier, device=device)


def build_MHTDRE(input_dim: int, device: str, config: dict, **hp) -> MultiHeadTDRE:
    num_waypoints = config["mhtdre_num_waypoints"]
    classifier = make_multi_head_binary_classifier(
        input_dim=input_dim,
        num_heads=num_waypoints - 1,
        hidden_dim=hp["hidden_dim"],
        head_dim=hp["head_dim"],
        num_shared_layers=2,  # fixed: not worth tuning for dim=3 data
        learning_rate=hp["learning_rate"],
        num_epochs=hp["num_epochs"],
    )
    return MultiHeadTDRE(classifier=classifier, num_waypoints=num_waypoints, device=device)


def build_TSM(input_dim: int, device: str, config: dict, **hp) -> TSM:
    # batch_size=512: saturates GPU well for dim=3 data; not sensitive enough to tune
    return TSM(
        input_dim=input_dim,
        device=device,
        n_epochs=hp["n_epochs"],
        lr=hp["lr"],
        batch_size=512,
    )


def build_CTSM(input_dim: int, device: str, config: dict, **hp) -> CTSM:
    return CTSM(
        input_dim=input_dim,
        device=device,
        n_epochs=hp["n_epochs"],
        lr=hp["lr"],
        sigma=hp["sigma"],
        batch_size=512,
    )


def build_VFM(input_dim: int, device: str, config: dict, **hp):
    # integration_steps fixed at 3000 (default from grid search in spatial_adapters.py)
    return make_spatial_velo_denoiser(
        input_dim=input_dim,
        device=device,
        n_epochs=hp["n_epochs"],
        lr=hp["lr"],
        k=hp["k"],
    )


def build_FMDRE(input_dim: int, device: str, config: dict, **hp) -> FMDRE:
    # integration_steps fixed at 1000: sufficient for dim=3, tuning it adds noise
    return FMDRE(
        input_dim=input_dim,
        device=device,
        n_epochs=hp["n_epochs"],
        lr=hp["lr"],
        score_weight=hp["score_weight"],
        batch_size=256,
        integration_steps=1000,
    )


def build_FMDRE_S2(input_dim: int, device: str, config: dict, **hp) -> FMDRE_S2:
    return FMDRE_S2(
        input_dim=input_dim,
        device=device,
        n_epochs=hp["n_epochs"],
        lr=hp["lr"],
        p_uncond=hp["p_uncond"],
        batch_size=256,
        integration_steps=1000,
    )


# ---------------------------------------------------------------------------
# search spaces
# ---------------------------------------------------------------------------

SEARCH_SPACES = {
    # --- classifier-based methods ---
    # DefaultBinaryClassifier does full-batch GD, so batch_size is not applicable.
    # latent_dim controls network width; learning_rate and num_epochs are the main knobs.
    "BDRE": {
        "search_space": {
            "latent_dim":     ("choice", [32, 64, 128, 256]),
            "learning_rate":  ("log_uniform", 1e-4, 1e-2),
            "num_epochs":     ("log_uniform_int", 200, 1000),
        },
        "builder": build_BDRE,
    },
    "MDRE": {
        "search_space": {
            "latent_dim":     ("choice", [32, 64, 128, 256]),
            "learning_rate":  ("log_uniform", 1e-4, 1e-2),
            "num_epochs":     ("log_uniform_int", 200, 1000),
        },
        "builder": build_MDRE,
    },
    # MultiHeadBinaryClassifier: hidden_dim (backbone width) and head_dim are both
    # worth tuning since they jointly determine capacity; num_shared_layers fixed at 2.
    "MHTDRE": {
        "search_space": {
            "hidden_dim":     ("choice", [32, 64, 128]),
            "head_dim":       ("choice", [10, 20, 40]),
            "learning_rate":  ("log_uniform", 1e-4, 1e-2),
            "num_epochs":     ("log_uniform_int", 100, 800),
        },
        "builder": build_MHTDRE,
    },
    # --- score / flow methods ---
    # TSM: lr and n_epochs are the primary knobs; eps is not sensitive for dim=3.
    "TSM": {
        "search_space": {
            "n_epochs":  ("log_uniform_int", 200, 2000),
            "lr":        ("log_uniform", 1e-4, 1e-2),
        },
        "builder": build_TSM,
    },
    # CTSM: sigma controls the Schrodinger Bridge path width and is meaningful to tune.
    "CTSM": {
        "search_space": {
            "n_epochs":  ("log_uniform_int", 200, 2000),
            "lr":        ("log_uniform", 1e-4, 1e-2),
            "sigma":     ("log_uniform", 0.1, 3.0),
        },
        "builder": build_CTSM,
    },
    # VFM: k controls gamma curvature of the stochastic interpolant path, meaningful to tune.
    "VFM": {
        "search_space": {
            "n_epochs":  ("log_uniform_int", 200, 2000),
            "lr":        ("log_uniform", 5e-4, 5e-3),
            "k":         ("choice", [10, 20, 40]),
        },
        "builder": build_VFM,
    },
    # FMDRE: score_weight balances CFM vs score matching loss, sensitive in practice.
    "FMDRE": {
        "search_space": {
            "n_epochs":     ("log_uniform_int", 200, 1500),
            "lr":           ("log_uniform", 5e-4, 5e-3),
            "score_weight": ("log_uniform", 0.1, 10.0),
        },
        "builder": build_FMDRE,
    },
    # FMDRE_S2: p_uncond is the CFG dropout rate; the core difference from FMDRE.
    "FMDRE_S2": {
        "search_space": {
            "n_epochs":  ("log_uniform_int", 200, 1500),
            "lr":        ("log_uniform", 5e-4, 5e-3),
            "p_uncond":  ("uniform", 0.1, 0.9),
        },
        "builder": build_FMDRE_S2,
    },
}
