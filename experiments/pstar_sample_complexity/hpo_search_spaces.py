"""
HPO search spaces and builder functions for pstar_sample_complexity.

Each entry in SEARCH_SPACES maps a method name to:
  - search_space: dict of param_name -> (distribution_type, *args)
  - builder: callable (input_dim, device, config, **hyperparams) -> estimator

Only the most impactful hyperparameters are tuned per method. Batch sizes,
integration steps, and path geometry defaults are fixed at GPU-efficient values.

All builders accept three sample sets (p0, p1, pstar) since every triangular
method uses p* during fit().

Distribution types (from experiments.utils.hpo.sample):
  ("log_uniform", lo, hi)      continuous, uniform in log space
  ("log_uniform_int", lo, hi)  same but rounded to int
  ("uniform", lo, hi)          continuous uniform
  ("choice", [v1, v2, ...])    discrete uniform over list
"""

from src.density_ratio_estimation.mh_triangular_tdre import MultiHeadTriangularTDRE
from src.density_ratio_estimation.triangular_ctsm import TriangularCTSM
from src.density_ratio_estimation.triangular_ctsm_2d import TriangularCTSM2D
from src.density_ratio_estimation.triangular_fmdre import TriangularFMDRE
from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
from src.density_ratio_estimation.triangular_tsm import TriangularTSM
from src.density_ratio_estimation.triangular_vfm import TriangularVFM
from src.density_ratio_estimation.triangular_vfm_2d import TriangularVFM2D

from src.models.binary_classification import make_multi_head_binary_classifier
from src.models.multiclass_classification import make_multiclass_classifier

from src.waypoints.triangular_continuous import BarycentricCtsm1D, BarycentricVfm1D
from src.waypoints.triangular_continuous_2d import Stacked2DCtsm, Stacked2DVfm
from src.waypoints.triangular_waypoints import TriangularWaypointBuilder1D
from src.waypoints.curve_2d import Curve2D


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def build_MultiHeadTriangularTDRE(input_dim: int, device: str, config: dict, **hp) -> MultiHeadTriangularTDRE:
    num_waypoints = config["num_waypoints"]
    classifier = make_multi_head_binary_classifier(
        input_dim=input_dim,
        num_heads=num_waypoints - 1,
        hidden_dim=hp["hidden_dim"],
        head_dim=hp["head_dim"],
        num_shared_layers=2,  # fixed: sufficient shared capacity for dim=3
        learning_rate=hp["learning_rate"],
        num_epochs=hp["num_epochs"],
    )
    return MultiHeadTriangularTDRE(
        classifier=classifier,
        num_waypoints=num_waypoints,
        device=device,
    )


def build_TriangularCTSM(input_dim: int, device: str, config: dict, **hp) -> TriangularCTSM:
    # BarycentricCtsm1D: vertex fixed at 0.5 (symmetric canonical choice)
    path = BarycentricCtsm1D(sigma=hp["sigma"], vertex=0.5, eps=1e-3)
    return TriangularCTSM(
        input_dim=input_dim,
        path=path,
        n_epochs=hp["n_epochs"],
        lr=hp["lr"],
        batch_size=512,
        eps=1e-3,
        device=device,
    )


def build_TriangularCTSM2D(input_dim: int, device: str, config: dict, **hp) -> TriangularCTSM2D:
    # Stacked2DCtsm: t2_max and path_height fixed; sigma, k, gamma_schedule tuned
    path = Stacked2DCtsm(
        sigma=hp["sigma"],
        gamma_schedule=hp["gamma_schedule"],
        k=hp["k"],
        t2_max=0.3,
        eps=1e-3,
    )
    curve = Curve2D(path_height=1.0)
    return TriangularCTSM2D(
        input_dim=input_dim,
        path=path,
        curve=curve,
        n_epochs=hp["n_epochs"],
        lr=hp["lr"],
        batch_size=512,
        eps=1e-3,
        device=device,
    )


def build_TriangularFMDRE(input_dim: int, device: str, config: dict, **hp) -> TriangularFMDRE:
    # integration_steps=1000: sufficient for dim=3; batch_size=512 for GPU efficiency
    return TriangularFMDRE(
        input_dim=input_dim,
        device=device,
        n_epochs=hp["n_epochs"],
        lr=hp["lr"],
        score_weight=hp["score_weight"],
        batch_size=512,
        integration_steps=1000,
    )


def build_TriangularMDRE(input_dim: int, device: str, config: dict, **hp) -> TriangularMDRE:
    num_waypoints = config["num_waypoints"]
    classifier = make_multiclass_classifier(
        name="default",
        input_dim=input_dim,
        num_classes=num_waypoints,
        latent_dim=hp["latent_dim"],
        learning_rate=hp["learning_rate"],
        num_epochs=hp["num_epochs"],
    )
    builder = TriangularWaypointBuilder1D(
        midpoint_oversample=hp["midpoint_oversample"],
        gamma_power=hp["gamma_power"],
        vertex=hp.get("vertex", 0.5),
    )
    return TriangularMDRE(
        classifier=classifier,
        waypoint_builder=builder,
        device=device,
    )


def build_TriangularTSM(input_dim: int, device: str, config: dict, **hp) -> TriangularTSM:
    # vertex controls peak location of the triangular path; batch_size=512 fixed
    return TriangularTSM(
        input_dim=input_dim,
        device=device,
        hidden_dim=hp["hidden_dim"],
        n_epochs=hp["n_epochs"],
        lr=hp["lr"],
        vertex=hp["vertex"],
        batch_size=512,
    )


def build_TriangularVFM(input_dim: int, device: str, config: dict, **hp) -> TriangularVFM:
    # BarycentricVfm1D: vertex fixed at 0.5; integration_steps=1000 fixed
    path = BarycentricVfm1D(k=hp["k"], vertex=0.5, eps=1e-3)
    return TriangularVFM(
        input_dim=input_dim,
        path=path,
        n_epochs=hp["n_epochs"],
        lr=hp["lr"],
        batch_size=512,
        eps=1e-3,
        integration_steps=1000,
        device=device,
    )


def build_TriangularVFM2D(input_dim: int, device: str, config: dict, **hp) -> TriangularVFM2D:
    # Stacked2DVfm: only linear-stiff schedule supported; t2_max and path_height fixed
    path = Stacked2DVfm(
        k=hp["k"],
        gamma_schedule="linear-stiff",
        t2_max=0.3,
        eps=1e-3,
    )
    curve = Curve2D(path_height=1.0)
    return TriangularVFM2D(
        input_dim=input_dim,
        path=path,
        curve=curve,
        n_epochs=hp["n_epochs"],
        lr=hp["lr"],
        batch_size=512,
        eps=1e-3,
        integration_steps=1000,
        device=device,
    )


# ---------------------------------------------------------------------------
# search spaces
# ---------------------------------------------------------------------------

SEARCH_SPACES = {
    # MultiHeadTriangularTDRE: multi-head binary classifier wrapping TriangularTDRE.
    # hidden_dim and head_dim jointly determine capacity; num_shared_layers fixed at 2.
    "MultiHeadTriangularTDRE": {
        "search_space": {
            "hidden_dim":    ("choice", [32, 64, 128]),
            "head_dim":      ("choice", [10, 20, 40]),
            "learning_rate": ("log_uniform", 1e-4, 1e-2),
            "num_epochs":    ("log_uniform_int", 100, 800),
        },
        "builder": build_MultiHeadTriangularTDRE,
    },
    # TriangularCTSM (V2 barycentric): sigma controls SB path width; vertex fixed at 0.5.
    "TriangularCTSM": {
        "search_space": {
            "sigma":   ("log_uniform", 0.1, 3.0),
            "n_epochs": ("log_uniform_int", 200, 2000),
            "lr":      ("log_uniform", 1e-4, 1e-2),
        },
        "builder": build_TriangularCTSM,
    },
    # TriangularCTSM2D (V3 stacked-2D): sigma, k (path stiffness), gamma_schedule are tuned.
    # t2_max and path_height fixed at defaults; these are geometrically less sensitive.
    "TriangularCTSM2D": {
        "search_space": {
            "sigma":          ("log_uniform", 0.1, 3.0),
            "k":              ("choice", [5, 10, 20, 40]),
            "gamma_schedule": ("choice", ["sqrt", "linear-stiff"]),
            "n_epochs":       ("log_uniform_int", 200, 2000),
            "lr":             ("log_uniform", 1e-4, 1e-2),
        },
        "builder": build_TriangularCTSM2D,
    },
    # TriangularFMDRE: score_weight balances CFM vs score matching loss.
    "TriangularFMDRE": {
        "search_space": {
            "n_epochs":     ("log_uniform_int", 200, 1500),
            "lr":           ("log_uniform", 5e-4, 5e-3),
            "score_weight": ("log_uniform", 0.1, 10.0),
        },
        "builder": build_TriangularFMDRE,
    },
    # TriangularMDRE: classifier params + triangular path geometry (midpoint_oversample,
    # gamma_power). vertex fixed at 0.5.
    "TriangularMDRE": {
        "search_space": {
            "latent_dim":          ("choice", [32, 64, 128, 256]),
            "learning_rate":       ("log_uniform", 1e-4, 1e-2),
            "num_epochs":          ("log_uniform_int", 200, 1000),
            "midpoint_oversample": ("choice", [0, 1, 2, 4]),
            "gamma_power":         ("log_uniform", 0.5, 2.0),
        },
        "builder": build_TriangularMDRE,
    },
    # TriangularTSM: vertex controls triangular peak position on the path; hidden_dim
    # sets network width. batch_size=512 fixed.
    "TriangularTSM": {
        "search_space": {
            "hidden_dim": ("choice", [64, 128, 256]),
            "n_epochs":   ("log_uniform_int", 200, 2000),
            "lr":         ("log_uniform", 1e-4, 1e-2),
            "vertex":     ("uniform", 0.3, 0.7),
        },
        "builder": build_TriangularTSM,
    },
    # TriangularVFM (V2 barycentric): k controls path curvature. integration_steps=1000 fixed.
    "TriangularVFM": {
        "search_space": {
            "k":       ("choice", [10, 20, 40]),
            "n_epochs": ("log_uniform_int", 200, 2000),
            "lr":      ("log_uniform", 5e-4, 5e-3),
        },
        "builder": build_TriangularVFM,
    },
    # TriangularVFM2D (V3 stacked-2D): k controls stiffness of linear-stiff schedule.
    # gamma_schedule fixed at "linear-stiff" (only option implemented).
    "TriangularVFM2D": {
        "search_space": {
            "k":       ("choice", [5, 10, 20, 40]),
            "n_epochs": ("log_uniform_int", 200, 2000),
            "lr":      ("log_uniform", 5e-4, 5e-3),
        },
        "builder": build_TriangularVFM2D,
    },
}
