"""builders for density ratio estimators sampled from HPO search spaces.

maps flat scalar hyperparameter dicts (from gen_hpo_configs.py) into fully-instantiated
DensityRatioEstimator instances. each builder corresponds to one method: six baselines
(TSM, CTSM, VFM, FMDRE, FMDRE_S2, MHTTDRE) and six triangular variants (TriangularCTSM
V1/V2/V3, TriangularVFM V1/V2/V3). builders encapsulate estimator instantiation, path/curve
assembly, and hyperparameter routing. all builders are pure functions with uniform signature.
"""

import torch

from src.density_ratio_estimation.tsm import TSM
from src.density_ratio_estimation.ctsm import CTSM
from src.density_ratio_estimation.spatial_adapters import make_spatial_velo_denoiser
from src.density_ratio_estimation.fmdre import FMDRE
from src.density_ratio_estimation.fmdre_s2 import FMDRE_S2

from src.density_ratio_estimation.mh_triangular_tdre import MultiHeadTriangularTDRE
from src.models.binary_classification import make_multi_head_binary_classifier

from src.density_ratio_estimation.triangular_ctsm import TriangularCTSM
from src.waypoints.piecewise_sb import PiecewiseSBCtsm1D
from src.waypoints.triangular_continuous import BarycentricCtsm1D

from src.density_ratio_estimation.triangular_ctsm_2d import TriangularCTSM2D
from src.waypoints.triangular_continuous_2d import Stacked2DCtsm
from src.waypoints.curve_2d import Curve2D

from src.density_ratio_estimation.triangular_vfm import TriangularVFM
from src.waypoints.piecewise_sb import PiecewiseSBVfm1D
from src.waypoints.triangular_continuous import BarycentricVfm1D

from src.density_ratio_estimation.triangular_vfm_2d import TriangularVFM2D
from src.waypoints.triangular_continuous_2d import Stacked2DVfm


def build_TSM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TSM:
    """return TSM estimator initialized from flat_hp dict."""
    return TSM(input_dim=input_dim, device=device, **flat_hp)


def build_CTSM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> CTSM:
    """return CTSM estimator initialized from flat_hp dict."""
    return CTSM(input_dim=input_dim, device=device, **flat_hp)


def build_VFM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp):
    """return VFM estimator initialized from flat_hp dict."""
    return make_spatial_velo_denoiser(input_dim=input_dim, device=device, **flat_hp)


def build_FMDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> FMDRE:
    """return FMDRE estimator initialized from flat_hp dict."""
    return FMDRE(input_dim=input_dim, device=device, **flat_hp)


def build_FMDRE_S2(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> FMDRE_S2:
    """return FMDRE_S2 estimator initialized from flat_hp dict."""
    return FMDRE_S2(input_dim=input_dim, device=device, **flat_hp)


def build_MHTTDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> MultiHeadTriangularTDRE:
    """return MHTTDRE estimator with multi-head classifier built from flat_hp dict."""
    classifier = make_multi_head_binary_classifier(
        input_dim=input_dim,
        num_heads=num_waypoints - 1,
        hidden_dim=flat_hp["hidden_dim"],
        head_dim=flat_hp["head_dim"],
        num_shared_layers=flat_hp["num_shared_layers"],
        learning_rate=flat_hp["learning_rate"],
        num_epochs=flat_hp["num_epochs"]
    )
    return MultiHeadTriangularTDRE(
        classifier=classifier,
        num_waypoints=num_waypoints,
        device=device
    )


def build_TriangularCTSM_V1(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularCTSM:
    """return TriangularCTSM V1 (piecewise-SB) estimator initialized from flat_hp dict."""
    path = PiecewiseSBCtsm1D(
        sigma=flat_hp["sigma"],
        vertex=flat_hp["vertex"],
        eps=flat_hp["eps"]
    )
    return TriangularCTSM(
        input_dim=input_dim,
        path=path,
        n_epochs=flat_hp["n_epochs"],
        lr=flat_hp["lr"],
        batch_size=flat_hp["batch_size"],
        eps=flat_hp["eps"],
        device=device
    )


def build_TriangularCTSM_V2(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularCTSM:
    """return TriangularCTSM V2 (barycentric) estimator initialized from flat_hp dict."""
    path = BarycentricCtsm1D(
        sigma=flat_hp["sigma"],
        vertex=0.5,
        eps=flat_hp["eps"]
    )
    return TriangularCTSM(
        input_dim=input_dim,
        path=path,
        n_epochs=flat_hp["n_epochs"],
        lr=flat_hp["lr"],
        batch_size=flat_hp["batch_size"],
        eps=flat_hp["eps"],
        device=device
    )


def build_TriangularCTSM_V3(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularCTSM2D:
    """return TriangularCTSM V3 (2D stacked) estimator initialized from flat_hp dict."""
    path = Stacked2DCtsm(
        sigma=flat_hp["sigma"],
        gamma_schedule=flat_hp["gamma_schedule"],
        k=flat_hp["k"],
        t2_max=flat_hp["t2_max"],
        eps=flat_hp["eps"]
    )
    curve = Curve2D(path_height=flat_hp["path_height"])
    return TriangularCTSM2D(
        input_dim=input_dim,
        path=path,
        curve=curve,
        n_epochs=flat_hp["n_epochs"],
        lr=flat_hp["lr"],
        batch_size=flat_hp["batch_size"],
        eps=flat_hp["eps"],
        device=device
    )


def build_TriangularVFM_V1(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularVFM:
    """return TriangularVFM V1 (piecewise-SB) estimator initialized from flat_hp dict."""
    path = PiecewiseSBVfm1D(
        sigma=flat_hp["sigma"],
        vertex=flat_hp["vertex"],
        gamma_min=flat_hp["gamma_min"],
        eps=flat_hp["eps"]
    )
    return TriangularVFM(
        input_dim=input_dim,
        path=path,
        n_epochs=flat_hp["n_epochs"],
        lr=flat_hp["lr"],
        batch_size=flat_hp["batch_size"],
        eps=flat_hp["eps"],
        integration_steps=flat_hp["integration_steps"],
        device=device
    )


def build_TriangularVFM_V2(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularVFM:
    """return TriangularVFM V2 (barycentric) estimator initialized from flat_hp dict."""
    path = BarycentricVfm1D(
        k=flat_hp["k"],
        vertex=0.5,
        eps=flat_hp["eps"]
    )
    return TriangularVFM(
        input_dim=input_dim,
        path=path,
        n_epochs=flat_hp["n_epochs"],
        lr=flat_hp["lr"],
        batch_size=flat_hp["batch_size"],
        eps=flat_hp["eps"],
        integration_steps=flat_hp["integration_steps"],
        device=device
    )


def build_TriangularVFM_V3(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularVFM2D:
    """return TriangularVFM V3 (2D stacked) estimator initialized from flat_hp dict."""
    path = Stacked2DVfm(
        k=flat_hp["k"],
        gamma_schedule=flat_hp["gamma_schedule"],
        t2_max=flat_hp["t2_max"],
        eps=flat_hp["eps"]
    )
    curve = Curve2D(path_height=flat_hp["path_height"])
    return TriangularVFM2D(
        input_dim=input_dim,
        path=path,
        curve=curve,
        n_epochs=flat_hp["n_epochs"],
        lr=flat_hp["lr"],
        batch_size=flat_hp["batch_size"],
        eps=flat_hp["eps"],
        integration_steps=flat_hp["integration_steps"],
        device=device
    )
