"""universal builder functions for all DRE methods sampled from HPO.

maps flat scalar hyperparameter dicts (from hpo_search_spaces.py) into
fully-instantiated DensityRatioEstimator instances. each builder corresponds
to one method and encapsulates estimator instantiation, classifier construction,
path/curve assembly, and hyperparameter routing. all builders are pure functions
with uniform signature: build_X(input_dim, device, num_waypoints, **flat_hp)
-> estimator.
"""

import torch

from src.density_ratio_estimation.tsm import TSM
from src.density_ratio_estimation.ctsm import CTSM
from src.density_ratio_estimation.triangular_tsm import TriangularTSM
from src.density_ratio_estimation.spatial_adapters import make_spatial_velo_denoiser
from src.density_ratio_estimation.fmdre import FMDRE
from src.density_ratio_estimation.fmdre_s2 import FMDRE_S2
from src.density_ratio_estimation.triangular_fmdre import TriangularFMDRE
from src.density_ratio_estimation.bdre import BDRE
from src.density_ratio_estimation.mdre import MDRE
from src.density_ratio_estimation.mh_tdre import MultiHeadTDRE
from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
from src.density_ratio_estimation.mh_triangular_tdre import MultiHeadTriangularTDRE
from src.density_ratio_estimation.tabular_plugin import (
    TabularPluginDRE,
    SmoothedTabularPluginDRE,
)

from src.density_ratio_estimation.triangular_ctsm import TriangularCTSM
from src.density_ratio_estimation.triangular_ctsm_2d import TriangularCTSM2D
from src.density_ratio_estimation.triangular_vfm import TriangularVFM
from src.density_ratio_estimation.triangular_vfm_2d import TriangularVFM2D

from src.waypoints.piecewise_sb import PiecewiseSBCtsm1D, PiecewiseSBVfm1D
from src.waypoints.triangular_continuous import BarycentricCtsm1D, BarycentricVfm1D
from src.waypoints.triangular_continuous_2d import Stacked2DCtsm, Stacked2DVfm
from src.waypoints.curve_2d import Curve2D
from src.waypoints.waypoints1d import DefaultWaypointBuilder1D

from src.models.binary_classification import (
    make_binary_classifier,
    make_pairwise_binary_classifiers,
    make_multi_head_binary_classifier,
)
from src.models.multiclass_classification import make_multiclass_classifier


def build_TSM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TSM:
    """return TSM estimator initialized from flat_hp dict."""
    return TSM(input_dim=input_dim, device=device, **flat_hp)


def build_CTSM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> CTSM:
    """return CTSM estimator initialized from flat_hp dict."""
    return CTSM(input_dim=input_dim, device=device, **flat_hp)


def build_TriangularTSM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularTSM:
    """return TriangularTSM estimator initialized from flat_hp dict."""
    return TriangularTSM(input_dim=input_dim, device=device, **flat_hp)


def build_VFM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp):
    """return VFM estimator initialized from flat_hp dict."""
    return make_spatial_velo_denoiser(input_dim=input_dim, device=device, **flat_hp)


def build_FMDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> FMDRE:
    """return FMDRE estimator initialized from flat_hp dict."""
    return FMDRE(input_dim=input_dim, device=device, **flat_hp)


def build_FMDRE_S2(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> FMDRE_S2:
    """return FMDRE_S2 estimator initialized from flat_hp dict."""
    return FMDRE_S2(input_dim=input_dim, device=device, **flat_hp)


def build_TriangularFMDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularFMDRE:
    """return TriangularFMDRE estimator initialized from flat_hp dict."""
    return TriangularFMDRE(input_dim=input_dim, device=device, **flat_hp)


def build_MHTTDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> MultiHeadTriangularTDRE:
    """return MHTTDRE estimator with multi-head classifier built from flat_hp dict.

    num_waypoints and vertex can be HP-sampled via flat_hp; fall back to arg/0.5.
    """
    nwp = flat_hp.pop("num_waypoints", num_waypoints)
    if nwp is None:
        nwp = 5
    vertex = flat_hp.pop("vertex", 0.5)
    classifier = make_multi_head_binary_classifier(
        input_dim=input_dim,
        num_heads=nwp - 1,
        hidden_dim=flat_hp["hidden_dim"],
        head_dim=flat_hp["head_dim"],
        num_shared_layers=flat_hp["num_shared_layers"],
        learning_rate=flat_hp["learning_rate"],
        num_epochs=flat_hp["num_epochs"]
    )
    return MultiHeadTriangularTDRE(
        classifier=classifier,
        num_waypoints=nwp,
        vertex=vertex,
        device=device
    )


def build_MHTDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> MultiHeadTDRE:
    """return MHTDRE estimator with multi-head classifier built from flat_hp dict.

    num_waypoints can be HP-sampled via flat_hp; falls back to arg, then 10.
    """
    nwp = flat_hp.pop("num_waypoints", num_waypoints)
    if nwp is None:
        nwp = 10
    classifier = make_multi_head_binary_classifier(
        input_dim=input_dim,
        num_heads=nwp - 1,
        hidden_dim=flat_hp["hidden_dim"],
        head_dim=flat_hp["head_dim"],
        num_shared_layers=flat_hp["num_shared_layers"],
        learning_rate=flat_hp["learning_rate"],
        num_epochs=flat_hp["num_epochs"]
    )
    return MultiHeadTDRE(
        classifier=classifier,
        waypoint_builder=DefaultWaypointBuilder1D(),
        num_waypoints=nwp,
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
        integration_steps=flat_hp.get("integration_steps", 1000),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
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
        integration_steps=flat_hp.get("integration_steps", 1000),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
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
        integration_steps=flat_hp.get("integration_steps", 1000),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
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
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
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
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
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
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        device=device
    )


def build_BDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> BDRE:
    """return BDRE estimator with binary classifier.

    flat_hp is forwarded to make_binary_classifier; missing keys use class defaults.
    classifier_name defaults to "default"; override via flat_hp["classifier_name"].
    """
    classifier_name = flat_hp.pop("classifier_name", "default")
    classifier = make_binary_classifier(name=classifier_name, input_dim=input_dim, **flat_hp)
    return BDRE(classifier=classifier, device=device)


def build_MDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> MDRE:
    """return MDRE estimator with multiclass classifier (num_classes = num_waypoints).

    num_waypoints can be HP-sampled via flat_hp; falls back to arg, then 10.
    flat_hp is forwarded to make_multiclass_classifier; missing keys use class defaults.
    """
    nwp = flat_hp.pop("num_waypoints", num_waypoints)
    if nwp is None:
        nwp = 10
    classifier_name = flat_hp.pop("classifier_name", "default")
    classifier = make_multiclass_classifier(
        name=classifier_name, input_dim=input_dim, num_classes=nwp, **flat_hp,
    )
    return MDRE(classifier=classifier, device=device)


def build_TriangularMDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularMDRE:
    """return TriangularMDRE estimator with triangular-path multiclass classifier.

    constructs multiclass classifier with num_waypoints classes and training
    hyperparams, then wraps in TriangularMDRE with midpoint oversampling and
    gamma power params. classifier_name defaults to "default"; override via
    flat_hp["classifier_name"] if needed.
    """
    classifier_name = flat_hp.pop("classifier_name", "default")
    vertex = flat_hp.pop("vertex", 0.5)
    nwp = flat_hp.pop("num_waypoints", num_waypoints)
    if nwp is None:
        nwp = 10
    midpoint_oversample = flat_hp.pop("midpoint_oversample")
    gamma_power = flat_hp.pop("gamma_power")
    classifier = make_multiclass_classifier(
        name=classifier_name, input_dim=input_dim, num_classes=nwp, **flat_hp,
    )
    return TriangularMDRE(
        classifier=classifier,
        device=device,
        midpoint_oversample=midpoint_oversample,
        gamma_power=gamma_power,
        vertex=vertex,
    )


def build_TabularPluginDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TabularPluginDRE:
    """return TabularPluginDRE (empirical plug-in) estimator.

    TabularPluginDRE is a counting-based method with no HPO-tunable continuous
    parameters. all config is passed as fixed kwargs set by the experiment.
    """
    return TabularPluginDRE(
        n_states=flat_hp["n_states"],
        n_actions=flat_hp["n_actions"],
        encoding_cfg=flat_hp["encoding_cfg"],
        decode=flat_hp["decode"],
        smoothing_alpha=flat_hp.get("smoothing_alpha", 0.5),
        device=device
    )


def build_SmoothedTabularPluginDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> SmoothedTabularPluginDRE:
    """return SmoothedTabularPluginDRE (oracle smoothed plug-in) estimator.

    SmoothedTabularPluginDRE is an oracle estimator with no HPO-tunable
    continuous parameters. all config is passed as fixed kwargs set by the
    experiment.
    """
    return SmoothedTabularPluginDRE(
        n_states=flat_hp["n_states"],
        n_actions=flat_hp["n_actions"],
        encoding_cfg=flat_hp["encoding_cfg"],
        smoothing_alpha=flat_hp.get("smoothing_alpha", 0.5),
        device=device
    )
