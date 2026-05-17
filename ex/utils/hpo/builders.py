"""universal builder functions for all DRE methods sampled from HPO.

maps flat scalar hyperparameter dicts (from hpo_search_spaces.py) into
fully-instantiated DRE instances. each builder corresponds
to one method and encapsulates estimator instantiation, classifier construction,
path/curve assembly, and hyperparameter routing. all builders are pure functions
with uniform signature: build_X(input_dim, device, num_waypoints, **flat_hp)
-> estimator.
"""

import torch

from src.methods.reg.common._time_samplers import (
    make_uniform, make_uniform_scaled, make_product,
    make_piecewise_sb_sampler, time_sampler_from_legacy_cfg,
)
from src.methods.reg.common._cfgs import OptimCfg, EmaCfg
from src.methods.reg.tsm import TSM
from src.methods.reg.ctsm import CTSM
from src.methods.reg.tsm.tri import TriangularTSM
from src.methods.reg.vfm import make_vfm
from src.methods.reg.fmdre import FMDRE
from src.methods.reg.fmdre.s2 import FMDRE_S2
from src.methods.reg.fmdre.tri import TriangularFMDRE
from src.methods.cls.bdre import BDRE
from src.methods.cls.mdre import MDRE
from src.methods.cls.tdre.mh import MultiHeadTDRE
from src.methods.cls.mdre.tri import TriangularMDRE
from src.methods.cls.tdre.mh_tri import MultiHeadTriangularTDRE
from src.methods.cls.tabular_plugin import (
    TabularPluginDRE,
    SmoothedTabularPluginDRE,
)

from src.methods.reg.ctsm.tri import TriangularCTSMV1 as TriangularCTSM
from src.methods.reg.ctsm.tri.v3 import TriangularCTSM2D
from src.methods.reg.vfm.tri import TriangularVFMV1 as TriangularVFM
from src.methods.reg.vfm.tri.v3 import TriangularVFM2D
from src.methods.reg.vfm import VFMOrthros

from src.waypoints.path_builders import (
    direct_vfm, psb, bary_ctsm, bary_vfm,
    rect_ctsm, rect_vfm,
)
from src.methods.reg.common._curves import LowArcCurve2D as Curve2D
from src.waypoints.waypoints1d import DefaultWaypointBuilder1D
from src.waypoints.triangular_waypoints import TriangularWaypointBuilder1D

from src.models.binary_classification import (
    make_binary_classifier,
    make_pairwise_binary_classifiers,
    make_multi_head_binary_classifier,
)
from src.models.multiclass_classification import make_multiclass_classifier


def build_TSM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TSM:
    """return TSM estimator initialized from flat_hp dict."""
    return TSM(input_dim=input_dim, device=device, **flat_hp)


def _optim_from_hp(flat_hp: dict) -> OptimCfg:
    """build OptimCfg from legacy flat hp keys (lr, grad_clip_norm)."""
    return OptimCfg(
        lr=flat_hp["lr"],
        grad_clip_norm=flat_hp.get("grad_clip_norm"),
    )


def _ema_from_hp(flat_hp: dict) -> EmaCfg:
    """build EmaCfg from legacy flat hp keys (ema_decay)."""
    decay = flat_hp.get("ema_decay")
    return EmaCfg(decay=decay) if decay is not None else EmaCfg()


def build_CTSM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> CTSM:
    """return stock CTSM estimator initialized from flat_hp dict."""
    eps = flat_hp.get("eps", 1e-3)
    time = time_sampler_from_legacy_cfg(
        flat_hp.get("time_dist", "uniform"), eps=eps, apply_iw=True,
    )
    return CTSM(
        input_dim=input_dim,
        device=device,
        time=time,
        sigma=flat_hp["sigma"],
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp.get("integration_steps", 1000),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        activation=flat_hp.get("activation", "elu"),
    )


def build_TriangularTSM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularTSM:
    """return TriangularTSM estimator initialized from flat_hp dict."""
    return TriangularTSM(input_dim=input_dim, device=device, **flat_hp)


def build_VFM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp):
    """return VFM estimator initialized from flat_hp dict."""
    return make_vfm(input_dim=input_dim, device=device, **flat_hp)


def build_FMDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> FMDRE:
    """return FMDRE estimator initialized from flat_hp dict. defaults to exact divergence."""
    flat_hp.setdefault("div_method", "exact")
    return FMDRE(input_dim=input_dim, device=device, **flat_hp)


def build_FMDRE_S2(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> FMDRE_S2:
    """return FMDRE_S2 estimator initialized from flat_hp dict. defaults to exact divergence."""
    flat_hp.setdefault("div_method", "exact")
    return FMDRE_S2(input_dim=input_dim, device=device, **flat_hp)


def build_TriangularFMDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularFMDRE:
    """return TriangularFMDRE estimator initialized from flat_hp dict. defaults to exact divergence."""
    flat_hp.setdefault("div_method", "exact")
    return TriangularFMDRE(input_dim=input_dim, device=device, **flat_hp)


def build_MHTTDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> MultiHeadTriangularTDRE:
    """return MHTTDRE estimator with multi-head classifier built from flat_hp dict.

    num_waypoints and waypoint-builder HPs can be HP-sampled via flat_hp;
    fall back to arg/defaults for num_waypoints, 0/1.0/0.5 for waypoint builder.
    """
    nwp = flat_hp.pop("num_waypoints", num_waypoints)
    if nwp is None:
        nwp = 5
    # pop waypoint-builder HPs with sensible defaults.
    midpoint_oversample = flat_hp.pop("midpoint_oversample", 0)
    gamma_power = flat_hp.pop("gamma_power", 1.0)
    vertex = flat_hp.pop("vertex", 0.5)
    classifier = make_multi_head_binary_classifier(
        input_dim=input_dim,
        num_heads=nwp - 1,
        hidden_dim=flat_hp["hidden_dim"],
        head_dim=flat_hp["head_dim"],
        num_shared_layers=flat_hp["num_shared_layers"],
        learning_rate=flat_hp["learning_rate"],
        num_epochs=flat_hp["num_epochs"],
        batch_size=flat_hp.get("batch_size"),
    )
    builder = TriangularWaypointBuilder1D(
        midpoint_oversample=midpoint_oversample,
        gamma_power=gamma_power,
        vertex=vertex,
    )
    return MultiHeadTriangularTDRE(
        classifier=classifier,
        waypoint_builder=builder,
        num_waypoints=nwp,
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
        num_epochs=flat_hp["num_epochs"],
        batch_size=flat_hp.get("batch_size"),
    )
    return MultiHeadTDRE(
        classifier=classifier,
        waypoint_builder=DefaultWaypointBuilder1D(),
        num_waypoints=nwp,
        device=device
    )


def build_TriangularCTSM_V1(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularCTSM:
    """return TriangularCTSM V1 (piecewise-SB) estimator initialized from flat_hp dict."""
    inner_eps = flat_hp.get("inner_eps", 0.0)
    vertex = flat_hp["vertex"]
    path = psb(sigma=flat_hp["sigma"], vertex=vertex, inner_eps=inner_eps, eps=flat_hp["eps"])
    # psb default sampler: avoid the forbidden band when inner_eps > 0; else uniform.
    if inner_eps > 0:
        time = make_piecewise_sb_sampler(vertex=vertex, inner_eps=inner_eps, eps=path.eps)
    else:
        time = time_sampler_from_legacy_cfg(
            flat_hp.get("time_dist", "uniform"), eps=path.eps, apply_iw=True,
        )
    return TriangularCTSM(
        input_dim=input_dim,
        path=path,
        time=time,
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp.get("integration_steps", 1000),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        activation=flat_hp.get("activation", "elu"),
        device=device,
    )


def build_TriangularCTSM_V2(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularCTSM:
    """return TriangularCTSM V2 (barycentric) estimator initialized from flat_hp dict."""
    path = bary_ctsm(
        sigma=flat_hp["sigma"],
        vertex=flat_hp.get("vertex", 0.5),
        eps=flat_hp["eps"],
    )
    time = time_sampler_from_legacy_cfg(
        flat_hp.get("time_dist", "uniform"), eps=path.eps, apply_iw=True,
    )
    return TriangularCTSM(
        input_dim=input_dim,
        path=path,
        time=time,
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp.get("integration_steps", 1000),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        activation=flat_hp.get("activation", "elu"),
        device=device,
    )


def build_TriangularCTSM_V3(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularCTSM2D:
    """return TriangularCTSM V3 (2D stacked) estimator initialized from flat_hp dict."""
    path = rect_ctsm(sigma=flat_hp["sigma"], eps=flat_hp["eps"])
    curve = Curve2D(path_height=flat_hp["path_height"])
    time = make_product(
        make_uniform(eps=path.eps),
        make_uniform_scaled(eps=path.eps, max=flat_hp["t2_max"]),
    )
    return TriangularCTSM2D(
        input_dim=input_dim,
        path=path,
        time=time,
        curve=curve,
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp.get("integration_steps", 1000),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        activation=flat_hp.get("activation", "elu"),
        device=device,
    )


def build_TriangularVFM_V1(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularVFM:
    """return TriangularVFM V1 (piecewise-SB) estimator initialized from flat_hp dict."""
    inner_eps = flat_hp.get("inner_eps", 0.0)
    vertex = flat_hp["vertex"]
    path = psb(
        sigma=flat_hp["sigma"], vertex=vertex,
        gamma_min=flat_hp["gamma_min"],
        inner_eps=inner_eps, eps=flat_hp["eps"],
    )
    if inner_eps > 0:
        time = make_piecewise_sb_sampler(vertex=vertex, inner_eps=inner_eps, eps=path.eps)
    else:
        time = time_sampler_from_legacy_cfg(
            flat_hp.get("time_dist", "uniform"), eps=path.eps, apply_iw=True,
        )
    return TriangularVFM(
        input_dim=input_dim,
        path=path,
        time=time,
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp["integration_steps"],
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        activation=flat_hp.get("activation", "gelu"),
        div_method=flat_hp.get("div_method", "exact"),
        device=device,
    )


def build_TriangularVFM_V2(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularVFM:
    """return TriangularVFM V2 (barycentric) estimator initialized from flat_hp dict."""
    path = bary_vfm(
        k=flat_hp["k"],
        vertex=flat_hp.get("vertex", 0.5),
        eps=flat_hp["eps"],
    )
    time = time_sampler_from_legacy_cfg(
        flat_hp.get("time_dist", "uniform"), eps=path.eps, apply_iw=True,
    )
    return TriangularVFM(
        input_dim=input_dim,
        path=path,
        time=time,
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp["integration_steps"],
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        activation=flat_hp.get("activation", "gelu"),
        div_method=flat_hp.get("div_method", "exact"),
        device=device,
    )


def build_TriangularVFM_V3(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularVFM2D:
    """return TriangularVFM V3 (2D stacked) estimator initialized from flat_hp dict."""
    path = rect_vfm(k=flat_hp["k"], eps=flat_hp["eps"])
    curve = Curve2D(path_height=flat_hp["path_height"])
    time = make_product(
        make_uniform(eps=path.eps),
        make_uniform_scaled(eps=path.eps, max=flat_hp["t2_max"]),
    )
    return TriangularVFM2D(
        input_dim=input_dim,
        path=path,
        time=time,
        curve=curve,
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp["integration_steps"],
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        activation=flat_hp.get("activation", "gelu"),
        div_method=flat_hp.get("div_method", "exact"),
        device=device,
    )


def build_VFMOrthros(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> VFMOrthros:
    """return VFMOrthros estimator initialized from flat_hp dict."""
    # gamma_min: noise floor for stable orthros inference (see VFMOrthros docstring)
    gamma_min = flat_hp.get("gamma_min", 0.1)
    path = direct_vfm(k=flat_hp["k"], gamma_min=gamma_min, eps=flat_hp["eps"])
    time = time_sampler_from_legacy_cfg(
        flat_hp.get("time_dist", "uniform"), eps=path.eps, apply_iw=True,
    )
    return VFMOrthros(
        input_dim=input_dim,
        path=path,
        time=time,
        test_gamma_min=gamma_min,
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp["integration_steps"],
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_shared_layers=flat_hp.get("n_shared_layers", 2),
        activation=flat_hp.get("activation", "gelu"),
        n_hutch_samples=flat_hp.get("n_hutch_samples", 1),
        device=device,
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
    hyperparams, then wraps in TriangularMDRE with waypoint_builder constructed
    from midpoint_oversample, gamma_power, and vertex HPs. classifier_name
    defaults to "default"; override via flat_hp["classifier_name"] if needed.
    """
    classifier_name = flat_hp.pop("classifier_name", "default")
    nwp = flat_hp.pop("num_waypoints", num_waypoints)
    if nwp is None:
        nwp = 10
    # pop waypoint-builder HPs with sensible defaults.
    midpoint_oversample = flat_hp.pop("midpoint_oversample", 0)
    gamma_power = flat_hp.pop("gamma_power", 1.0)
    vertex = flat_hp.pop("vertex", 0.5)
    # estimator-only HPs popped before forwarding to classifier factory.
    max_train_samples = flat_hp.pop("max_train_samples", None)
    classifier = make_multiclass_classifier(
        name=classifier_name, input_dim=input_dim, num_classes=nwp, **flat_hp,
    )
    builder = TriangularWaypointBuilder1D(
        midpoint_oversample=midpoint_oversample,
        gamma_power=gamma_power,
        vertex=vertex,
    )
    return TriangularMDRE(
        classifier=classifier,
        waypoint_builder=builder,
        device=device,
        max_train_samples=max_train_samples,
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


# registry mapping method label -> builder callable. consumed by the optuna
# worker (ex.utils.hpo.optuna.worker.run_worker) to resolve a builder
# from METADATA['builder'] in a suggest_hp module.
BUILDERS_REGISTRY = {
    "TSM": build_TSM,
    "CTSM": build_CTSM,
    "VFM": build_VFM,
    "VFMOrthros": build_VFMOrthros,
    "BDRE": build_BDRE,
    "MDRE": build_MDRE,
    "MultiHeadTDRE": build_MHTDRE,
    "FMDRE": build_FMDRE,
    "FMDRE_S2": build_FMDRE_S2,
    "TriangularFMDRE": build_TriangularFMDRE,
    "TriangularTSM": build_TriangularTSM,
    "MultiHeadTriangularTDRE": build_MHTTDRE,
    "TriangularMDRE": build_TriangularMDRE,
    "TriangularCTSM_V1": build_TriangularCTSM_V1,
    "TriangularCTSM_V2": build_TriangularCTSM_V2,
    "TriangularCTSM_V3": build_TriangularCTSM_V3,
    "TriangularVFM_V1": build_TriangularVFM_V1,
    "TriangularVFM_V2": build_TriangularVFM_V2,
    "TriangularVFM_V3": build_TriangularVFM_V3,
    "TabularPluginDRE": build_TabularPluginDRE,
    "SmoothedTabularPluginDRE": build_SmoothedTabularPluginDRE,
}
