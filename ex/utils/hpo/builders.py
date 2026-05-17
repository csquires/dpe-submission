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
from src.methods.reg.common._cfgs import OptimCfg, SchedCfg, EmaCfg, TimeCfg
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
    direct_1d, bary_1d, psb_1d, rect_2d,
    stiff_noise, bridge_noise, stiff_noise_2d, bridge_noise_2d,
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


# ---------------------------------------------------------------------------
# cfg-translation helpers: flat HPO scalars -> config dataclasses.
# ---------------------------------------------------------------------------


def _optim_from_hp(flat_hp: dict) -> OptimCfg:
    """build OptimCfg from flat hp keys (lr, grad_clip_norm, weight_decay)."""
    return OptimCfg(
        lr=flat_hp["lr"],
        grad_clip_norm=flat_hp.get("grad_clip_norm"),
        weight_decay=flat_hp.get("weight_decay", 0.0),
    )


def _ema_from_hp(flat_hp: dict) -> EmaCfg:
    """build EmaCfg from flat hp keys (ema_decay)."""
    decay = flat_hp.get("ema_decay")
    return EmaCfg(decay=decay) if decay is not None else EmaCfg()


def _sched_from_hp(flat_hp: dict) -> SchedCfg:
    """build SchedCfg from flat hp keys (cosine_min_factor).

    cosine_min_factor defaults to 1.0 (annealing off, byte-identical to the
    legacy SchedCfg() default).
    """
    return SchedCfg(cosine_min_factor=flat_hp.get("cosine_min_factor", 1.0))


def _time_from_hp(flat_hp: dict, *, eps: float) -> TimeCfg:
    """build TimeCfg from flat hp keys (time_dist, apply_iw) at the given eps."""
    return TimeCfg.from_dist(
        flat_hp.get("time_dist", "uniform"),
        eps=eps,
        apply_iw=flat_hp.get("apply_iw", True),
    )


# ---------------------------------------------------------------------------
# noise-schedule helpers: build a Sched1D/Sched2D from flat hp keys.
# ---------------------------------------------------------------------------
# the schedule TYPE (`sched`: "stiff" vs "bridge") and the amplitude `sigma`
# are HPO-selectable; `k` parameterizes the stiff sigmoid-product gamma.


def _sched_1d(flat_hp: dict, *, default_k: float = 20.0):
    """resolve a 1d noise schedule (Sched1D) from flat hp.

    sched="stiff" -> stiff_noise(k, sigma); sched="bridge" -> bridge_noise(sigma).
    """
    kind = flat_hp.get("sched", "stiff")
    sigma = flat_hp.get("sigma", 1.0)
    if kind == "stiff":
        return stiff_noise(k=flat_hp.get("k", default_k), sigma=sigma)
    if kind == "bridge":
        return bridge_noise(sigma=sigma)
    raise ValueError(f"sched must be 'stiff' or 'bridge'; got {kind!r}")


def _sched_2d(flat_hp: dict, *, default_k: float = 20.0):
    """resolve a 2d noise schedule (Sched2D) from flat hp.

    sched="stiff" -> stiff_noise_2d(k, sigma); sched="bridge" -> bridge_noise_2d(sigma).
    """
    kind = flat_hp.get("sched", "stiff")
    sigma = flat_hp.get("sigma", 1.0)
    if kind == "stiff":
        return stiff_noise_2d(k=flat_hp.get("k", default_k), sigma=sigma)
    if kind == "bridge":
        return bridge_noise_2d(sigma=sigma)
    raise ValueError(f"sched must be 'stiff' or 'bridge'; got {kind!r}")


def build_TSM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TSM:
    """return TSM estimator initialized from flat_hp dict.

    TSM takes config objects (optim, ema, time); flat HPO scalars are translated
    explicitly via the cfg helpers, never forwarded raw.
    """
    eps = flat_hp.get("eps", 1e-3)
    return TSM(
        input_dim=input_dim,
        device=device,
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        time=_time_from_hp(flat_hp, eps=eps),
        reweight=flat_hp.get("reweight", False),
        activation=flat_hp.get("activation", "silu"),
        integration_steps=flat_hp.get("integration_steps", 200),
    )


def build_CTSM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> CTSM:
    """return stock CTSM estimator initialized from flat_hp dict.

    builds explicit train and test direct paths via the general direct_1d builder
    so the noise schedule (sched, sigma) is fully HPO-selectable for both paths.
    """
    eps = flat_hp.get("eps", 1e-3)
    gamma_min = flat_hp.get("gamma_min", 0.0)
    inner_eps = flat_hp.get("inner_eps", 0.0)
    path = direct_1d(
        sched=_sched_1d(flat_hp), inner_eps=inner_eps, gamma_min=gamma_min, eps=eps,
    )
    test_eps = flat_hp.get("test_eps", eps)
    test_path = direct_1d(
        sched=_sched_1d(_test_sched_hp(flat_hp)),
        inner_eps=flat_hp.get("test_inner_eps", 0.0),
        gamma_min=flat_hp.get("test_gamma_min", 0.0),
        eps=test_eps,
    )
    time = time_sampler_from_legacy_cfg(
        flat_hp.get("time_dist", "uniform"), eps=path.eps,
        apply_iw=flat_hp.get("apply_iw", True),
    )
    return CTSM(
        input_dim=input_dim,
        device=device,
        path=path,
        test_path=test_path,
        time=time,
        sigma=flat_hp.get("sigma", 1.0),
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp.get("integration_steps", 1000),
        activation=flat_hp.get("activation", "elu"),
        reweight=flat_hp.get("reweight", False),
    )


def _test_sched_hp(flat_hp: dict) -> dict:
    """view of flat_hp with test_* schedule keys aliased onto sched/sigma/k.

    lets the schedule helpers build an independent test-path schedule from the
    test_sched / test_sigma search keys (falling back to the train values).
    """
    return {
        "sched": flat_hp.get("test_sched", flat_hp.get("sched", "stiff")),
        "sigma": flat_hp.get("test_sigma", flat_hp.get("sigma", 1.0)),
        "k": flat_hp.get("test_k", flat_hp.get("k", 20.0)),
    }


def build_TriangularTSM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularTSM:
    """return TriangularTSM estimator initialized from flat_hp dict.

    TriangularTSM takes config objects (optim, ema, time); flat HPO scalars are
    translated explicitly. its bell path is parameterized by vertex/peak_max, not
    a noise schedule, so there is no sched/sigma wiring here.
    """
    eps = flat_hp.get("eps", 1e-3)
    return TriangularTSM(
        input_dim=input_dim,
        device=device,
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        time=_time_from_hp(flat_hp, eps=eps),
        reweight=flat_hp.get("reweight", False),
        vertex=flat_hp.get("vertex", 0.5),
        peak_max=flat_hp.get("peak_max", 1.0),
        activation=flat_hp.get("activation", "silu"),
    )


def build_VFM(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp):
    """return VFM estimator initialized from flat_hp dict.

    builds explicit train and test direct paths via direct_1d so the noise
    schedule (sched, sigma, k) is HPO-selectable for both paths; passes the
    paths directly so VFM never falls back to its internal direct_vfm default.
    """
    eps = flat_hp.get("eps", 1e-3)
    path = direct_1d(
        sched=_sched_1d(flat_hp),
        inner_eps=flat_hp.get("inner_eps", 0.0),
        gamma_min=flat_hp.get("gamma_min", 0.0),
        eps=eps,
    )
    test_path = direct_1d(
        sched=_sched_1d(_test_sched_hp(flat_hp)),
        inner_eps=flat_hp.get("test_inner_eps", 0.0),
        gamma_min=flat_hp.get("test_gamma_min", 0.0),
        eps=flat_hp.get("test_eps", eps),
    )
    time = time_sampler_from_legacy_cfg(
        flat_hp.get("time_dist", "uniform"), eps=path.eps,
        apply_iw=flat_hp.get("apply_iw", True),
    )
    return make_vfm(
        input_dim=input_dim,
        device=device,
        path=path,
        test_path=test_path,
        time=time,
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        lr=flat_hp["lr"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp.get("integration_steps", 3000),
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        activation=flat_hp.get("activation", "gelu"),
        layernorm=flat_hp.get("layernorm", "off"),
        antithetic=flat_hp.get("antithetic", True),
        reweight=flat_hp.get("reweight", False),
        precond=flat_hp.get("precond", False),
        div_method=flat_hp.get("div_method", "hutchinson"),
        div_noise=flat_hp.get("div_noise", "rademacher"),
        n_hutch_samples=flat_hp.get("n_hutch_samples", 1),
    )


def build_FMDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> FMDRE:
    """return FMDRE estimator initialized from flat_hp dict.

    FMDRE takes config objects (optim, ema, time); flat HPO scalars are translated
    explicitly. div_method is deliberately pinned to "exact".
    """
    eps = flat_hp.get("eps", 1e-3)
    return FMDRE(
        input_dim=input_dim,
        device=device,
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        time=_time_from_hp(flat_hp, eps=eps),
        score_weight=flat_hp.get("score_weight", 1.0),
        # FMDRE-family div_method is deliberately pinned "exact" (not an HPO knob)
        div_method="exact",
        integration_steps=flat_hp.get("integration_steps", 10000),
        reweight=flat_hp.get("reweight", False),
        precond=flat_hp.get("precond", False),
    )


def build_FMDRE_S2(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> FMDRE_S2:
    """return FMDRE_S2 estimator initialized from flat_hp dict.

    FMDRE_S2 takes config objects (optim, ema, time); flat HPO scalars are
    translated explicitly. div_method is deliberately pinned to "exact".
    """
    eps = flat_hp.get("eps", 1e-3)
    return FMDRE_S2(
        input_dim=input_dim,
        device=device,
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        time=_time_from_hp(flat_hp, eps=eps),
        score_weight=flat_hp.get("score_weight", 1.0),
        # FMDRE-family div_method is deliberately pinned "exact" (not an HPO knob)
        div_method="exact",
        integration_steps=flat_hp.get("integration_steps", 10000),
        p_uncond=flat_hp.get("p_uncond", 0.1),
        # sentinel_cond: FMDRE_S2 internal CFG sentinel (not an HPO knob)
        reweight=flat_hp.get("reweight", False),
        precond=flat_hp.get("precond", False),
    )


def build_TriangularFMDRE(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularFMDRE:
    """return TriangularFMDRE estimator initialized from flat_hp dict.

    TriangularFMDRE takes config objects (optim, ema, time); flat HPO scalars are
    translated explicitly. div_method is deliberately pinned to "exact".
    """
    eps = flat_hp.get("eps", 1e-3)
    return TriangularFMDRE(
        input_dim=input_dim,
        device=device,
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        time=_time_from_hp(flat_hp, eps=eps),
        score_weight=flat_hp.get("score_weight", 1.0),
        # FMDRE-family div_method is deliberately pinned "exact" (not an HPO knob)
        div_method="exact",
        integration_steps=flat_hp.get("integration_steps", 10000),
        triangular_p_uncond=flat_hp.get("triangular_p_uncond", 0.0),
        layernorm=flat_hp.get("layernorm", "off"),
        reweight=flat_hp.get("reweight", False),
        precond=flat_hp.get("precond", False),
    )


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
        weight_decay=flat_hp.get("weight_decay", 0.0),
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
        weight_decay=flat_hp.get("weight_decay", 0.0),
    )
    return MultiHeadTDRE(
        classifier=classifier,
        waypoint_builder=DefaultWaypointBuilder1D(),
        num_waypoints=nwp,
        device=device
    )


def build_TriangularCTSM_V1(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularCTSM:
    """return TriangularCTSM V1 (piecewise-SB) estimator initialized from flat_hp dict.

    builds explicit train and test psb paths via the general psb_1d builder so the
    noise schedule (sched, sigma) is HPO-selectable for both paths.
    """
    inner_eps = flat_hp.get("inner_eps", 0.0)
    vertex = flat_hp["vertex"]
    eps = flat_hp["eps"]
    path = psb_1d(
        sched=_sched_1d(flat_hp), vertex=vertex,
        inner_eps=inner_eps, gamma_min=flat_hp.get("gamma_min", 0.0), eps=eps,
    )
    test_path = psb_1d(
        sched=_sched_1d(_test_sched_hp(flat_hp)), vertex=vertex,
        inner_eps=flat_hp.get("test_inner_eps", 0.0),
        gamma_min=flat_hp.get("test_gamma_min", 0.0),
        eps=flat_hp.get("test_eps", eps),
    )
    # psb default sampler: avoid the forbidden band when inner_eps > 0; else uniform.
    if inner_eps > 0:
        time = make_piecewise_sb_sampler(vertex=vertex, inner_eps=inner_eps, eps=path.eps)
    else:
        time = time_sampler_from_legacy_cfg(
            flat_hp.get("time_dist", "uniform"), eps=path.eps,
            apply_iw=flat_hp.get("apply_iw", True),
        )
    return TriangularCTSM(
        input_dim=input_dim,
        path=path,
        test_path=test_path,
        time=time,
        sigma=flat_hp.get("sigma", 1.0),
        vertex=vertex,
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp.get("integration_steps", 1000),
        activation=flat_hp.get("activation", "elu"),
        reweight=flat_hp.get("reweight", False),
        device=device,
    )


def build_TriangularCTSM_V2(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularCTSM:
    """return TriangularCTSM V2 (barycentric) estimator initialized from flat_hp dict.

    builds explicit train and test barycentric paths via the general bary_1d
    builder so the noise schedule (sched, sigma) is HPO-selectable for both paths.
    """
    vertex = flat_hp.get("vertex", 0.5)
    eps = flat_hp["eps"]
    path = bary_1d(
        sched=_sched_1d(flat_hp), vertex=vertex,
        inner_eps=flat_hp.get("inner_eps", 0.0),
        gamma_min=flat_hp.get("gamma_min", 0.0), eps=eps,
    )
    test_path = bary_1d(
        sched=_sched_1d(_test_sched_hp(flat_hp)), vertex=vertex,
        inner_eps=flat_hp.get("test_inner_eps", 0.0),
        gamma_min=flat_hp.get("test_gamma_min", 0.0),
        eps=flat_hp.get("test_eps", eps),
    )
    time = time_sampler_from_legacy_cfg(
        flat_hp.get("time_dist", "uniform"), eps=path.eps,
        apply_iw=flat_hp.get("apply_iw", True),
    )
    return TriangularCTSM(
        input_dim=input_dim,
        path=path,
        test_path=test_path,
        time=time,
        sigma=flat_hp.get("sigma", 1.0),
        vertex=vertex,
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp.get("integration_steps", 1000),
        activation=flat_hp.get("activation", "elu"),
        reweight=flat_hp.get("reweight", False),
        device=device,
    )


def build_TriangularCTSM_V3(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularCTSM2D:
    """return TriangularCTSM V3 (2D stacked) estimator initialized from flat_hp dict.

    builds explicit train and test rect-2d paths via the general rect_2d builder
    so the noise schedule (sched, sigma) is HPO-selectable for both paths.
    """
    eps = flat_hp["eps"]
    path = rect_2d(
        sched=_sched_2d(flat_hp),
        inner_eps=flat_hp.get("inner_eps", 0.02),
        gamma_min=flat_hp.get("gamma_min", 0.0), eps=eps,
    )
    test_path = rect_2d(
        sched=_sched_2d(_test_sched_hp(flat_hp)),
        inner_eps=flat_hp.get("test_inner_eps", 0.0),
        gamma_min=flat_hp.get("test_gamma_min", 0.0),
        eps=flat_hp.get("test_eps", eps),
    )
    curve = Curve2D(path_height=flat_hp["path_height"])
    time = make_product(
        make_uniform(eps=path.eps),
        make_uniform_scaled(eps=path.eps, max=flat_hp["t2_max"]),
    )
    return TriangularCTSM2D(
        input_dim=input_dim,
        path=path,
        test_path=test_path,
        time=time,
        curve=curve,
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp.get("integration_steps", 1000),
        activation=flat_hp.get("activation", "elu"),
        reweight=flat_hp.get("reweight", False),
        device=device,
    )


def build_TriangularVFM_V1(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularVFM:
    """return TriangularVFM V1 (piecewise-SB) estimator initialized from flat_hp dict.

    builds explicit train and test psb paths via the general psb_1d builder so the
    noise schedule (sched, sigma) is HPO-selectable for both paths. note: V1's
    barycentric `k` constructor scalar is vestigial -- the psb path built here is
    passed via `path=`, so the estimator never reads `k`.
    """
    inner_eps = flat_hp.get("inner_eps", 0.0)
    vertex = flat_hp["vertex"]
    eps = flat_hp["eps"]
    gamma_min = flat_hp["gamma_min"]
    path = psb_1d(
        sched=_sched_1d(flat_hp), vertex=vertex,
        gamma_min=gamma_min, inner_eps=inner_eps, eps=eps,
    )
    test_path = psb_1d(
        sched=_sched_1d(_test_sched_hp(flat_hp)), vertex=vertex,
        gamma_min=flat_hp.get("test_gamma_min", gamma_min),
        inner_eps=flat_hp.get("test_inner_eps", 0.0),
        eps=flat_hp.get("test_eps", eps),
    )
    if inner_eps > 0:
        time = make_piecewise_sb_sampler(vertex=vertex, inner_eps=inner_eps, eps=path.eps)
    else:
        time = time_sampler_from_legacy_cfg(
            flat_hp.get("time_dist", "uniform"), eps=path.eps,
            apply_iw=flat_hp.get("apply_iw", True),
        )
    return TriangularVFM(
        input_dim=input_dim,
        path=path,
        test_path=test_path,
        time=time,
        vertex=vertex,
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp["integration_steps"],
        activation=flat_hp.get("activation", "gelu"),
        layernorm=flat_hp.get("layernorm", "off"),
        antithetic=flat_hp.get("antithetic", True),
        reweight=flat_hp.get("reweight", False),
        precond=flat_hp.get("precond", False),
        div_method=flat_hp.get("div_method", "hutchinson"),
        div_noise=flat_hp.get("div_noise", "rademacher"),
        n_hutch_samples=flat_hp.get("n_hutch_samples", 1),
        device=device,
    )


def build_TriangularVFM_V2(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularVFM:
    """return TriangularVFM V2 (barycentric) estimator initialized from flat_hp dict.

    builds explicit train and test barycentric paths via the general bary_1d
    builder so the noise schedule (sched, sigma) is HPO-selectable for both paths.
    """
    vertex = flat_hp.get("vertex", 0.5)
    eps = flat_hp["eps"]
    path = bary_1d(
        sched=_sched_1d(flat_hp), vertex=vertex,
        inner_eps=flat_hp.get("inner_eps", 0.0),
        gamma_min=flat_hp.get("gamma_min", 0.0), eps=eps,
    )
    test_path = bary_1d(
        sched=_sched_1d(_test_sched_hp(flat_hp)), vertex=vertex,
        inner_eps=flat_hp.get("test_inner_eps", 0.0),
        gamma_min=flat_hp.get("test_gamma_min", 0.0),
        eps=flat_hp.get("test_eps", eps),
    )
    time = time_sampler_from_legacy_cfg(
        flat_hp.get("time_dist", "uniform"), eps=path.eps,
        apply_iw=flat_hp.get("apply_iw", True),
    )
    return TriangularVFM(
        input_dim=input_dim,
        path=path,
        test_path=test_path,
        time=time,
        vertex=vertex,
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp["integration_steps"],
        activation=flat_hp.get("activation", "gelu"),
        layernorm=flat_hp.get("layernorm", "off"),
        antithetic=flat_hp.get("antithetic", True),
        reweight=flat_hp.get("reweight", False),
        precond=flat_hp.get("precond", False),
        div_method=flat_hp.get("div_method", "hutchinson"),
        div_noise=flat_hp.get("div_noise", "rademacher"),
        n_hutch_samples=flat_hp.get("n_hutch_samples", 1),
        device=device,
    )


def build_TriangularVFM_V3(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> TriangularVFM2D:
    """return TriangularVFM V3 (2D stacked) estimator initialized from flat_hp dict.

    builds explicit train and test rect-2d paths via the general rect_2d builder
    so the noise schedule (sched, sigma) is HPO-selectable for both paths.
    """
    eps = flat_hp["eps"]
    path = rect_2d(
        sched=_sched_2d(flat_hp),
        inner_eps=flat_hp.get("inner_eps", 0.0),
        gamma_min=flat_hp.get("gamma_min", 0.05), eps=eps,
    )
    test_path = rect_2d(
        sched=_sched_2d(_test_sched_hp(flat_hp)),
        inner_eps=flat_hp.get("test_inner_eps", 0.0),
        gamma_min=flat_hp.get("test_gamma_min", 0.0),
        eps=flat_hp.get("test_eps", eps),
    )
    curve = Curve2D(path_height=flat_hp["path_height"])
    time = make_product(
        make_uniform(eps=path.eps),
        make_uniform_scaled(eps=path.eps, max=flat_hp["t2_max"]),
    )
    return TriangularVFM2D(
        input_dim=input_dim,
        path=path,
        test_path=test_path,
        time=time,
        curve=curve,
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp["integration_steps"],
        activation=flat_hp.get("activation", "gelu"),
        layernorm=flat_hp.get("layernorm", "off"),
        antithetic=flat_hp.get("antithetic", True),
        reweight=flat_hp.get("reweight", False),
        div_method=flat_hp.get("div_method", "hutchinson"),
        div_noise=flat_hp.get("div_noise", "rademacher"),
        n_hutch_samples=flat_hp.get("n_hutch_samples", 1),
        device=device,
    )


def build_VFMOrthros(input_dim: int, device: str | torch.device, num_waypoints: int, **flat_hp) -> VFMOrthros:
    """return VFMOrthros estimator initialized from flat_hp dict.

    builds explicit train and test direct paths via direct_1d so the noise
    schedule (sched, sigma, k) and gamma_min/test_eps are HPO-selectable for both
    paths; passes test_path directly so VFMOrthros never falls back to its
    internal direct_vfm default.
    """
    # gamma_min: noise floor for stable orthros inference (see VFMOrthros docstring)
    gamma_min = flat_hp.get("gamma_min", 0.1)
    eps = flat_hp["eps"]
    path = direct_1d(
        sched=_sched_1d(flat_hp),
        inner_eps=flat_hp.get("inner_eps", 0.0),
        gamma_min=gamma_min, eps=eps,
    )
    # test_eps clips the inference domain away from the tau->0 corner; test_path
    # is built explicitly with the test_* schedule keys so the estimator default
    # is never relied upon.
    test_path = direct_1d(
        sched=_sched_1d(_test_sched_hp(flat_hp)),
        inner_eps=flat_hp.get("test_inner_eps", 0.0),
        gamma_min=flat_hp.get("test_gamma_min", gamma_min),
        eps=flat_hp.get("test_eps", 0.05),
    )
    time = time_sampler_from_legacy_cfg(
        flat_hp.get("time_dist", "uniform"), eps=path.eps,
        apply_iw=flat_hp.get("apply_iw", True),
    )
    return VFMOrthros(
        input_dim=input_dim,
        path=path,
        test_path=test_path,
        time=time,
        n_epochs=flat_hp["n_epochs"],
        batch_size=flat_hp["batch_size"],
        optim=_optim_from_hp(flat_hp),
        sched=_sched_from_hp(flat_hp),
        ema=_ema_from_hp(flat_hp),
        integration_steps=flat_hp["integration_steps"],
        hidden_dim=flat_hp.get("hidden_dim", 256),
        n_hidden_layers=flat_hp.get("n_hidden_layers", 3),
        n_shared_layers=flat_hp.get("n_shared_layers", 2),
        activation=flat_hp.get("activation", "gelu"),
        layernorm=flat_hp.get("layernorm", "off"),
        antithetic=flat_hp.get("antithetic", False),
        reweight=flat_hp.get("reweight", False),
        precond=flat_hp.get("precond", False),
        div_method=flat_hp.get("div_method", "hutchinson"),
        div_noise=flat_hp.get("div_noise", "rademacher"),
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
    # max_train_samples is a TriangularMDRE data cap (not an HPO knob).
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
# keyed by builder function name -- this is what suggest_hp METADATA["builder"]
# stores and what worker.py looks up. building from fn.__name__ keeps the key
# and the function identity in lockstep (no manual key string to drift).
BUILDERS_REGISTRY = {
    fn.__name__: fn
    for fn in (
        build_TSM, build_CTSM, build_VFM, build_VFMOrthros, build_BDRE,
        build_MDRE, build_MHTDRE, build_FMDRE, build_FMDRE_S2,
        build_TriangularFMDRE, build_TriangularTSM, build_MHTTDRE,
        build_TriangularMDRE, build_TriangularCTSM_V1, build_TriangularCTSM_V2,
        build_TriangularCTSM_V3, build_TriangularVFM_V1, build_TriangularVFM_V2,
        build_TriangularVFM_V3, build_TabularPluginDRE,
        build_SmoothedTabularPluginDRE,
    )
}
