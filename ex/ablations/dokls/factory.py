"""dokls local builder registry and entry point.

builds DRE legs for the dokls ablation without modifying shared infra.
exports LEG_BUILDERS (method -> leg factory) and build() (method + route -> ELDR).

leg interface (what TwoLegELDR/DirectELDR drive, see two_leg.py):
    init_fit(samples_num, samples_den) -> None   # log(num/den) direction
    train_step() -> float                         # one gradient step
    finalize() -> None
    predict_ldr(xs) -> [N]

cls legs are the step-wise critics from critics.py / methods.py (real train_step).
reg legs wrap a src reg method and drive one reg step via _make_train_step from
_trainer.py, replicating the method's fit() prologue (model/optim/sched/ema/loss).
the per-leg step budget (n_steps//2 for two_leg, n_steps for direct) is threaded
by build() into each leg so the scheduler/ema anneal over the right horizon.
"""

from typing import Callable

import torch

from src.methods.common.base import DRE
from src.methods.reg.tsm import TSM
from src.methods.reg.ctsm import CTSM
from src.methods.reg.vfm import make_vfm, VFM
from src.methods.reg.fmdre import FMDRE
from src.methods.reg.common._cfgs import OptimCfg, SchedCfg, EmaCfg, TimeCfg
from src.methods.reg.common._cfgs import (
    make_optim, make_sched, make_ema, make_time_sampler
)
from src.methods.reg.common._trainer import (
    _make_train_step, _make_build_batch, _make_clip, _noop
)
from src.methods.reg.common._losses import tsm_loss, make_fm_loss
from src.methods.reg.common._paradigm_funcs import (
    ctsm_regression_target_direct_1d, vfm_velocity_target_direct_1d,
)
from src.methods.reg.common._weighting import resolve_outer_lambda
from src.methods.reg.common._precond import (
    endpoint_moments, make_coeffs, make_lambda, wrap, wrap_fm,
)
from src.methods.reg.common._time_samplers import time_sampler_from_legacy_cfg
from src.waypoints.path_builders import direct_1d, stiff_noise, bridge_noise
from src.waypoints.waypoints1d import DefaultWaypointBuilder1D

from ex.ablations.dokls.critics import StepBinaryCritic, StepMultiHeadCritic
from ex.ablations.dokls.losses import BCESpec
from ex.ablations.dokls.methods import BDRE_NWJ, BDRE_DV, MHT_NWJ, MHT_DV
from ex.ablations.dokls.two_leg import TwoLegELDR, DirectELDR


# ---------------------------------------------------------------------------
# cfg helpers (copied from builders.py to avoid circular imports)
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
    """build SchedCfg from flat hp keys (cosine_min_factor)."""
    return SchedCfg(cosine_min_factor=flat_hp.get("cosine_min_factor", 1.0))


def _time_from_hp(flat_hp: dict, *, eps: float) -> TimeCfg:
    """build TimeCfg from flat hp keys (time_dist, apply_iw) at the given eps."""
    return TimeCfg.from_dist(
        flat_hp.get("time_dist", "uniform"),
        eps=eps,
        apply_iw=flat_hp.get("apply_iw", True),
    )


def _sched_1d(flat_hp: dict, *, default_k: float = 20.0):
    """resolve a 1d noise schedule (Sched1D) from flat hp."""
    kind = flat_hp.get("sched", "stiff")
    sigma = flat_hp.get("sigma", 1.0)
    if kind == "stiff":
        return stiff_noise(k=flat_hp.get("k", default_k), sigma=sigma)
    if kind == "bridge":
        return bridge_noise(sigma=sigma)
    raise ValueError(f"sched must be 'stiff' or 'bridge'; got {kind!r}")


def _test_sched_hp(flat_hp: dict) -> dict:
    """view of flat_hp with test_* schedule keys aliased onto sched/sigma/k."""
    return {
        "sched": flat_hp.get("test_sched", flat_hp.get("sched", "stiff")),
        "sigma": flat_hp.get("test_sigma", flat_hp.get("sigma", 1.0)),
        "k": flat_hp.get("test_k", flat_hp.get("k", 20.0)),
    }


# ---------------------------------------------------------------------------
# reg-leg stepping driver
# ---------------------------------------------------------------------------


def _reg_step(
    *,
    model,
    model_module,
    optim,
    sched,
    ema,
    grad_clip_norm,
    time_sampler,
    loss_fn,
    loss_kw,
    x0,
    x1,
    batch_size,
    device,
) -> Callable[[], float]:
    """bind one reg gradient step over (x0=num, x1=den) into a 0-arg closure.

    mirrors train_loop's hot-path wiring: bootstrap batch, sample (tau, iw),
    call loss_fn, step optim, then sched/ema. `model` is the forward callable
    (possibly a precond wrapper); `model_module` is the parameter-carrying net
    used for clipping / ema. returns the closure produced by _make_train_step.
    """
    build_batch = _make_build_batch(x0, x1, None, batch_size, device, needs_xstar=False)
    do_clip = _make_clip(model_module.parameters(), grad_clip_norm)
    do_sched = sched.step if sched is not None else _noop
    do_ema = (lambda: ema.update(model_module)) if ema is not None else _noop
    return _make_train_step(
        model, build_batch, time_sampler, loss_fn, loss_kw, optim,
        do_clip, do_sched, do_ema, batch_size, device,
    )


def _setup_tsm(m: TSM, x0, x1):
    """build the TSM single-net step; loss is the module-level tsm_loss."""
    m.init_model()
    model = m.model
    optim = make_optim(model.parameters(), m.optim)
    sched = make_sched(optim, m.n_steps, m.optim.lr, m.sched)
    ema = make_ema(model, m.ema)  # note: TSM.predict_ldr ignores ema (raw net)
    step = _reg_step(
        model=model, model_module=model, optim=optim, sched=sched, ema=ema,
        grad_clip_norm=m.optim.grad_clip_norm,
        time_sampler=make_time_sampler(m.time),
        loss_fn=tsm_loss, loss_kw={"reweight": m.reweight, "eps": m.time.eps},
        x0=x0, x1=x1, batch_size=m.batch_size, device=m.device,
    )
    return [step], [model]


def _setup_ctsm(m: CTSM, x0, x1):
    """build the CTSM single-net step; reconstructs CTSM.fit's inline SB loss.

    stores the ema on m.ema_obj because CTSM.predict_ldr applies it.
    """
    m.init_model()
    model = m.model
    optim = make_optim(model.parameters(), m.optim)
    sched = make_sched(optim, m.n_steps, m.optim.lr, m.sched)
    m.ema_obj = make_ema(model, m.ema)
    path_arg = m.path
    reweight_arg = m.reweight

    def loss_fn(net, batch, tau, iw):
        x0b, x1b = batch["x0"], batch["x1"]
        epsilon = torch.randn_like(x0b)
        x_tau, target, lambda_t = ctsm_regression_target_direct_1d(
            path_arg, x0b, x1b, tau, epsilon,
        )
        pred = net(x_tau, tau)
        loss_per_sample = ((target - lambda_t * pred) ** 2).mean(dim=-1)
        if reweight_arg:
            outer = (tau.squeeze(-1) * (1 - tau.squeeze(-1))).clamp(min=1e-8)
        else:
            outer = 1.0
        return (loss_per_sample * outer * iw.squeeze(-1)).mean()

    step = _reg_step(
        model=model, model_module=model, optim=optim, sched=sched, ema=m.ema_obj,
        grad_clip_norm=m.optim.grad_clip_norm, time_sampler=m.time,
        loss_fn=loss_fn, loss_kw={},
        x0=x0, x1=x1, batch_size=m.batch_size, device=m.device,
    )
    return [step], [model]


def _setup_fmdre(m: FMDRE, x0, x1):
    """build the FMDRE single-net step via make_fm_loss (+ optional precond)."""
    m.init_model()
    model = m.model
    if m.precond:
        m._moments = endpoint_moments({"x_data": torch.cat([x0, x1], dim=0)})
        coeff_v = make_coeffs("fm", m._moments, "velocity")
        coeff_s = make_coeffs("fm", m._moments, "score")
        outer_weight = make_lambda(coeff_v)
        model_to_train = wrap_fm(model, coeff_v, coeff_s)
    else:
        outer_weight = None
        model_to_train = model
    optim = make_optim(model.parameters(), m.optim)
    sched = make_sched(optim, m.n_steps, m.optim.lr, m.sched)
    ema = make_ema(model, m.ema)  # note: FMDRE.predict_ldr runs the raw net
    loss_fn = make_fm_loss(
        score_weight=m.score_weight, p_uncond=0.0, sentinel_cond=-1.0,
        reweight=m.reweight, outer_weight=outer_weight,
    )
    step = _reg_step(
        model=model_to_train, model_module=model, optim=optim, sched=sched, ema=ema,
        grad_clip_norm=m.optim.grad_clip_norm,
        time_sampler=make_time_sampler(m.time),
        loss_fn=loss_fn, loss_kw={},
        x0=x0, x1=x1, batch_size=m.batch_size, device=m.device,
    )
    return [step], [model]


def _setup_vfm(m: VFM, x0, x1):
    """build the VFM two-net (b, eta) steps; mirrors VFM.fit's interleaved wiring.

    train_step runs one b-update then one eta-update per call, matching
    train_interleaved. ema_b/ema_eta are stored on m for predict_ldr.
    """
    m.init_model()
    net_b, net_eta = m.net_b, m.net_eta
    path = m.path
    reweight = m.reweight
    antithetic = m.antithetic

    if m.precond:
        m._moments = endpoint_moments({"x0": x0, "x1": x1})
        coeff_b = make_coeffs(path, m._moments, "velocity")
        coeff_eta = make_coeffs(path, m._moments, "noise")
        lambda_b = make_lambda(coeff_b)
        lambda_eta = make_lambda(coeff_eta)
        net_b_callable = wrap(net_b, coeff_b)
        net_eta_callable = wrap(net_eta, coeff_eta)
    else:
        net_b_callable = net_b
        net_eta_callable = net_eta

    def w_b(tau):
        return lambda_b(tau) if m.precond else resolve_outer_lambda(reweight, tau)

    def w_eta(tau):
        return lambda_eta(tau) if m.precond else resolve_outer_lambda(reweight, tau)

    if antithetic:
        def loss_b(model, batch, tau, iw):
            x0b, x1b = batch["x0"], batch["x1"]
            z = torch.randn_like(x0b)
            x_t_p, v_p = vfm_velocity_target_direct_1d(path, x0b, x1b, tau, z)
            b_p = model(x_t_p, tau)
            l_p = 0.5 * (b_p ** 2).sum(-1) - (v_p * b_p).sum(-1)
            x_t_m, v_m = vfm_velocity_target_direct_1d(path, x0b, x1b, tau, -z)
            b_m = model(x_t_m, tau)
            l_m = 0.5 * (b_m ** 2).sum(-1) - (v_m * b_m).sum(-1)
            return (0.5 * (l_p + l_m) * w_b(tau) * iw.squeeze(-1)).mean()
    else:
        def loss_b(model, batch, tau, iw):
            x0b, x1b = batch["x0"], batch["x1"]
            z = torch.randn_like(x0b)
            x_t, v_star = vfm_velocity_target_direct_1d(path, x0b, x1b, tau, z)
            b = model(x_t, tau)
            return ((0.5 * (b ** 2).sum(-1) - (v_star * b).sum(-1)) * w_b(tau) * iw.squeeze(-1)).mean()

    def loss_eta(model, batch, tau, iw):
        x0b, x1b = batch["x0"], batch["x1"]
        z = torch.randn_like(x0b)
        x_t, _ = vfm_velocity_target_direct_1d(path, x0b, x1b, tau, z)
        eta = model(x_t, tau)
        return ((0.5 * (eta ** 2).sum(-1) - (z * eta).sum(-1)) * w_eta(tau) * iw.squeeze(-1)).mean()

    optim_b = make_optim(net_b.parameters(), m.optim)
    optim_eta = make_optim(net_eta.parameters(), m.optim)
    sched_b = make_sched(optim_b, m.n_steps, m.optim.lr, m.sched)
    sched_eta = make_sched(optim_eta, m.n_steps, m.optim.lr, m.sched)
    m.ema_b = make_ema(net_b, m.ema)
    m.ema_eta = make_ema(net_eta, m.ema)

    step_b = _reg_step(
        model=net_b_callable, model_module=net_b, optim=optim_b, sched=sched_b,
        ema=m.ema_b, grad_clip_norm=m.optim.grad_clip_norm, time_sampler=m.time,
        loss_fn=loss_b, loss_kw={},
        x0=x0, x1=x1, batch_size=m.batch_size, device=m.device,
    )
    step_eta = _reg_step(
        model=net_eta_callable, model_module=net_eta, optim=optim_eta, sched=sched_eta,
        ema=m.ema_eta, grad_clip_norm=m.optim.grad_clip_norm, time_sampler=m.time,
        loss_fn=loss_eta, loss_kw={},
        x0=x0, x1=x1, batch_size=m.batch_size, device=m.device,
    )
    return [step_b, step_eta], [net_b, net_eta]


_REG_SETUP = {TSM: _setup_tsm, CTSM: _setup_ctsm, VFM: _setup_vfm, FMDRE: _setup_fmdre}


class _RegMethodStep:
    """expose a reg method (TSM/CTSM/VFM/FMDRE) through the stepping leg interface.

    init_fit replicates the method's fit() prologue (build model, optim, sched,
    ema, loss) and binds one step closure per network. train_step advances every
    network once (b then eta for VFM). predict_ldr delegates to the method, which
    reads the trained modules + any ema stored on it. finalize sets eval mode,
    matching the base fit()'s exit envelope. maps num -> x0 (p0), den -> x1 (p1)
    so predict_ldr(fit(num, den)) = log(num/den), matching the DRE sign contract.

    a plain class (not a DRE): the leg interface is duck-typed and fit() is never
    called on a leg (two_leg drives init_fit/train_step/finalize/predict_ldr).
    """

    def __init__(self, method: DRE, device) -> None:
        """wrap a constructed reg method; device is the method's compute device."""
        self.input_dim = method.input_dim
        self.method = method
        self.device = device
        self._steps = None
        self._modules = None

    def init_fit(self, samples_num: torch.Tensor, samples_den: torch.Tensor) -> None:
        """build model + step closures over (num as x0, den as x1)."""
        m = self.method
        x0 = samples_num.float().to(m.device)
        x1 = samples_den.float().to(m.device)
        setup = _REG_SETUP[type(m)]
        self._steps, self._modules = setup(m, x0, x1)

    def train_step(self) -> float:
        """advance each network once in train mode; return the last step's loss."""
        for mod in self._modules:
            mod.train()
        loss = 0.0
        for step in self._steps:
            loss = step()
        return loss

    def finalize(self) -> None:
        """set all networks to eval mode (base fit()'s final state)."""
        for mod in self._modules:
            mod.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """delegate to the reg method (respects its ema handling)."""
        return self.method.predict_ldr(xs)


class _MHTStep:
    """base MultiHeadTDRE leg: StepMultiHeadCritic with per-head BCE over waypoints.

    init_fit builds a chain of `num_waypoints` bridge samples between num and den
    (DefaultWaypointBuilder1D), then trains one BCE head per adjacent pair. the
    per-head log-ratios telescope: sum_i log(w_i / w_{i+1}) = log(num / den).
    """

    def __init__(
        self, *, input_dim, device, num_waypoints, latent_dim, n_hidden_layers,
        lr, weight_decay, n_steps, batch_size,
    ) -> None:
        """store leg hyperparameters; defer critic build to init_fit."""
        self.input_dim = input_dim
        self.device = device
        self.num_waypoints = num_waypoints
        self.latent_dim = latent_dim
        self.n_hidden_layers = n_hidden_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.waypoint_builder = DefaultWaypointBuilder1D()
        self.critic = None

    def init_fit(self, samples_num: torch.Tensor, samples_den: torch.Tensor) -> None:
        """build waypoints, then a per-head numerator/denominator critic."""
        waypoints = self.waypoint_builder.build_waypoints(
            samples_num, samples_den, self.num_waypoints
        )  # [num_waypoints, batch, input_dim]
        num_heads = self.num_waypoints - 1
        num_list = [waypoints[i] for i in range(num_heads)]
        den_list = [waypoints[i + 1] for i in range(num_heads)]
        self.critic = StepMultiHeadCritic(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            n_hidden_layers=self.n_hidden_layers,
            num_heads=num_heads,
            loss_specs=[BCESpec() for _ in range(num_heads)],
            lr=self.lr,
            weight_decay=self.weight_decay,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            device=self.device,
        )
        self.critic.init_fit(num_list, den_list)

    def train_step(self) -> float:
        """one optimizer step; return summed per-head loss."""
        return self.critic.train_step()

    def finalize(self) -> None:
        """per-head finalization (BCE is a no-op)."""
        self.critic.finalize()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """telescoped log-density ratio over heads."""
        return self.critic.predict_ldr(xs)


# ---------------------------------------------------------------------------
# builder functions (one per method in LEG_BUILDERS)
# ---------------------------------------------------------------------------


def _build_bdre(input_dim: int, device: str | torch.device, **hp) -> object:
    """build the BDRE leg as a step-wise binary critic under BCE loss."""
    device = str(device)
    return StepBinaryCritic(
        input_dim=input_dim,
        latent_dim=hp.get("latent_dim", 128),
        n_hidden_layers=hp.get("n_hidden_layers", 2),
        loss_spec=BCESpec(),
        lr=hp.get("lr", hp.get("learning_rate", 1e-3)),
        weight_decay=hp.get("weight_decay", 0.0),
        n_steps=hp.get("n_steps", 6400),
        batch_size=hp.get("batch_size", 256),
        device=device,
    )


def _build_bdre_nwj(input_dim: int, device: str | torch.device, **hp) -> DRE:
    """build BDRE_NWJ leg (local NWJ critic)."""
    device = str(device)
    return BDRE_NWJ(input_dim=input_dim, device=device, **hp)


def _build_bdre_dv(input_dim: int, device: str | torch.device, **hp) -> DRE:
    """build BDRE_DV leg (local DV critic)."""
    device = str(device)
    return BDRE_DV(input_dim=input_dim, device=device, **hp)


def _build_mhtdre(input_dim: int, device: str | torch.device, **hp) -> object:
    """build the MultiHeadTDRE leg as a step-wise multi-head BCE critic."""
    device = str(device)
    return _MHTStep(
        input_dim=input_dim,
        device=device,
        num_waypoints=hp.get("num_waypoints", 10),
        latent_dim=hp.get("latent_dim", 128),
        n_hidden_layers=hp.get("n_hidden_layers", 2),
        lr=hp.get("lr", hp.get("learning_rate", 1e-3)),
        weight_decay=hp.get("weight_decay", 0.0),
        n_steps=hp.get("n_steps", 6400),
        batch_size=hp.get("batch_size", 256),
    )


def _build_mht_nwj(input_dim: int, device: str | torch.device, **hp) -> DRE:
    """build MHT_NWJ leg (local multi-head NWJ critic)."""
    device = str(device)
    return MHT_NWJ(input_dim=input_dim, device=device, **hp)


def _build_mht_dv(input_dim: int, device: str | torch.device, **hp) -> DRE:
    """build MHT_DV leg (local multi-head DV critic)."""
    device = str(device)
    return MHT_DV(input_dim=input_dim, device=device, **hp)


def _build_tsm_leg(input_dim: int, device: str | torch.device, **hp) -> DRE:
    """build TSM leg via stepping wrapper."""
    device = str(device)
    eps = hp.get("eps", 1e-3)
    tsm = TSM(
        input_dim=input_dim,
        device=device,
        hidden_dim=hp.get("hidden_dim", 256),
        n_hidden_layers=hp.get("n_hidden_layers", 3),
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        optim=_optim_from_hp(hp),
        sched=_sched_from_hp(hp),
        ema=_ema_from_hp(hp),
        time=_time_from_hp(hp, eps=eps),
        reweight=hp.get("reweight", False),
        activation=hp.get("activation", "silu"),
        integration_steps=hp.get("integration_steps", 200),
        early_stop_cfg=hp.pop("early_stop_cfg", None),
    )
    return _RegMethodStep(tsm, device)


def _build_ctsm_leg(input_dim: int, device: str | torch.device, **hp) -> DRE:
    """build CTSM leg via stepping wrapper."""
    device = str(device)
    eps = hp.get("eps", 1e-3)
    path = direct_1d(
        sched=_sched_1d(hp),
        inner_eps=hp.get("inner_eps", 0.0),
        gamma_min=hp.get("gamma_min", 0.0),
        eps=eps,
    )
    test_path = direct_1d(
        sched=_sched_1d(_test_sched_hp(hp)),
        inner_eps=hp.get("test_inner_eps", 0.0),
        gamma_min=hp.get("test_gamma_min", 0.0),
        eps=hp.get("test_eps", eps),
    )
    time = time_sampler_from_legacy_cfg(
        hp.get("time_dist", "uniform"),
        eps=path.eps,
        apply_iw=hp.get("apply_iw", True),
    )
    ctsm = CTSM(
        input_dim=input_dim,
        device=device,
        path=path,
        test_path=test_path,
        time=time,
        sigma=hp.get("sigma", 1.0),
        hidden_dim=hp.get("hidden_dim", 256),
        n_hidden_layers=hp.get("n_hidden_layers", 3),
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        optim=_optim_from_hp(hp),
        sched=_sched_from_hp(hp),
        ema=_ema_from_hp(hp),
        integration_steps=hp.get("integration_steps", 1000),
        activation=hp.get("activation", "elu"),
        reweight=hp.get("reweight", False),
        early_stop_cfg=hp.pop("early_stop_cfg", None),
    )
    return _RegMethodStep(ctsm, device)


def _build_vfm_leg(input_dim: int, device: str | torch.device, **hp) -> DRE:
    """build VFM leg via stepping wrapper."""
    device = str(device)
    eps = hp.get("eps", 1e-3)
    path = direct_1d(
        sched=_sched_1d(hp),
        inner_eps=hp.get("inner_eps", 0.0),
        gamma_min=hp.get("gamma_min", 0.0),
        eps=eps,
    )
    test_path = direct_1d(
        sched=_sched_1d(_test_sched_hp(hp)),
        inner_eps=hp.get("test_inner_eps", 0.0),
        gamma_min=hp.get("test_gamma_min", 0.0),
        eps=hp.get("test_eps", eps),
    )
    time = time_sampler_from_legacy_cfg(
        hp.get("time_dist", "uniform"),
        eps=path.eps,
        apply_iw=hp.get("apply_iw", True),
    )
    vfm = make_vfm(
        input_dim=input_dim,
        device=device,
        path=path,
        test_path=test_path,
        time=time,
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        optim=_optim_from_hp(hp),
        sched=_sched_from_hp(hp),
        ema=_ema_from_hp(hp),
        integration_steps=hp.get("integration_steps", 3000),
        hidden_dim=hp.get("hidden_dim", 256),
        n_hidden_layers=hp.get("n_hidden_layers", 3),
        activation=hp.get("activation", "gelu"),
        layernorm=hp.get("layernorm", "off"),
        antithetic=hp.get("antithetic", True),
        reweight=hp.get("reweight", False),
        precond=hp.get("precond", False),
        div_method=hp.get("div_method", "hutchinson"),
        div_noise=hp.get("div_noise", "rademacher"),
        n_hutch_samples=hp.get("n_hutch_samples", 1),
        early_stop_cfg=hp.pop("early_stop_cfg", None),
    )
    return _RegMethodStep(vfm, device)


def _build_fmdre_leg(input_dim: int, device: str | torch.device, **hp) -> DRE:
    """build FMDRE leg via stepping wrapper."""
    device = str(device)
    eps = hp.get("eps", 1e-3)
    n_hidden_layers = hp.get("n_hidden_layers", 3)
    fmdre = FMDRE(
        input_dim=input_dim,
        device=device,
        hidden_dim=hp.get("hidden_dim", 256),
        n_hidden_layers=n_hidden_layers,
        n_shared_layers=hp.get("n_shared_layers", n_hidden_layers),
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        optim=_optim_from_hp(hp),
        sched=_sched_from_hp(hp),
        ema=_ema_from_hp(hp),
        time=_time_from_hp(hp, eps=eps),
        score_weight=hp.get("score_weight", 1.0),
        div_method="exact",
        n_hutch_samples=4,
        integration_steps=hp.get("integration_steps", 10000),
        reweight=hp.get("reweight", False),
        precond=hp.get("precond", False),
        infer_eps=hp.get("infer_eps") or min(eps, 2e-3),
        early_stop_cfg=hp.pop("early_stop_cfg", None),
    )
    return _RegMethodStep(fmdre, device)


# ---------------------------------------------------------------------------
# registry and entry point
# ---------------------------------------------------------------------------


LEG_BUILDERS: dict[str, Callable] = {
    "BDRE": _build_bdre,
    "BDRE_NWJ": _build_bdre_nwj,
    "BDRE_DV": _build_bdre_dv,
    "MultiHeadTDRE": _build_mhtdre,
    "MHT_NWJ": _build_mht_nwj,
    "MHT_DV": _build_mht_dv,
    "TSM": _build_tsm_leg,
    "CTSM": _build_ctsm_leg,
    "VFM": _build_vfm_leg,
    "FMDRE": _build_fmdre_leg,
}


def build(method: str, route: str, input_dim: int, device: str | torch.device, **hp):
    """factory entry point for DoklsAdapter.

    Args:
        method: base method name (key in LEG_BUILDERS).
        route: "two_leg" or "direct".
        input_dim: feature dimension.
        device: torch device.
        **hp: flat hyperparameter dict (from HPO).

    Returns:
        TwoLegELDR or DirectELDR instance.

    Raises:
        KeyError: if method not in LEG_BUILDERS.
        ValueError: if route not in {"two_leg", "direct"}.

    two_leg drives each leg for n_steps//2 steps, direct drives one leg for
    n_steps. two_leg.py does not forward a per-leg n_steps to the builder, so
    build() wraps the registry builder in a closure that injects the correct
    per-leg budget; the reg legs consume it for their scheduler/ema horizon.
    """
    if method not in LEG_BUILDERS:
        raise KeyError(
            f"unknown method {method!r}; available: {sorted(LEG_BUILDERS.keys())}"
        )

    if route not in {"two_leg", "direct"}:
        raise ValueError(
            f"route must be 'two_leg' or 'direct'; got {route!r}"
        )

    base_builder = LEG_BUILDERS[method]

    # pop n_steps BEFORE **hp splat so it is not passed twice to the ELDR; the
    # per-leg budget is injected into each leg via the closure below.
    n_steps = hp.pop("n_steps", 6400)
    n_leg_steps = n_steps // 2 if route == "two_leg" else n_steps

    def leg_builder(leg_input_dim, leg_device, **leg_hp):
        """registry builder with the route-correct per-leg n_steps injected."""
        return base_builder(leg_input_dim, leg_device, n_steps=n_leg_steps, **leg_hp)

    if route == "two_leg":
        return TwoLegELDR(
            leg_builder=leg_builder,
            input_dim=input_dim,
            device=device,
            n_steps=n_steps,
            **hp,
        )
    else:  # route == "direct"
        return DirectELDR(
            leg_builder=leg_builder,
            input_dim=input_dim,
            device=device,
            n_steps=n_steps,
            **hp,
        )
