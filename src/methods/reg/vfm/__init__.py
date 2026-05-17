"""VFM family: velocity flow matching DRE.

The two-source (DRE) variants VFM and VFMOrthros live in this module;
triangular variants (V1 barycentric, V2 piecewise-SB, V3 2D) live under `.tri`.
"""
import warnings
from typing import Optional, Literal, Callable

import torch

from ..common._cfgs import (
    OptimCfg, SchedCfg, EmaCfg,
    make_optim, make_sched, make_ema,
)
from ..common._trainer import train_two_phase, train_loop
from ..common._weighting import resolve_outer_lambda
from ..common._estimator_helpers import _validate_and_store_slots
from ..common._paradigm_funcs import (
    vfm_velocity_target_direct_1d,
    vfm_time_score_1d,
    vfm_orthros_time_score_1d,
)
from ..common._time_samplers import make_uniform, TimeSampler1D
from ..common._curves import IdentityCurve1D, Curve
from ..common._integrators import integrator_trapezoid, Integrator
from ..common._predict_ldr import predict_ldr_via_curve
from src.waypoints.dataclass_paths import DirectPath1D
from src.waypoints.path_builders import direct_vfm
from ...common.base import DRE
from src.models.common.mlp import MLP
from src.models.flow.orthros_net import OrthrosNet
from src.models.flow.div_estimators import build_div_fn


class VFM(DRE):
    """VFM stock: velocity flow matching on direct (two-source) path, two-phase training.

    trains b (velocity) and eta (denoiser) sequentially; integrates the time-score
    via predict_ldr_via_curve with IdentityCurve1D to predict log(p0/p1).

    uses DirectPath1D (no xstar), TimeSampler1D, IdentityCurve1D, and Integrator
    (Pillar E: four-slot surface). legacy scalar k and n_t parameters preserved
    for backward compatibility; k parameterizes the path if path=None.
    """
    def __init__(
        self,
        input_dim: int,
        *,
        path: Optional[DirectPath1D] = None,
        time: Optional[TimeSampler1D] = None,
        curve: Curve = None,
        integrator: Integrator = None,
        k: float = 0.5,
        inner_eps: float = 0.0,
        gamma_min: float = 0.0,
        test_inner_eps: float = 0.0,
        test_gamma_min: float = 0.0,
        test_path: Optional["DirectPath1D"] = None,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_epochs: int = 1000,
        batch_size: int = 512,
        optim: Optional[OptimCfg] = None,
        sched: SchedCfg = None,
        ema: EmaCfg = None,
        device: Optional[str] = None,
        antithetic: bool = False,
        div_method: Literal['hutchinson', 'exact'] = 'hutchinson',
        div_noise: Literal['rademacher', 'gaussian'] = 'rademacher',
        n_hutch_samples: int = 1,
        integration_steps: int = 10000,
        activation: str = "silu",
        layernorm: str = "off",
        reweight: bool = False,
        precond: bool = False,
        n_t: Optional[int] = None,
    ) -> None:
        super().__init__(input_dim)

        # resolve all four slots before validation
        if path is None:
            path = direct_vfm(k=k, inner_eps=inner_eps, gamma_min=gamma_min, eps=1e-3)

        if time is None:
            time = make_uniform(eps=path.eps)

        if test_path is None:
            test_path = direct_vfm(k=k, inner_eps=test_inner_eps, gamma_min=test_gamma_min, eps=1e-3)

        if curve is None:
            curve = IdentityCurve1D()

        if integrator is None:
            integrator = integrator_trapezoid

        # F2/F3 sampler/path inner_eps consistency
        samp_ie = getattr(time, "inner_eps", 0.0) if time is not None else 0.0
        if (samp_ie > 0) != (inner_eps > 0):
            warnings.warn(
                f"asymmetric inner_eps: sampler={samp_ie}, path={inner_eps}. "
                "Probably unintentional.", UserWarning, stacklevel=2,
            )
        elif samp_ie > 0 and inner_eps > 0:
            assert abs(samp_ie - inner_eps) < 1e-9, \
                f"sampler/path inner_eps mismatch: {samp_ie} vs {inner_eps}"

        # validate and store four-slot surface
        _validate_and_store_slots(
            self,
            input_dim=input_dim,
            path=path,
            time=time,
            curve=curve,
            integrator=integrator,
            expected_path_type=DirectPath1D,
            expected_curve_dim=1,
            device=device,
        )

        # store test path and clamping scalars
        self.test_path = test_path
        self.inner_eps = inner_eps
        self.gamma_min = gamma_min
        self.test_inner_eps = test_inner_eps
        self.test_gamma_min = test_gamma_min

        # store network and training hyperparameters
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optim = optim if optim is not None else OptimCfg(lr=1e-3)
        self.sched = sched if sched is not None else SchedCfg()
        self.ema = ema if ema is not None else EmaCfg()
        self.k = k

        # deprecation warning for n_t
        if n_t is not None:
            warnings.warn(
                "`n_t` is deprecated and unused; it has no effect on VFM training "
                "or inference and will be removed in a future revision.",
                DeprecationWarning,
                stacklevel=2,
            )

        # divergence-related parameters
        if div_method not in ('hutchinson', 'exact'):
            raise ValueError(f"div_method must be 'hutchinson' or 'exact'; got {div_method!r}")
        if div_noise not in ('rademacher', 'gaussian'):
            raise ValueError(f"div_noise must be 'rademacher' or 'gaussian'; got {div_noise!r}")
        if n_hutch_samples < 1:
            raise ValueError(f"n_hutch_samples must be >= 1; got {n_hutch_samples}")

        self.div_method = div_method
        self.div_noise = div_noise
        self.n_hutch_samples = n_hutch_samples
        self._div_fn = build_div_fn(div_method, noise=div_noise, n_samples=n_hutch_samples)

        # miscellaneous parameters
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off', 'pre', 'post'}}; got {layernorm!r}")

        self.activation = activation
        self.layernorm = layernorm
        self.antithetic = antithetic
        self.integration_steps = integration_steps
        self.reweight = reweight
        self.precond = precond
        self._moments = None  # placeholder; filled by fit() if precond=True

        # network placeholders (frozen attribute names for checkpoint compat)
        self.net_b = None
        self.net_eta = None
        self.ema_b = None
        self.ema_eta = None

    def init_model(self) -> None:
        """instantiate net_b (velocity) and net_eta (denoiser) MLPs on device."""
        self.net_b = MLP(
            self.input_dim,
            self.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            layernorm=self.layernorm,
        ).to(self.device)
        self.net_eta = MLP(
            self.input_dim,
            self.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            layernorm=self.layernorm,
        ).to(self.device)


    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """train b then eta sequentially via train_two_phase.

        the loss_b closure binds antithetic, reweight, path geometry, and
        the time sampler so the inner loop sees only one specialized body.
        """
        self.init_model()
        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)

        # estimate endpoint moments once (used by both losses and predict_ldr)
        if self.precond:
            from ..common._precond import endpoint_moments
            self._moments = endpoint_moments(
                {"x0": samples_p0, "x1": samples_p1},
            )

        optim_b = make_optim(self.net_b.parameters(), self.optim)
        optim_eta = make_optim(self.net_eta.parameters(), self.optim)
        sched_b = make_sched(optim_b, self.n_epochs, self.optim.lr, self.sched)
        sched_eta = make_sched(optim_eta, self.n_epochs, self.optim.lr, self.sched)
        ema_b = make_ema(self.net_b, self.ema)
        ema_eta = make_ema(self.net_eta, self.ema)
        self.ema_b, self.ema_eta = ema_b, ema_eta

        path = self.path
        reweight = self.reweight
        antithetic = self.antithetic

        # resolve loss-weight function: lambda (precond) or reweight (standard)
        if self.precond:
            from ..common._precond import make_coeffs, make_lambda
            coeff_b = make_coeffs(path, self._moments, "velocity")
            coeff_eta = make_coeffs(path, self._moments, "noise")
            lambda_b = make_lambda(coeff_b)
            lambda_eta = make_lambda(coeff_eta)
            if reweight:
                warnings.warn(
                    "precond=True ignores reweight=True; EDM lambda replaces reweight. "
                    "Remove reweight or set precond=False.",
                    UserWarning,
                    stacklevel=2,
                )

        # resolve nets: wrap if precond, else use raw nets
        if self.precond:
            from ..common._precond import wrap
            net_b_callable = wrap(self.net_b, coeff_b)
            net_eta_callable = wrap(self.net_eta, coeff_eta)
        else:
            net_b_callable = self.net_b
            net_eta_callable = self.net_eta

        # bind weight function: lambda (precond) or reweight (standard)
        def get_weight_b(tau):
            return lambda_b(tau) if self.precond else resolve_outer_lambda(reweight, tau)
        def get_weight_eta(tau):
            return lambda_eta(tau) if self.precond else resolve_outer_lambda(reweight, tau)

        if antithetic:
            def loss_b_antithetic(model, batch, tau, iw):
                x0, x1 = batch["x0"], batch["x1"]
                z = torch.randn_like(x0)
                # positive noise
                x_t_p, v_star_p = vfm_velocity_target_direct_1d(path, x0, x1, tau, z)
                b_p = model(x_t_p, tau)
                l_p = 0.5 * (b_p ** 2).sum(-1) - (v_star_p * b_p).sum(-1)
                # negative noise
                x_t_m, v_star_m = vfm_velocity_target_direct_1d(path, x0, x1, tau, -z)
                b_m = model(x_t_m, tau)
                l_m = 0.5 * (b_m ** 2).sum(-1) - (v_star_m * b_m).sum(-1)
                outer = get_weight_b(tau)
                return (0.5 * (l_p + l_m) * outer * iw.squeeze(-1)).mean()
            loss_b = loss_b_antithetic
        else:
            def loss_b_naive(model, batch, tau, iw):
                x0, x1 = batch["x0"], batch["x1"]
                z = torch.randn_like(x0)
                x_t, v_star = vfm_velocity_target_direct_1d(path, x0, x1, tau, z)
                outer = get_weight_b(tau)
                b = model(x_t, tau)
                return ((0.5 * (b ** 2).sum(-1) - (v_star * b).sum(-1)) * outer * iw.squeeze(-1)).mean()
            loss_b = loss_b_naive

        loss_b.required_keys = frozenset({"x0", "x1"})
        loss_b.requires_tau_grad = False

        def loss_eta(model, batch, tau, iw):
            x0, x1 = batch["x0"], batch["x1"]
            z = torch.randn_like(x0)
            x_t, _ = vfm_velocity_target_direct_1d(path, x0, x1, tau, z)
            eta = model(x_t, tau)
            outer = get_weight_eta(tau)
            return ((0.5 * (eta ** 2).sum(-1) - (z * eta).sum(-1)) * outer * iw.squeeze(-1)).mean()

        loss_eta.required_keys = frozenset({"x0", "x1"})
        loss_eta.requires_tau_grad = False

        train_two_phase(
            model_b=net_b_callable,
            model_eta=net_eta_callable,
            model_module_b=self.net_b,
            model_module_eta=self.net_eta,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_b=loss_b,
            loss_eta=loss_eta,
            optim_b=optim_b,
            optim_eta=optim_eta,
            n_steps_b=self.n_epochs,
            n_steps_eta=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=self.time,
            scheduler_b=sched_b,
            scheduler_eta=sched_eta,
            ema_b=ema_b,
            ema_eta=ema_eta,
            grad_clip_norm_b=self.optim.grad_clip_norm,
            grad_clip_norm_eta=self.optim.grad_clip_norm,
            eps=self.path.eps,
        )

        self.net_b.eval()
        self.net_eta.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """integrate the time-score over tau in [eps, 1-eps] using predict_ldr_via_curve.

        applies EMA if available, uses unified curve + integrator surface.
        """
        if self.net_b is None or self.net_eta is None:
            raise RuntimeError("VFM model is not trained. Call fit() before predict_ldr().")

        self.net_b.eval()
        self.net_eta.eval()
        samples = xs.float().to(self.device)

        def time_score_fn(path, ts, xs):
            """loop per curve point; vfm_time_score_1d expects tau:[B,1] matching xs:[B,D]."""
            n = xs.shape[0]
            out = []
            for i in range(ts.shape[0]):
                tau_i = ts[i:i+1].expand(n, 1)
                out.append(vfm_time_score_1d(net_b_eval, net_eta_eval, path, xs, tau_i, self._div_fn))
            return torch.stack(out, dim=0)

        if self.ema_b is not None:
            self.ema_b.apply_to(self.net_b)
        if self.ema_eta is not None:
            self.ema_eta.apply_to(self.net_eta)

        # rebuild wrappers around EMA-applied nets (wrapper is parameter-free)
        if self.precond:
            from ..common._precond import make_coeffs, wrap
            coeff_b = make_coeffs(self.path, self._moments, "velocity")
            coeff_eta = make_coeffs(self.path, self._moments, "noise")
            net_b_eval = wrap(self.net_b, coeff_b)
            net_eta_eval = wrap(self.net_eta, coeff_eta)
        else:
            net_b_eval = self.net_b
            net_eta_eval = self.net_eta

        try:
            ldr = predict_ldr_via_curve(
                time_score_fn=time_score_fn,
                path=self.test_path,
                curve=self.curve,
                integrator=self.integrator,
                n_points=self.integration_steps,
                samples=samples,
            )
            return ldr
        finally:
            if self.ema_b is not None:
                self.ema_b.restore(self.net_b)
            if self.ema_eta is not None:
                self.ema_eta.restore(self.net_eta)


def make_vfm(input_dim: int, device: str = "cuda", **kwargs) -> VFM:
    """factory for VFM with sensible defaults; overrides passed via kwargs.

    when no explicit `optim` is given, `lr` is rerouted through OptimCfg;
    when `optim` is given it is forwarded as-is and `lr` is dropped (the optim
    object already carries it). every other kwarg must be a valid VFM
    constructor parameter -- unknown keys raise, never silently dropped.
    """
    defaults = {
        "k": 20,
        "n_epochs": 1000,
        "hidden_dim": 256,
        "n_hidden_layers": 3,
        "batch_size": 512,
        "integration_steps": 3000,
        "antithetic": True,
    }
    optim = kwargs.pop("optim", None)
    lr = kwargs.pop("lr", 1.3e-3)
    if optim is None:
        optim = OptimCfg(lr=lr)
    defaults.update(kwargs)
    return VFM(input_dim, device=device, optim=optim, **defaults)


class VFMOrthros(DRE):
    """VFMOrthros: single-network orthros-variant velocity flow matching.

    single-phase training on OrthrosNet (unified 2-head architecture). head 0
    predicts the endpoint posterior E[x0|x_t]; head 1 predicts the denoiser
    E[z|x_t]. at inference the x1 endpoint is derived from the interpolant
    constraint and the velocity is reconstructed; the orthros time-score is
    integrated via predict_ldr_via_curve with IdentityCurve1D to predict
    log(p0/p1).

    uses DirectPath1D (no xstar), TimeSampler1D, IdentityCurve1D, and Integrator
    (Pillar E: four-slot surface). inherits legacy k parameter from VFM for path
    construction; n_t is not supported (not preserved).

    parameterization note: predicting the denoiser directly (rather than
    reconstructing it from two endpoint heads) keeps eta_hat O(1), so the score
    -eta_hat/gamma carries only the unavoidable 1/gamma of any interpolant
    marginal score -- VFM-level stability, no 1/gamma^2 blow-up. gamma_min
    defaults to 0.1 (tunable) as a conservative noise floor; see
    vfm_orthros_time_score_1d.

    the one residual cost of the 2-head form is the derived x1 endpoint's 1/beta
    factor at the tau->0 corner. clip it by passing a `test_path` whose eps
    excludes that corner (inference integrates over [eps, 1-eps]); ~0.05 recovers
    VFM-parity MAE empirically, and `test_eps` in the HPO search space tunes it.
    the bare default test_path (eps=1e-3, symmetric with VFM) is *not* clipped;
    callers configure test_path. a 3-head variant predicting {x0, x1, eta} would
    remove the corner outright.
    """
    def __init__(
        self,
        input_dim: int,
        *,
        path: Optional[DirectPath1D] = None,
        time: Optional[TimeSampler1D] = None,
        curve: Curve = None,
        integrator: Integrator = None,
        k: float = 0.5,
        inner_eps: float = 0.0,
        gamma_min: float = 0.1,
        test_inner_eps: float = 0.0,
        test_gamma_min: float = 0.1,
        test_path: Optional["DirectPath1D"] = None,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_shared_layers: int = 2,
        n_epochs: int = 1000,
        batch_size: int = 512,
        optim: Optional[OptimCfg] = None,
        sched: SchedCfg = None,
        ema: EmaCfg = None,
        device: Optional[str] = None,
        antithetic: bool = False,
        div_method: Literal['hutchinson', 'exact'] = 'hutchinson',
        div_noise: Literal['rademacher', 'gaussian'] = 'rademacher',
        n_hutch_samples: int = 1,
        integration_steps: int = 10000,
        activation: str = "silu",
        layernorm: str = "off",
        reweight: bool = False,
        precond: bool = False,
    ) -> None:
        super().__init__(input_dim)

        # resolve all four slots before validation
        if path is None:
            path = direct_vfm(k=k, inner_eps=inner_eps, gamma_min=gamma_min, eps=1e-3)

        if time is None:
            time = make_uniform(eps=path.eps)

        if test_path is None:
            # bare default is symmetric with VFM (eps=1e-3); the corner-clip is
            # applied by configuring test_path explicitly.
            test_path = direct_vfm(k=k, inner_eps=test_inner_eps, gamma_min=test_gamma_min, eps=1e-3)

        if curve is None:
            curve = IdentityCurve1D()

        if integrator is None:
            integrator = integrator_trapezoid

        # F2/F3 sampler/path inner_eps consistency
        samp_ie = getattr(time, "inner_eps", 0.0) if time is not None else 0.0
        if (samp_ie > 0) != (inner_eps > 0):
            warnings.warn(
                f"asymmetric inner_eps: sampler={samp_ie}, path={inner_eps}. "
                "Probably unintentional.", UserWarning, stacklevel=2,
            )
        elif samp_ie > 0 and inner_eps > 0:
            assert abs(samp_ie - inner_eps) < 1e-9, \
                f"sampler/path inner_eps mismatch: {samp_ie} vs {inner_eps}"

        # validate and store four-slot surface
        _validate_and_store_slots(
            self,
            input_dim=input_dim,
            path=path,
            time=time,
            curve=curve,
            integrator=integrator,
            expected_path_type=DirectPath1D,
            expected_curve_dim=1,
            device=device,
        )

        # store test path and clamping scalars
        self.test_path = test_path
        self.inner_eps = inner_eps
        self.gamma_min = gamma_min
        self.test_inner_eps = test_inner_eps
        self.test_gamma_min = test_gamma_min

        # store network and training hyperparameters
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_shared_layers = n_shared_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optim = optim if optim is not None else OptimCfg(lr=1e-3)
        self.sched = sched if sched is not None else SchedCfg()
        self.ema = ema if ema is not None else EmaCfg()
        self.k = k

        # divergence-related parameters
        if div_method not in ('hutchinson', 'exact'):
            raise ValueError(f"div_method must be 'hutchinson' or 'exact'; got {div_method!r}")
        if div_noise not in ('rademacher', 'gaussian'):
            raise ValueError(f"div_noise must be 'rademacher' or 'gaussian'; got {div_noise!r}")
        if n_hutch_samples < 1:
            raise ValueError(f"n_hutch_samples must be >= 1; got {n_hutch_samples}")

        self.div_method = div_method
        self.div_noise = div_noise
        self.n_hutch_samples = n_hutch_samples
        self._div_fn = build_div_fn(div_method, noise=div_noise, n_samples=n_hutch_samples)

        # miscellaneous parameters
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off', 'pre', 'post'}}; got {layernorm!r}")

        self.activation = activation
        self.layernorm = layernorm
        self.antithetic = antithetic
        self.integration_steps = integration_steps
        self.reweight = reweight
        self.precond = precond
        self._moments = None  # placeholder; filled by fit() if precond=True

        # network placeholders (single network + ema runtime object)
        self.net = None
        self.ema_net = None

    def init_model(self) -> None:
        """instantiate self.net (OrthrosNet) on device."""
        self.net = OrthrosNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.n_hidden_layers,
            n_shared_layers=self.n_shared_layers,
            activation=self.activation,
            layernorm=self.layernorm,
        ).to(self.device)

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """train orthros net via single-phase train_loop.

        posterior-mean MSE loss. loss closure binds antithetic, reweight, path
        geometry, and time sampler so the inner loop sees one specialized body.
        """
        self.init_model()
        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)

        # estimate endpoint moments once (used by loss and predict_ldr)
        if self.precond:
            from ..common._precond import endpoint_moments
            self._moments = endpoint_moments(
                {"x0": samples_p0, "x1": samples_p1},
            )

        optim = make_optim(self.net.parameters(), self.optim)
        sched = make_sched(optim, self.n_epochs, self.optim.lr, self.sched)
        ema_net = make_ema(self.net, self.ema)
        self.ema_net = ema_net

        path = self.path
        reweight = self.reweight
        antithetic = self.antithetic

        # resolve loss-weight function and wrappers
        if self.precond:
            from ..common._precond import make_coeffs, make_lambda, wrap_2head
            coeff_x0 = make_coeffs(path, self._moments, "x0")
            coeff_eta = make_coeffs(path, self._moments, "noise")
            lambda_fn = make_lambda(coeff_eta)  # both heads share the denoiser lambda
            net_callable = wrap_2head(self.net, coeff_x0, coeff_eta)
            if reweight:
                warnings.warn(
                    "precond=True ignores reweight=True; EDM lambda replaces reweight. "
                    "Remove reweight or set precond=False.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            net_callable = self.net

        if antithetic:
            def loss_orthros_antithetic(model, batch, tau, iw):
                x0, x1 = batch["x0"], batch["x1"]
                z = torch.randn_like(x0)

                # positive noise: head 0 -> x0 endpoint, head 1 -> denoiser z
                x_t_p, _ = vfm_velocity_target_direct_1d(path, x0, x1, tau, z)
                x0_hat_p, eta_hat_p = model(x_t_p, tau)
                l_p = ((x0_hat_p - x0)**2 + (eta_hat_p - z)**2).sum(-1)

                # negative noise: the denoiser target flips with the noise
                x_t_m, _ = vfm_velocity_target_direct_1d(path, x0, x1, tau, -z)
                x0_hat_m, eta_hat_m = model(x_t_m, tau)
                l_m = ((x0_hat_m - x0)**2 + (eta_hat_m + z)**2).sum(-1)

                outer = lambda_fn(tau) if self.precond else resolve_outer_lambda(reweight, tau)
                return (0.5 * (l_p + l_m) * outer * iw.squeeze(-1)).mean()
            loss_orthros = loss_orthros_antithetic
        else:
            def loss_orthros_naive(model, batch, tau, iw):
                x0, x1 = batch["x0"], batch["x1"]
                z = torch.randn_like(x0)
                x_t, _ = vfm_velocity_target_direct_1d(path, x0, x1, tau, z)
                # head 0 regresses the x0 endpoint posterior E[x0|x_t];
                # head 1 regresses the denoiser E[z|x_t]
                x0_hat, eta_hat = model(x_t, tau)
                mse = ((x0_hat - x0)**2 + (eta_hat - z)**2).sum(-1)
                outer = lambda_fn(tau) if self.precond else resolve_outer_lambda(reweight, tau)
                return (mse * outer * iw.squeeze(-1)).mean()
            loss_orthros = loss_orthros_naive

        loss_orthros.required_keys = frozenset({"x0", "x1"})
        loss_orthros.requires_tau_grad = False

        train_loop(
            model=net_callable,
            model_module=self.net,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_fn=loss_orthros,
            optim=optim,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=self.time,
            scheduler=sched,
            ema=ema_net,
            grad_clip_norm=self.optim.grad_clip_norm,
            eps=self.path.eps,
        )

        self.net.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """integrate orthros time-score over tau in [eps, 1-eps].

        applies EMA if available, uses unified curve + integrator surface.
        """
        if self.net is None:
            raise RuntimeError(
                "VFMOrthros model is not trained. Call fit() before predict_ldr()."
            )

        self.net.eval()
        samples = xs.float().to(self.device)

        def time_score_fn(path, ts, xs):
            """loop per curve point; vfm_orthros_time_score_1d expects tau:[B,1] matching xs:[B,D]."""
            n = xs.shape[0]
            out = []
            for i in range(ts.shape[0]):
                tau_i = ts[i:i+1].expand(n, 1)
                out.append(
                    vfm_orthros_time_score_1d(net_eval, path, xs, tau_i, self._div_fn)
                )
            return torch.stack(out, dim=0)

        if self.ema_net is not None:
            self.ema_net.apply_to(self.net)

        # rebuild wrapper around EMA-applied net (wrapper is parameter-free)
        if self.precond:
            from ..common._precond import make_coeffs, wrap_2head
            coeff_x0 = make_coeffs(self.path, self._moments, "x0")
            coeff_eta = make_coeffs(self.path, self._moments, "noise")
            net_eval = wrap_2head(self.net, coeff_x0, coeff_eta)
        else:
            net_eval = self.net

        try:
            ldr = predict_ldr_via_curve(
                time_score_fn=time_score_fn,
                path=self.test_path,
                curve=self.curve,
                integrator=self.integrator,
                n_points=self.integration_steps,
                samples=samples,
            )
            return ldr
        finally:
            if self.ema_net is not None:
                self.ema_net.restore(self.net)


from .tri import TriangularVFMV1, TriangularVFMV2, TriangularVFM2D

__all__ = [
    "VFM", "make_vfm", "VFMOrthros",
    "TriangularVFMV1", "TriangularVFMV2", "TriangularVFM2D",
]
