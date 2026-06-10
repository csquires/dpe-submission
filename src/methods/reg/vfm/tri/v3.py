"""TriangularVFM2D: VFM DRE with 2D-time stacked interpolant.

trains two velocity heads (b_1, b_2) and one denoiser (eta) sequentially on a
2D-time stacked interpolant; integrates the time-score along a curve at inference.
"""
import functools
from typing import Optional, Literal, Callable
import warnings

import torch

from ....common.base import ELDR
from ...common._cfgs import (
    OptimCfg, SchedCfg, EmaCfg,
    make_optim, make_sched, make_ema,
)
from ...common._weighting import resolve_outer_lambda, outer_path_var_v3
from ...common._estimator_helpers import _validate_and_store_slots
from ...common._paradigm_funcs import vfm_velocity_target_2d, vfm_time_score_2d
from ...common._predict_ldr import predict_ldr_via_curve
from ...common._curves import Curve, LowArcCurve2D
from ...common._integrators import Integrator, integrator_trapezoid
from ...common._trainer import train_interleaved_3
from ...common._sequential import train_two_phase_3
from ...common._precond import endpoint_moments, make_coeffs_2d, make_lambda_2d, wrap_2d
from src.waypoints.dataclass_paths import TriangularPath2D
from src.waypoints.path_builders import rect_vfm
from src.methods.reg.common._time_samplers import TimeSampler2D, make_uniform, make_uniform_scaled, make_product
from src.models.flow.div_estimators import build_div_fn
from src.models.time_score_matching.velocity_network_2d import MLP2D


class TriangularVFM2D(ELDR):
    """vfm dre with 2d-time stacked interpolant path.

    trains two velocity heads (b_1, b_2) and one denoiser (eta) sequentially
    on three distributions p_0, p_1, p_*. inference integrates the time-score
    along self.curve from tau=eps to 1-eps via predict_ldr_via_curve.
    clamping: inner_eps and gamma_min for train path; test_inner_eps and
    test_gamma_min for inference path. train path uses gamma_min=0.05 by default
    (legacy vfm v3 behavior); test path uses gamma_min=0.0 by default for unclipped
    inference.

    contract: fit(samples_p0, samples_p1, samples_pstar) with three [n, d]
    tensors; predict_ldr(xs) returns log(p_0/p_1) as [n_samples] cpu tensor.

    training_strategy controls whether b1/b2/eta are optimized together (interleaved,
    default) or sequentially (b1+b2 first, then eta). when training_strategy='sequential',
    optuna pruning (step_cb) is not supported; hyperband will raise notimplementederror.
    use interleaved for hpo.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        path: Optional[TriangularPath2D] = None,
        test_path: Optional[TriangularPath2D] = None,
        time: Optional[TimeSampler2D] = None,
        curve: Optional[Curve] = None,
        integrator: Integrator = integrator_trapezoid,
        inner_eps: float = 0.0,
        gamma_min: float = 0.05,
        test_inner_eps: float = 0.0,
        test_gamma_min: float = 0.0,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_steps: int = 1000,
        batch_size: int = 512,
        optim: Optional[OptimCfg] = None,
        sched: Optional[SchedCfg] = None,
        ema: Optional[EmaCfg] = None,
        device: Optional[str] = None,
        antithetic: bool = True,
        div_method: Literal['hutchinson', 'exact'] = 'hutchinson',
        div_noise: Literal['rademacher', 'gaussian'] = 'rademacher',
        n_hutch_samples: int = 1,
        integration_steps: int = 200,
        activation: str = "gelu",
        layernorm: str = "off",
        reweight: bool = False,
        precond: bool = False,
        training_strategy: str = "interleaved",
        strategy_cfg: dict | None = None,
        early_stop_cfg: dict | None = None,
    ) -> None:
        """initialize triangularvfm2d with four-slot surface and 2d-time sampling.

        trains two velocity heads (b1, b2) and one denoiser (eta) sequentially on
        a 2d-time stacked interpolant path; integrates the time-score along curve
        at inference via predict_ldr_via_curve.

        constructor arguments:
          input_dim: dimensionality of data.
          path: optional triangularpath2d; defaults to rect_vfm(k=20.0, ...).
                defines the stacked 2d interpolant and its eps and t2_max bounds.
          test_path: optional triangularpath2d for inference; if none, built from
                     rect_vfm with test_inner_eps, test_gamma_min.
          time: optional timesampler2d; defaults to make_product of two make_uniform samplers.
                returns (t1, t2, iw) on each call; scaled to the t2_max kwarg.
          curve: optional curve; defaults to lowarccurve2d(path_height=1.0).
                maps tau in [eps, 1-eps] to (t1, t2) coordinates; must have dim=2.
          integrator: callable(scores [n_points, n_samples], tau [n_points]) -> [n_samples].
                defaults to integrator_trapezoid. trapezoid, mean, or simpson.
          inner_eps: float; coordinate clamping window half-width for train path.
                     defaults to 0.0 (no coord clamping).
          gamma_min: float; lower bound on gamma(t) for train path.
                     defaults to 0.05 (legacy vfm v3 gamma_min).
          test_inner_eps: float; coordinate clamping for test path.
                          defaults to 0.0.
          test_gamma_min: float; value floor for test path.
                          defaults to 0.0.
          hidden_dim, n_hidden_layers, n_steps, batch_size: network and training shape.
          optim, sched, ema: configuration objects (optimcfg, schedcfg, emacfg).
          device: torch device string; defaults to cuda if available, else cpu.
          antithetic: bool; apply antithetic variance reduction in b-phase velocity loss.
          div_method, div_noise, n_hutch_samples: divergence estimation (hutchinson or exact).
          integration_steps: number of tau quadrature points for time-score integration.
          activation: mlp activation; one of {elu, gelu, silu}.
          layernorm: layer norm; one of {off, pre, post}.
          reweight: bool; apply path-variance reweighting to per-sample losses.
          precond: bool; apply edm-style affine preconditioning to all three networks.
                   defaults to False. when True, masks reweight setting (edm lambda replaces path variance).

        raises:
          typeerror: if path is not triangularpath2d, time is not callable,
                    curve.dim != 2, or integrator is not callable.
        """
        super().__init__(input_dim)

        # resolve all four slots BEFORE validation. caller-provided `time=`
        # owns the t2-domain bound; otherwise fall back to a hardcoded default.
        if path is None:
            path = rect_vfm(
                k=20.0,
                inner_eps=inner_eps, gamma_min=gamma_min, eps=1e-3,
            )
        if test_path is None:
            test_path = rect_vfm(
                k=20.0,
                inner_eps=test_inner_eps, gamma_min=test_gamma_min, eps=1e-3,
            )
        if curve is None:
            curve = LowArcCurve2D(path_height=1.0)
        if time is None:
            time = make_product(
                make_uniform(eps=path.eps),
                make_uniform_scaled(eps=path.eps, max=0.3),
            )

        # f2/f3 sampler/path inner_eps consistency check
        samp_ie = getattr(time, "inner_eps", 0.0) if time is not None else 0.0
        if (samp_ie > 0) != (inner_eps > 0):
            warnings.warn(
                f"asymmetric inner_eps: sampler={samp_ie}, path={inner_eps}. "
                "Probably unintentional.", UserWarning, stacklevel=2,
            )
        elif samp_ie > 0 and inner_eps > 0:
            assert abs(samp_ie - inner_eps) < 1e-9, \
                f"sampler/path inner_eps mismatch: {samp_ie} vs {inner_eps}"

        # validate and store slots
        _validate_and_store_slots(
            self,
            input_dim=input_dim,
            path=path,
            time=time,
            curve=curve,
            integrator=integrator,
            expected_path_type=TriangularPath2D,
            expected_curve_dim=2,
            device=device,
        )


        # store test_path (no separate validation; same type by construction)
        self.test_path = test_path

        # store clamping scalars for hpo introspection
        self.inner_eps = inner_eps
        self.gamma_min = gamma_min
        self.test_inner_eps = test_inner_eps
        self.test_gamma_min = test_gamma_min

        # store training/network scalars
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.optim = optim if optim is not None else OptimCfg(lr=1e-3)
        self.sched = sched if sched is not None else SchedCfg()
        self.ema = ema if ema is not None else EmaCfg()
        self.antithetic = antithetic
        self.div_method = div_method
        self.div_noise = div_noise
        self.n_hutch_samples = n_hutch_samples
        self.integration_steps = integration_steps
        self.activation = activation
        self.layernorm = layernorm
        self.reweight = reweight
        self.precond = precond
        self._moments = None  # populated in fit() if precond=True

        # validate scalars
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}")
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off', 'pre', 'post'}}; got {layernorm!r}")
        if div_method not in ('hutchinson', 'exact'):
            raise ValueError(f"div_method must be 'hutchinson' or 'exact'; got {div_method!r}")
        if div_noise not in ('rademacher', 'gaussian'):
            raise ValueError(f"div_noise must be 'rademacher' or 'gaussian'; got {div_noise!r}")
        if n_hutch_samples < 1:
            raise ValueError(f"n_hutch_samples must be >= 1; got {n_hutch_samples}")

        # build divergence estimator
        self._div_fn = build_div_fn(
            self.div_method,
            noise=self.div_noise,
            n_samples=self.n_hutch_samples
        )

        # network placeholders
        self.net_b1: Optional[object] = None
        self.net_b2: Optional[object] = None
        self.net_eta: Optional[object] = None
        self.ema_b1: Optional[object] = None
        self.ema_b2: Optional[object] = None
        self.ema_eta: Optional[object] = None

        # store training strategy configuration
        self.training_strategy = training_strategy
        self.strategy_cfg = strategy_cfg or {}
        self.early_stop_cfg = early_stop_cfg

        # bind self._train based on training strategy
        if training_strategy == "interleaved":
            self._train = functools.partial(
                train_interleaved_3,
                n_steps=self.n_steps,
                early_stop_cfg=self.early_stop_cfg,
            )
        elif training_strategy == "sequential":
            n_steps_b = self.strategy_cfg.get("n_steps_b", self.n_steps // 2)
            n_steps_eta = self.strategy_cfg.get("n_steps_eta", self.n_steps - n_steps_b)
            self._train = functools.partial(
                train_two_phase_3,
                n_steps_b=n_steps_b,
                n_steps_eta=n_steps_eta,
                early_stop_cfg=self.early_stop_cfg,
            )
        else:
            raise ValueError(f"unknown training_strategy: {training_strategy!r}")

    def init_model(self) -> None:
        """instantiate net_b1, net_b2, net_eta as mlp2d on self.device.

        called once at the start of fit(); networks are moved to self.device
        and ready for training. layer norm and activation are read from
        self.activation and self.layernorm (from Pillar A / spec 04).
        """
        self.net_b1 = MLP2D(
            self.input_dim,
            self.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            layernorm=self.layernorm,
        ).to(self.device)
        self.net_b2 = MLP2D(
            self.input_dim,
            self.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            layernorm=self.layernorm,
        ).to(self.device)
        self.net_eta = MLP2D(
            self.input_dim,
            self.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            layernorm=self.layernorm,
        ).to(self.device)

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """init networks; train b1/b2 and eta via interleaved optimization.

        uses self.path, self.time, self.optim, self.sched, self.ema configs
        to build optimizers, schedulers, and ema wrappers per network.
        self.time(B, device) -> (t1, t2, iw) replaces hardcoded torch.rand.
        velocity group {b1,b2} and eta advance together each step (interleaved),
        with optional optuna pruning via step_cb/eval_data.

        args:
            samples_p0, samples_p1, samples_pstar: [n, d] sample tensors.
            step_cb: optional callback(step, metric) for optuna pruning.
            eval_data: optional dict with "pstar" and "true_ldrs" for eval_fn.
            step_cb_interval: call step_cb every n steps (default 50).
        """
        meta_out: dict = {}
        n_star = samples_pstar.shape[0]

        if n_star < 1:
            raise ValueError(f"samples_pstar must have at least 1 row; got n_star={n_star}")

        if n_star < self.batch_size // 4:
            warnings.warn(
                f"n_star={n_star} is small relative to batch_size={self.batch_size}; "
                f"pstar bootstrap will sample with high replication"
            )

        # move samples to device and cast to float
        samples_p0 = samples_p0.float().to(self.device)  # [n0, D]
        samples_p1 = samples_p1.float().to(self.device)  # [n1, D]
        samples_pstar = samples_pstar.float().to(self.device)  # [n_star, D]

        # compute endpoint moments if preconditioning enabled
        if self.precond:
            self._moments = endpoint_moments({
                "x0": samples_p0,
                "x1": samples_p1,
                "xstar": samples_pstar,
            })

        # initialize model
        self.init_model()

        # create optimizers from shared cfg
        optim_b = make_optim(
            list(self.net_b1.parameters()) + list(self.net_b2.parameters()),
            self.optim
        )
        optim_eta = make_optim(self.net_eta.parameters(), self.optim)

        # create schedulers from shared cfg
        sched_b = make_sched(optim_b, self.n_steps, self.optim.lr, self.sched)
        sched_eta = make_sched(optim_eta, self.n_steps, self.optim.lr, self.sched)

        # create emas from shared cfg
        ema_b1 = make_ema(self.net_b1, self.ema)
        ema_b2 = make_ema(self.net_b2, self.ema)
        ema_eta = make_ema(self.net_eta, self.ema)
        self.ema_b1 = ema_b1
        self.ema_b2 = ema_b2
        self.ema_eta = ema_eta

        # build precond wrappers if enabled
        if self.precond:
            if self.reweight:
                warnings.warn(
                    "precond=True will mask reweight setting; "
                    "EDM lambda replaces path-variance weighting.",
                    UserWarning, stacklevel=2,
                )
            coeff_b1 = make_coeffs_2d(self.path, self._moments, "velocity_1")
            coeff_b2 = make_coeffs_2d(self.path, self._moments, "velocity_2")
            coeff_eta = make_coeffs_2d(self.path, self._moments, "noise")
            lambda_b1 = make_lambda_2d(coeff_b1)
            lambda_b2 = make_lambda_2d(coeff_b2)
            lambda_eta = make_lambda_2d(coeff_eta)
            b1_fwd = wrap_2d(self.net_b1, coeff_b1)
            b2_fwd = wrap_2d(self.net_b2, coeff_b2)
            eta_fwd = wrap_2d(self.net_eta, coeff_eta)
        else:
            lambda_b1 = lambda_b2 = lambda_eta = None
            b1_fwd = self.net_b1
            b2_fwd = self.net_b2
            eta_fwd = self.net_eta

        # build eval_fn if both callbacks and eval data are provided
        eval_fn = None
        if step_cb is not None and eval_data is not None:
            eval_pstar = eval_data["pstar"]
            eval_true_ldrs = eval_data["true_ldrs"]

            def eval_fn(_model) -> torch.Tensor:
                """mae between predict_ldr(eval_pstar) and true_ldrs; _model ignored."""
                predicted = self.predict_ldr(eval_pstar)
                target = eval_true_ldrs.to(predicted.device)
                return torch.abs(predicted - target).mean()

        # build loss_b thunk: joint velocity loss from both net_b1 and net_b2
        n0, n1, n_star = samples_p0.shape[0], samples_p1.shape[0], samples_pstar.shape[0]
        path = self.path
        reweight = self.reweight
        precond = self.precond

        # lambda-asymmetry diagnostic state (only updated when precond=True).
        # tracks running mean E[lambda_b1] / E[lambda_b2] across batches;
        # emits a one-shot UserWarning if the ratio drifts outside [0.1, 10]
        # after at least _LAM_WARN_AFTER steps. Flags potential gradient
        # interference under the shared b-phase optimizer.
        _lam_b1_sum = 0.0
        _lam_b2_sum = 0.0
        _lam_step_count = 0
        _lam_warned = False
        _LAM_WARN_AFTER = 100
        _LAM_RATIO_HI = 10.0
        _LAM_RATIO_LO = 0.1

        if self.antithetic:
            def _compute_b(x0, x1, xstar):
                nonlocal _lam_b1_sum, _lam_b2_sum, _lam_step_count, _lam_warned
                # sample 2D time via self.time (replaces hardcoded torch.rand)
                t1, t2, iw = self.time(self.batch_size, self.device)  # [B, 1] each
                z = torch.randn_like(x0)  # [B, D]

                # antithetic: build BOTH branches end-to-end via the helper.
                # x_t_minus = mu - gamma*z, with v*_minus = dmu + dgamma*(-z).
                # reusing the +z velocities for the minus branch would bias b.
                x_t_p, v1_star_p, v2_star_p = vfm_velocity_target_2d(
                    path, x0, x1, xstar, t1, t2, z
                )
                x_t_m, v1_star_m, v2_star_m = vfm_velocity_target_2d(
                    path, x0, x1, xstar, t1, t2, -z
                )

                # b1 predictions
                b1_p = b1_fwd(x_t_p, t1, t2)  # [B, D]
                b1_m = b1_fwd(x_t_m, t1, t2)
                # b2 predictions
                b2_p = b2_fwd(x_t_p, t1, t2)
                b2_m = b2_fwd(x_t_m, t1, t2)

                # per-sample b1 loss, 0.5 * (lp + lm) per V1/V2 convention.
                # iw applied multiplicatively per the sampler's importance
                # weight; identical to V3 CTSM and the existing 1D V1/V2 vfm.
                lp_b1 = 0.5 * (b1_p ** 2).sum(dim=-1) - (v1_star_p * b1_p).sum(dim=-1)
                lm_b1 = 0.5 * (b1_m ** 2).sum(dim=-1) - (v1_star_m * b1_m).sum(dim=-1)

                if precond:
                    out_b1 = lambda_b1(t1, t2)  # [B]
                    out_b2 = lambda_b2(t1, t2)  # [B]
                    # lambda-asymmetry diagnostic: track running mean ratio,
                    # warn once if ratio drifts outside [_LAM_RATIO_LO, _LAM_RATIO_HI]
                    # after at least _LAM_WARN_AFTER steps.
                    _lam_b1_sum += out_b1.mean().item()
                    _lam_b2_sum += out_b2.mean().item()
                    _lam_step_count += 1
                    if not _lam_warned and _lam_step_count >= _LAM_WARN_AFTER:
                        e_b1 = _lam_b1_sum / _lam_step_count
                        e_b2 = max(_lam_b2_sum / _lam_step_count, 1e-12)
                        r = e_b1 / e_b2
                        if r > _LAM_RATIO_HI or r < _LAM_RATIO_LO:
                            warnings.warn(
                                f"V3 precond lambda_b1/lambda_b2 ratio = {r:.3f} "
                                f"after {_lam_step_count} steps; gradient "
                                f"interference possible under shared b-optimizer. "
                                f"Consider split optimizers or loss normalization.",
                                UserWarning, stacklevel=2,
                            )
                            _lam_warned = True
                else:
                    outer = outer_path_var_v3(t1, t2, path.gamma) if reweight \
                            else resolve_outer_lambda(False, t1)  # [B, 1]
                    out_b1 = out_b2 = outer.squeeze(-1)  # [B]

                per_b1 = 0.5 * (lp_b1 + lm_b1) * out_b1 * iw.squeeze(-1)

                # per-sample b2 loss
                lp_b2 = 0.5 * (b2_p ** 2).sum(dim=-1) - (v2_star_p * b2_p).sum(dim=-1)
                lm_b2 = 0.5 * (b2_m ** 2).sum(dim=-1) - (v2_star_m * b2_m).sum(dim=-1)
                per_b2 = 0.5 * (lp_b2 + lm_b2) * out_b2 * iw.squeeze(-1)

                return per_b1.mean() + per_b2.mean()
        else:
            def _compute_b(x0, x1, xstar):
                nonlocal _lam_b1_sum, _lam_b2_sum, _lam_step_count, _lam_warned
                t1, t2, iw = self.time(self.batch_size, self.device)
                z = torch.randn_like(x0)

                x_t, v1_star, v2_star = vfm_velocity_target_2d(
                    path, x0, x1, xstar, t1, t2, z
                )

                b1_pred = b1_fwd(x_t, t1, t2)
                b2_pred = b2_fwd(x_t, t1, t2)

                if precond:
                    out_b1 = lambda_b1(t1, t2)  # [B]
                    out_b2 = lambda_b2(t1, t2)  # [B]
                    # lambda-asymmetry diagnostic (same as antithetic branch)
                    _lam_b1_sum += out_b1.mean().item()
                    _lam_b2_sum += out_b2.mean().item()
                    _lam_step_count += 1
                    if not _lam_warned and _lam_step_count >= _LAM_WARN_AFTER:
                        e_b1 = _lam_b1_sum / _lam_step_count
                        e_b2 = max(_lam_b2_sum / _lam_step_count, 1e-12)
                        r = e_b1 / e_b2
                        if r > _LAM_RATIO_HI or r < _LAM_RATIO_LO:
                            warnings.warn(
                                f"V3 precond lambda_b1/lambda_b2 ratio = {r:.3f} "
                                f"after {_lam_step_count} steps; gradient "
                                f"interference possible under shared b-optimizer. "
                                f"Consider split optimizers or loss normalization.",
                                UserWarning, stacklevel=2,
                            )
                            _lam_warned = True
                else:
                    outer = outer_path_var_v3(t1, t2, path.gamma) if reweight \
                            else resolve_outer_lambda(False, t1)  # [B, 1]
                    out_b1 = out_b2 = outer.squeeze(-1)  # [B]

                per_b1 = (
                    0.5 * (b1_pred ** 2).sum(dim=-1)
                    - (v1_star * b1_pred).sum(dim=-1)
                ) * out_b1 * iw.squeeze(-1)

                per_b2 = (
                    0.5 * (b2_pred ** 2).sum(dim=-1)
                    - (v2_star * b2_pred).sum(dim=-1)
                ) * out_b2 * iw.squeeze(-1)

                return per_b1.mean() + per_b2.mean()

        def loss_b():
            idx0 = torch.randint(0, n0, (self.batch_size,), device=self.device)
            idx1 = torch.randint(0, n1, (self.batch_size,), device=self.device)
            idx_star = torch.randint(0, n_star, (self.batch_size,), device=self.device)
            return _compute_b(samples_p0[idx0], samples_p1[idx1], samples_pstar[idx_star])

        # build loss_eta thunk: denoiser loss
        def loss_eta():
            idx0 = torch.randint(0, n0, (self.batch_size,), device=self.device)
            idx1 = torch.randint(0, n1, (self.batch_size,), device=self.device)
            idx_star = torch.randint(0, n_star, (self.batch_size,), device=self.device)
            x0, x1, xstar = samples_p0[idx0], samples_p1[idx1], samples_pstar[idx_star]
            t1, t2, iw = self.time(self.batch_size, self.device)
            z = torch.randn_like(x0)
            w = path.weights(t1, t2)
            mu = w.alpha * x0 + w.beta * x1 + w.w_star * xstar
            gamma_t = path.gamma(t1, t2)
            x_t = (mu + gamma_t * z).detach()
            eta_pred = eta_fwd(x_t, t1, t2)

            if precond:
                out_eta = lambda_eta(t1, t2)  # [B]
            else:
                outer = outer_path_var_v3(t1, t2, path.gamma) if reweight \
                        else resolve_outer_lambda(False, t1)  # [B, 1]
                out_eta = outer.squeeze(-1)  # [B]

            per_sample = (
                0.5 * (eta_pred ** 2).sum(dim=-1) - (z * eta_pred).sum(dim=-1)
            ) * out_eta * iw.squeeze(-1)
            return per_sample.mean()

        # build grad-clip param lists
        grad_clip = self.optim.grad_clip_norm
        b_params = list(self.net_b1.parameters()) + list(self.net_b2.parameters())
        eta_params = list(self.net_eta.parameters())

        # train via strategy-specific trainer
        self._train(
            self.net_b1, self.net_b2, self.net_eta,
            loss_b, loss_eta,
            optim_b, optim_eta,
            scheduler_b=sched_b,
            scheduler_eta=sched_eta,
            ema_b1=ema_b1,
            ema_b2=ema_b2,
            ema_eta=ema_eta,
            b_params=b_params,
            eta_params=eta_params,
            grad_clip_norm=grad_clip,
            step_cb=step_cb,
            eval_fn=eval_fn,
            step_cb_interval=step_cb_interval,
            _meta_out=meta_out,
        )

        # store training metadata for introspection
        self._final_step = meta_out.get("final_step", self.n_steps)
        self._stop_reason = meta_out.get("stop_reason", None)

        # post-training cleanup
        self.net_b1.eval()
        self.net_b2.eval()
        self.net_eta.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """estimate log p_0(x) / p_1(x) via time-score line integral.

        integrates the time-score (from net_b1, net_b2, net_eta) along self.curve
        in [path.eps, 1-path.eps] via predict_ldr_via_curve and self.integrator.
        ema applied/restored around the inference block.

        args:
            xs: [n, d] test points; moved to self.device.

        returns:
            [n] log density ratios, cpu float32.

        raises:
            runtimeerror: if any network is uninitialized (call fit first).
        """
        if self.net_b1 is None or self.net_b2 is None or self.net_eta is None:
            raise RuntimeError(
                "TriangularVFM2D model is not trained. Call fit() before predict_ldr()."
            )

        self.net_b1.eval()
        self.net_b2.eval()
        self.net_eta.eval()

        samples = xs.float().to(self.device)  # [n_samples, D]

        try:
            # apply ema wrappers inside try for safe restoration
            if self.ema_b1 is not None:
                self.ema_b1.apply_to(self.net_b1)
            if self.ema_b2 is not None:
                self.ema_b2.apply_to(self.net_b2)
            if self.ema_eta is not None:
                self.ema_eta.apply_to(self.net_eta)

            # build precond wrappers if enabled; rebuilt AFTER ema apply so
            # wrappers see ema-averaged weights.
            if self.precond:
                coeff_b1 = make_coeffs_2d(self.path, self._moments, "velocity_1")
                coeff_b2 = make_coeffs_2d(self.path, self._moments, "velocity_2")
                coeff_eta = make_coeffs_2d(self.path, self._moments, "noise")
                b1_use = wrap_2d(self.net_b1, coeff_b1)
                b2_use = wrap_2d(self.net_b2, coeff_b2)
                eta_use = wrap_2d(self.net_eta, coeff_eta)
            else:
                b1_use = self.net_b1
                b2_use = self.net_b2
                eta_use = self.net_eta

            # define time_score_fn callback
            def time_score_fn(path, ts, samples):
                """closure over nets and _div_fn.

                args:
                    path: triangularpath2d (passed by predict_ldr_via_curve).
                    ts: [chunk_len, 2] curve points (t1, t2 coordinates).
                    samples: [n_samples, D] test points (broadcast).

                returns:
                    [chunk_len, n_samples, 2] raw per-axis time-scores (s1, s2).
                    chain rule applied inside predict_ldr_via_curve.
                """
                # loop over chunk_len; vfm_time_score_2d expects t1, t2 [B,1]
                # matching samples [B, D] batch dim.
                m = samples.shape[0]
                outs = []
                for i in range(ts.shape[0]):
                    t1_i = ts[i:i+1, 0:1].expand(m, 1)
                    t2_i = ts[i:i+1, 1:2].expand(m, 1)
                    s1, s2 = vfm_time_score_2d(
                        b1_use, b2_use, eta_use,
                        path, samples, t1_i, t2_i, self._div_fn,
                    )
                    outs.append(torch.stack([s1, s2], dim=-1))
                return torch.stack(outs, dim=0)  # [chunk_len, n_samples, 2]

            # existing inference call (UNCHANGED -- still inside try)
            ldr = predict_ldr_via_curve(
                time_score_fn=time_score_fn,
                path=self.test_path,
                curve=self.curve,
                integrator=self.integrator,
                n_points=self.integration_steps,
                samples=samples,
            )
            return ldr  # [n_samples] on CPU
        finally:
            # restore original weights if ema was applied (UNCHANGED)
            if self.ema_b1 is not None:
                self.ema_b1.restore(self.net_b1)
            if self.ema_b2 is not None:
                self.ema_b2.restore(self.net_b2)
            if self.ema_eta is not None:
                self.ema_eta.restore(self.net_eta)
