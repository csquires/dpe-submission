"""TriangularVFM2D: VFM DRE with 2D-time stacked interpolant.

trains two velocity heads (b_1, b_2) and one denoiser (eta) sequentially on a
2D-time stacked interpolant; integrates the time-score along a curve at inference.
"""
from typing import Optional, Literal, Callable
import warnings

import torch

from ....common.base import ELDR
from ...common._cfgs import (
    OptimCfg, SchedCfg, EmaCfg,
    make_optim, make_sched, make_ema,
)
from ...common._weighting import resolve_outer_lambda
from ...common._estimator_helpers import _validate_and_store_slots
from ...common._paradigm_funcs import vfm_velocity_target_2d, vfm_time_score_2d
from ...common._predict_ldr import predict_ldr_via_curve
from ...common._curves import Curve, LowArcCurve2D
from ...common._integrators import Integrator, integrator_trapezoid
from ...common._trainer import train_interleaved_3
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
        n_epochs: int = 1000,
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
          hidden_dim, n_hidden_layers, n_epochs, batch_size: network and training shape.
          optim, sched, ema: configuration objects (optimcfg, schedcfg, emacfg).
          device: torch device string; defaults to cuda if available, else cpu.
          antithetic: bool; apply antithetic variance reduction in b-phase velocity loss.
          div_method, div_noise, n_hutch_samples: divergence estimation (hutchinson or exact).
          integration_steps: number of tau quadrature points for time-score integration.
          activation: mlp activation; one of {elu, gelu, silu}.
          layernorm: layer norm; one of {off, pre, post}.
          reweight: bool; apply path-variance reweighting to per-sample losses.

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
        self.n_epochs = n_epochs
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

        # initialize model
        self.init_model()

        # create optimizers from shared cfg
        optim_b = make_optim(
            list(self.net_b1.parameters()) + list(self.net_b2.parameters()),
            self.optim
        )
        optim_eta = make_optim(self.net_eta.parameters(), self.optim)

        # create schedulers from shared cfg
        sched_b = make_sched(optim_b, self.n_epochs, self.optim.lr, self.sched)
        sched_eta = make_sched(optim_eta, self.n_epochs, self.optim.lr, self.sched)

        # create emas from shared cfg
        ema_b1 = make_ema(self.net_b1, self.ema)
        ema_b2 = make_ema(self.net_b2, self.ema)
        ema_eta = make_ema(self.net_eta, self.ema)
        self.ema_b1 = ema_b1
        self.ema_b2 = ema_b2
        self.ema_eta = ema_eta

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
        net_b1, net_b2, net_eta = self.net_b1, self.net_b2, self.net_eta
        reweight = self.reweight

        if self.antithetic:
            def _compute_b(x0, x1, xstar):
                # sample 2D time via self.time (replaces hardcoded torch.rand)
                t1, t2, iw = self.time(self.batch_size, self.device)  # [B, 1] each
                z = torch.randn_like(x0)  # [B, D]
                outer = resolve_outer_lambda(reweight, t1)  # [B, 1]

                # velocity targets for both directions
                x_t, v1_star, v2_star = vfm_velocity_target_2d(
                    path, x0, x1, xstar, t1, t2, z
                )

                # positive and negative perturbations
                x_t_plus = x_t
                x_t_minus = x_t - 2 * torch.sqrt(path.gamma(t1, t2)) * z

                # b1 predictions
                b1_plus = net_b1(x_t_plus, t1, t2)    # [B, D]
                b1_minus = net_b1(x_t_minus, t1, t2)  # [B, D]

                # b2 predictions
                b2_plus = net_b2(x_t_plus, t1, t2)    # [B, D]
                b2_minus = net_b2(x_t_minus, t1, t2)  # [B, D]

                # per-sample b1 loss (antithetic)
                per_b1 = (
                    0.25 * (b1_plus ** 2).sum(dim=-1)
                    - 0.5 * (v1_star * b1_plus).sum(dim=-1)
                    + 0.25 * (b1_minus ** 2).sum(dim=-1)
                    - 0.5 * (v1_star * b1_minus).sum(dim=-1)
                ) * outer.squeeze(-1)  # [B]

                # per-sample b2 loss (antithetic)
                per_b2 = (
                    0.25 * (b2_plus ** 2).sum(dim=-1)
                    - 0.5 * (v2_star * b2_plus).sum(dim=-1)
                    + 0.25 * (b2_minus ** 2).sum(dim=-1)
                    - 0.5 * (v2_star * b2_minus).sum(dim=-1)
                ) * outer.squeeze(-1)  # [B]

                return per_b1.mean() + per_b2.mean()
        else:
            def _compute_b(x0, x1, xstar):
                t1, t2, iw = self.time(self.batch_size, self.device)
                z = torch.randn_like(x0)
                outer = resolve_outer_lambda(reweight, t1)

                x_t, v1_star, v2_star = vfm_velocity_target_2d(
                    path, x0, x1, xstar, t1, t2, z
                )

                b1_pred = net_b1(x_t, t1, t2)
                b2_pred = net_b2(x_t, t1, t2)

                per_b1 = (
                    0.5 * (b1_pred ** 2).sum(dim=-1)
                    - (v1_star * b1_pred).sum(dim=-1)
                ) * outer.squeeze(-1)

                per_b2 = (
                    0.5 * (b2_pred ** 2).sum(dim=-1)
                    - (v2_star * b2_pred).sum(dim=-1)
                ) * outer.squeeze(-1)

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
            outer = resolve_outer_lambda(reweight, t1)
            eta_pred = net_eta(x_t, t1, t2)
            per_sample = (
                0.5 * (eta_pred ** 2).sum(dim=-1) - (z * eta_pred).sum(dim=-1)
            ) * outer.squeeze(-1)
            return per_sample.mean()

        # build grad-clip param lists
        grad_clip = self.optim.grad_clip_norm
        b_params = list(net_b1.parameters()) + list(net_b2.parameters())
        eta_params = list(net_eta.parameters())

        # train with interleaved optimization
        train_interleaved_3(
            net_b1, net_b2, net_eta,
            loss_b, loss_eta,
            optim_b, optim_eta,
            n_steps=self.n_epochs,
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
        )

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

        # apply ema wrappers
        if self.ema_b1 is not None:
            self.ema_b1.apply_to(self.net_b1)
        if self.ema_b2 is not None:
            self.ema_b2.apply_to(self.net_b2)
        if self.ema_eta is not None:
            self.ema_eta.apply_to(self.net_eta)

        # define time_score_fn callback
        def time_score_fn(path, ts, samples):
            """closure over net_b1, net_b2, net_eta, _div_fn.

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
                    self.net_b1, self.net_b2, self.net_eta,
                    path, samples, t1_i, t2_i, self._div_fn,
                )
                outs.append(torch.stack([s1, s2], dim=-1))
            return torch.stack(outs, dim=0)  # [chunk_len, n_samples, 2]

        try:
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
            # restore original weights if ema was applied
            if self.ema_b1 is not None:
                self.ema_b1.restore(self.net_b1)
            if self.ema_b2 is not None:
                self.ema_b2.restore(self.net_b2)
            if self.ema_eta is not None:
                self.ema_eta.restore(self.net_eta)
