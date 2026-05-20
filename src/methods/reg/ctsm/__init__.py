"""CTSM family: continuous-time score matching DRE.

The two-source (DRE) variant CTSM uses the four-slot surface (path, time, curve,
integrator) defined in Pillar E of notes/waypoints_unification_mid_level.md.

Stock CTSM trains a single score network s_phi(x, tau) under SB loss
(Schroedinger-bridge setting, path parameterization via DirectPath1D).
Inference integrates -s_phi over tau in [eps, 1-eps] using the provided
integrator and curve (typically identity 1D for closed-form tau integration).

triangular variants (V1 barycentric, V2 piecewise-SB, V3 2D) live under `.tri`.
"""
from typing import Callable, Optional
import warnings

import torch

from ..common._cfgs import (
    OptimCfg, SchedCfg, EmaCfg,
    make_optim, make_sched, make_ema,
)
from ..common._ema import EMA
from ..common._trainer import train_loop
from ..common._paradigm_funcs import ctsm_regression_target_direct_1d
from ..common._estimator_helpers import _validate_and_store_slots
from ..common._predict_ldr import predict_ldr_via_curve
from ..common._time_samplers import make_uniform
from src.waypoints.dataclass_paths import DirectPath1D
from src.waypoints.path_builders import direct_ctsm
from src.methods.reg.common._curves import IdentityCurve1D
from src.methods.reg.common._integrators import integrator_trapezoid
from ...common.base import DRE
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D


class CTSM(DRE):
    """four-slot CTSM for DRE via SB loss; single regression head.

    trains s_phi(x, tau) under SB loss (Schroedinger-bridge setting,
    direct path with optional coord/value clamping).
    integrates -score over tau in [eps, 1-eps] at inference using separate test-path.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        path: Optional[DirectPath1D] = None,
        time: Optional[object] = None,
        curve: Optional[object] = None,
        integrator: Optional[object] = None,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_epochs: int = 1000,
        batch_size: int = 512,
        optim: OptimCfg = None,
        sched: SchedCfg = None,
        ema: EmaCfg = None,
        device: Optional[str] = None,
        sigma: float = 1.0,
        activation: str = "elu",
        integration_steps: int = 200,
        reweight: bool = False,
        inner_eps: float = 0.0,
        gamma_min: float = 0.0,
        test_inner_eps: float = 0.0,
        test_gamma_min: float = 0.0,
        test_path: Optional["DirectPath1D"] = None,
    ) -> None:
        """four-slot CTSM for DRE via SB loss; single regression head.

        Args:
            input_dim: feature dimension.
            path: DirectPath1D with linear weights (no x_star) and noise schedule gamma.
                if None, defaults to direct_ctsm with train clamping.
            time: TimeSampler1D callable returning (tau, iw) for training.
                if None, defaults to make_uniform(eps=path.eps).
            curve: Curve object with .points(tau) and .derivatives(tau); must have dim==1.
                defaults to IdentityCurve1D().
            integrator: Integrator callable(scores, taus) -> Tensor.
                defaults to integrator_trapezoid.
            hidden_dim: width of score network hidden layers.
            n_hidden_layers: depth of score network.
            n_epochs: training iterations.
            batch_size: minibatch size.
            optim: optimizer config (required keyword-only).
            sched: scheduler config; defaults to no scheduling.
            ema: EMA config; defaults to no EMA.
            device: torch device; auto-detects GPU if None.
            sigma: noise scale for the closed-form regression target.
            activation: nonlinearity {"elu", "gelu", "silu"}.
            integration_steps: discretization points for inference over [eps, 1-eps].
            reweight: apply outer reweighting in loss.
            inner_eps: coord-clamp window half-width for train path.
            gamma_min: value floor for train path gamma schedule.
            test_inner_eps: coord-clamp window for test path; defaults to 0.0.
            test_gamma_min: value floor for test path gamma schedule; defaults to 0.0.
        """
        super().__init__(input_dim)

        # step 3a: path and time defaults
        if path is None:
            path = direct_ctsm(
                sigma=sigma,
                inner_eps=inner_eps,
                gamma_min=gamma_min,
                eps=1e-3,
            )
        if test_path is None:
            test_path = direct_ctsm(
                sigma=sigma,
                inner_eps=test_inner_eps,
                gamma_min=test_gamma_min,
                eps=1e-3,
            )
        if time is None:
            time = make_uniform(eps=path.eps)
        if curve is None:
            curve = IdentityCurve1D()
        if integrator is None:
            integrator = integrator_trapezoid
        if sched is None:
            sched = SchedCfg()
        if ema is None:
            ema = EmaCfg()

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

        # F4 inactive gamma_min warning (CTSM-style sigma paths only)
        if inner_eps > 0 and gamma_min > 0:
            eff_inner = sigma * (inner_eps * (1.0 - inner_eps)) ** 0.5
            if gamma_min < eff_inner:
                warnings.warn(
                    f"gamma_min={gamma_min} below coord-clamp effective floor "
                    f"{eff_inner:.4g}; gamma_min is inactive in compose order.",
                    UserWarning, stacklevel=2,
                )

        # step 3b: validate and store four slots
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

        # store test_path (no separate validation; same type by construction)
        self.test_path = test_path

        # store legacy scalars for HPO introspection
        self.inner_eps = inner_eps
        self.gamma_min = gamma_min
        self.test_inner_eps = test_inner_eps
        self.test_gamma_min = test_gamma_min

        # step 3c: store network/training scalars
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.sigma = sigma
        self.integration_steps = integration_steps
        self.reweight = reweight

        if optim is None:
            raise TypeError("optim is required (keyword-only, non-default)")
        self.optim = optim
        self.sched = sched
        self.ema = ema

        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )
        self.activation = activation

        # step 3d: network placeholder (frozen name for checkpoint compat)
        self.model = None
        self.ema_obj: Optional[EMA] = None

    def init_model(self) -> None:
        """build single score network TimeScoreNetwork1D on device."""
        self.model = TimeScoreNetwork1D(
            self.input_dim,
            self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
        ).to(self.device)

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """train the single score network via SB loss under direct path parameterization.

        build optimizer, scheduler, EMA, and loss closure; delegate to train_loop
        with paradigm-specific regression target from ctsm_regression_target_direct_1d.

        Args:
            samples_p0: [N0, input_dim] samples from p0.
            samples_p1: [N1, input_dim] samples from p1.
            step_cb: optional callback invoked every step_cb_interval steps with
                (step_index: int, score: float). when None, no instrumentation.
            eval_data: optional dict with keys "pstar" and "true_ldrs" (paired by
                index). used to build eval_fn closure only if step_cb is not None.
                when None, eval_fn is not constructed.
            step_cb_interval: interval (in minibatch updates) for step_cb invocation
                (default 50).
        """
        self.init_model()

        # step 5a: move data to device
        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)

        # step 5b: build single optimizer, scheduler, EMA
        optim_obj = make_optim(self.model.parameters(), self.optim)
        sched_obj = make_sched(optim_obj, self.n_epochs, self.optim.lr, self.sched)
        self.ema_obj = make_ema(self.model, self.ema)

        # step 5c: define loss closure
        # wire ctsm_regression_target_direct_1d with path, sigma; call it inside loss_fn.
        path_arg = self.path
        sigma_arg = self.sigma
        reweight_arg = self.reweight

        def loss_fn(model, batch, tau, iw):
            """compute SB loss for direct CTSM.

            tau shape [B, 1], iw shape [B, 1].
            ctsm_regression_target_direct_1d returns (x_tau, target, lambda_t), all [B, D].
            """
            x0, x1 = batch["x0"], batch["x1"]
            epsilon = torch.randn_like(x0)

            x_tau, target, lambda_t = ctsm_regression_target_direct_1d(
                path_arg, x0, x1, tau, epsilon,
            )

            # score prediction at sampled point
            pred = model(x_tau, tau)  # [B, 1]

            # SB loss: MSE weighted by lambda_t, optionally reweighted
            residual = target - lambda_t * pred  # [B, D]
            loss_per_sample = (residual ** 2).mean(dim=-1)  # [B]

            # apply outer reweighting if requested (tau-dependent weighting)
            if reweight_arg:
                outer_weight = (tau.squeeze(-1) * (1 - tau.squeeze(-1))).clamp(min=1e-8)
            else:
                outer_weight = 1.0

            return (loss_per_sample * outer_weight * iw.squeeze(-1)).mean()

        loss_fn.required_keys = frozenset({"x0", "x1"})
        loss_fn.requires_tau_grad = False

        # build eval_fn if both callbacks and eval data are provided.
        # eval_data["pstar"] is the holdout eval inputs; "true_ldrs" the paired
        # ground-truth log-density-ratios used by Hyperband's per-step pruning.
        eval_fn = None
        if step_cb is not None and eval_data is not None:
            eval_pstar = eval_data["pstar"]
            eval_true_ldrs = eval_data["true_ldrs"]

            def eval_fn(_model) -> torch.Tensor:
                """mae between predict_ldr(eval_pstar) and true_ldrs."""
                predicted = self.predict_ldr(eval_pstar)
                target = eval_true_ldrs.to(predicted.device)
                return torch.abs(predicted - target).mean()

        # step 5d: call train_loop
        train_loop(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_fn=loss_fn,
            optim=optim_obj,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=self.time,
            scheduler=sched_obj,
            ema=self.ema_obj,
            grad_clip_norm=self.optim.grad_clip_norm,
            eps=self.path.eps,
            step_cb=step_cb,
            eval_fn=eval_fn,
            step_cb_interval=step_cb_interval,
        )

        # step 5e: set to eval mode
        self.model.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """integrate -score over tau in [eps, 1-eps]; return LDR estimate on CPU.

        calls predict_ldr_via_curve with a time-score closure that evaluates
        the single network model(x, tau). EMA shadow applied if available.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        samples = xs.float().to(self.device)

        # step 6a: apply EMA if set; use try/finally to restore
        if self.ema_obj is not None:
            self.ema_obj.apply_to(self.model)

        try:
            # step 6b: define time-score closure
            # takes (path, ts, samples) and returns scores [n_points, n_samples]
            def time_score_fn(path, ts, x):
                """time-score: -model(x, tau) as a vmap-ready closure.

                Args:
                    path: DirectPath1D (unused in direct CTSM).
                    ts: Tensor [chunk_len, 1] (curve.points(tau) for identity 1D).
                    x: Tensor [n_samples, data_dim].

                Returns:
                    Tensor [chunk_len, n_samples] containing -model(x, ts[:, 0]).
                """
                n_chunk = ts.shape[0]  # chunk_len
                n_samps = x.shape[0]   # n_samples

                # expand tau to [chunk_len, n_samples, 1] for broadcasting
                tau_expanded = ts.unsqueeze(1).expand(n_chunk, n_samps, 1)  # [chunk_len, n_samples, 1]
                x_expanded = x.unsqueeze(0).expand(n_chunk, n_samps, -1)    # [chunk_len, n_samples, data_dim]

                # reshape to flat batch for network inference
                tau_flat = tau_expanded.reshape(n_chunk * n_samps, 1)       # [chunk_len * n_samples, 1]
                x_flat = x_expanded.reshape(n_chunk * n_samps, -1)          # [chunk_len * n_samples, data_dim]

                # network call; returns [chunk_len * n_samples, 1]
                # arg order (x, tau) matches TimeScoreNetwork1D.forward and training
                score_flat = self.model(x_flat, tau_flat)

                # reshape back and squeeze spatial dim
                score = score_flat.reshape(n_chunk, n_samps, 1)             # [chunk_len, n_samples, 1]
                score_agg = score.squeeze(-1)                               # [chunk_len, n_samples]
                return -score_agg

            # step 6c: call unified inference function
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
            if self.ema_obj is not None:
                self.ema_obj.restore(self.model)


from .tri import TriangularCTSMV1, TriangularCTSMV2, TriangularCTSM2D

__all__ = ["CTSM", "TriangularCTSMV1", "TriangularCTSMV2", "TriangularCTSM2D"]
