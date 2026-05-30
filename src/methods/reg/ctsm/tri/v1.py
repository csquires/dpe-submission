"""TriangularCTSMV1: continuous-time score matching DRE on a barycentric triangular path.

Uses four-slot unified surface (Pillar E). Single network predicts closed-form
regression target via MSE loss. Default time sampler is uniform (no path-aware
singularity avoidance like V2).
"""
from typing import Optional, Callable
import warnings

import torch
from torch import Tensor

from ....common.base import ELDR
from ...common._estimator_helpers import _validate_and_store_slots
from ...common._cfgs import OptimCfg, SchedCfg, EmaCfg
from ...common._cfgs import make_optim, make_sched, make_ema
from ...common._ema import EMA
from ...common._trainer import train_loop
from ...common._losses import make_sb_loss
from src.waypoints.dataclass_paths import TriangularPath1D
from src.waypoints.path_builders import (
    bary_ctsm,
)
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D


class TriangularCTSMV1(ELDR):
    """barycentric triangular CTSM with four-slot unified surface."""

    def __init__(
        self,
        input_dim: int,
        *,
        path: Optional[TriangularPath1D] = None,
        test_path: Optional[TriangularPath1D] = None,
        time = None,
        curve = None,
        integrator = None,
        sigma: float = 1.0,
        vertex: float = 0.5,
        inner_eps: float = 0.0,
        gamma_min: float = 0.0,
        test_inner_eps: float = 0.0,
        test_gamma_min: float = 0.0,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_steps: int = 1000,
        batch_size: int = 512,
        optim: OptimCfg,
        sched: SchedCfg = SchedCfg(),
        ema: EmaCfg = EmaCfg(),
        activation: str = "elu",
        integration_steps: int = 200,
        reweight: bool = False,
        device: Optional[str] = None,
    ) -> None:
        """construct estimator with four slots and network hyperparameters.

        defaults: path uses barycentric+CTSM-gamma; time uses uniform sampler;
        curve is identity; integrator is trapezoid. all defaults resolved before
        slot validation.

        clamping scalars inner_eps, gamma_min (train) and test_inner_eps,
        test_gamma_min (test, inference) control coordinate and value clamping
        in the path gamma and weights. defaults: all 0.0 (smooth gamma, no
        singularities at interior tau).
        """
        # step 1: resolve path default
        if path is None:
            path = bary_ctsm(
                sigma=sigma, vertex=vertex,
                inner_eps=inner_eps, gamma_min=gamma_min,
                eps=1e-3
            )

        # step 1b: resolve test_path default (uses test_* clamping kwargs)
        if test_path is None:
            test_path = bary_ctsm(
                sigma=sigma, vertex=vertex,
                inner_eps=test_inner_eps, gamma_min=test_gamma_min,
                eps=1e-3
            )

        # F2/F3: sampler/path inner_eps consistency
        samp_ie = getattr(time, "inner_eps", 0.0) if time is not None else 0.0
        if (samp_ie > 0) != (inner_eps > 0):
            warnings.warn(
                f"asymmetric inner_eps: sampler={samp_ie}, path={inner_eps}. "
                "Probably unintentional.", UserWarning, stacklevel=2,
            )
        elif samp_ie > 0 and inner_eps > 0:
            assert abs(samp_ie - inner_eps) < 1e-9, \
                f"sampler/path inner_eps mismatch: {samp_ie} vs {inner_eps}"

        # F4: inactive gamma_min warning (CTSM-style sigma paths only)
        if inner_eps > 0 and gamma_min > 0:
            eff_inner = sigma * (inner_eps * (1.0 - inner_eps)) ** 0.5
            if gamma_min < eff_inner:
                warnings.warn(
                    f"gamma_min={gamma_min} below coord-clamp effective floor "
                    f"{eff_inner:.4g}; gamma_min is inactive in compose order.",
                    UserWarning, stacklevel=2,
                )

        # step 2: resolve time default (uniform sampler on [eps, 1-eps])
        if time is None:
            def uniform_sampler(B: int, device: torch.device) -> tuple[Tensor, Tensor]:
                # sample tau uniformly from [eps, 1-eps]
                eps = path.eps
                tau = torch.empty(B, 1, device=device, dtype=torch.float32).uniform_(eps, 1.0 - eps)
                iw = torch.ones(B, 1, device=device, dtype=torch.float32)
                return tau, iw
            time = uniform_sampler

        # step 3: resolve curve default (identity)
        if curve is None:
            from ...common._curves import IdentityCurve1D
            curve = IdentityCurve1D()

        # step 4: resolve integrator default (trapezoid)
        if integrator is None:
            from ...common._integrators import integrator_trapezoid
            integrator = integrator_trapezoid

        # step 5: validate and store slots
        _validate_and_store_slots(
            self,
            input_dim=input_dim,
            path=path,
            time=time,
            curve=curve,
            integrator=integrator,
            expected_path_type=TriangularPath1D,
            expected_curve_dim=1,
            device=device,
        )

        # store test_path (same type by construction; no separate validation)
        self.test_path = test_path

        # store legacy scalars for HPO introspection
        self.inner_eps = inner_eps
        self.gamma_min = gamma_min
        self.test_inner_eps = test_inner_eps
        self.test_gamma_min = test_gamma_min

        # step 6: store network scalars
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.sigma = sigma
        self.vertex = vertex
        self.integration_steps = integration_steps
        self.reweight = reweight

        # step 7: validate activation
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}")
        self.activation = activation

        # step 8: store cfg objects
        self.optim = optim
        self.sched = sched
        self.ema = ema

        # step 9: initialize network and ema placeholders
        self.model = None
        self.ema_obj: Optional[EMA] = None

    def init_model(self) -> None:
        """build TimeScoreNetwork1D; optimizer/EMA are created in fit."""
        self.model = TimeScoreNetwork1D(
            self.input_dim,
            self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
        ).to(self.device)

    def fit(
        self,
        samples_p0: Tensor,
        samples_p1: Tensor,
        samples_pstar: Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """train the network via continuous-time score matching regression."""
        # step 1: initialize network
        self.init_model()

        # step 2: build optimizer and scheduler
        optim_obj = make_optim(self.model.parameters(), self.optim)
        sched_obj = make_sched(optim_obj, self.n_steps, self.optim.lr, self.sched)

        # step 3: build EMA
        self.ema_obj = make_ema(self.model, self.ema)

        # step 4: create loss function
        loss_fn = make_sb_loss(path=self.path, reweight=self.reweight)

        # step 5: build eval_fn if both callbacks and eval data are provided
        eval_fn = None
        if step_cb is not None and eval_data is not None:
            eval_pstar = eval_data["pstar"]
            eval_true_ldrs = eval_data["true_ldrs"]

            def eval_fn(_model) -> torch.Tensor:
                """mae between predict_ldr(eval_pstar) and true_ldrs; _model ignored."""
                predicted = self.predict_ldr(eval_pstar)
                target = eval_true_ldrs.to(predicted.device)
                return torch.abs(predicted - target).mean()

        # step 6: call unified training loop
        train_loop(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_fn=loss_fn,
            optim=optim_obj,
            n_steps=self.n_steps,
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

    def predict_ldr(self, xs: Tensor) -> Tensor:
        """predict log-density ratio via integration of the learned score."""
        # step 1: check fitted state
        if self.model is None:
            raise RuntimeError("model not fitted; call fit() first.")

        # step 2: move data to device and set eval mode
        self.model.eval()
        xs = xs.to(self.device)
        n_samples = xs.shape[0]

        # step 3: apply EMA if available
        if self.ema_obj is not None:
            self.ema_obj.apply_to(self.model)

        try:
            # step 4: define time-score function (inline, per contract §120-148)
            def time_score_fn(path, ts, samples):
                # ts: [chunk_len, 1]; ts[:, 0] is tau values for this chunk
                # samples: [n_samples, D]
                # broadcast and call model, return [chunk_len, n_samples]
                chunk_len = ts.shape[0]
                results = []
                for i in range(chunk_len):
                    tau_scalar = ts[i, 0]
                    tau_batch = tau_scalar.view(1, 1).expand(n_samples, 1)  # [n_samples, 1]
                    results.append(self.model(samples, tau_batch))  # [n_samples, 1]
                return torch.stack(results, dim=0).squeeze(-1)  # [chunk_len, n_samples]

            # step 5: invoke unified integrator. when the test path has a
            # non-zero coordinate clamp around the vertex, excise the same band
            # from the inference linspace so we never query the net at tau values
            # the train sampler excluded.
            from ...common._predict_ldr import predict_ldr_via_curve
            excise = None
            if self.test_inner_eps > 0.0:
                v = float(self.vertex)
                lo = max(self.test_path.eps, v - self.test_inner_eps)
                hi = min(1.0 - self.test_path.eps, v + self.test_inner_eps)
                if lo < hi:
                    excise = (lo, hi)
            ldr = predict_ldr_via_curve(
                time_score_fn=time_score_fn,
                path=self.test_path,
                curve=self.curve,
                integrator=self.integrator,
                n_points=self.integration_steps,
                samples=xs,
                excise_band=excise,
            )
            return ldr.cpu()
        finally:
            if self.ema_obj is not None:
                self.ema_obj.restore(self.model)
