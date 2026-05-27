"""TriangularCTSMV2: continuous-time score matching DRE on a piecewise-Schroedinger-bridge triangular path.

V2 uses piecewise-linear weights (two legs joined at vertex) and a piecewise-SB gamma schedule
with a hard floor gamma_min at the vertex. The path has a forbidden support band of width O(inner_eps)
around tau=vertex where the path singularity creates gradient bias. By default, time sampler is
path-aware (make_piecewise_sb_sampler), automatically excluding the forbidden band at training time.

Explicit time sampler can be provided; users accepting the default assume the time sampler
avoids the forbidden band.
"""
from typing import Optional, Callable
import warnings

import torch
from torch import Tensor

from src.waypoints.path_builders import psb
from src.waypoints.dataclass_paths import TriangularPath1D

from ...common._time_samplers import (
    make_piecewise_sb_sampler,
    make_uniform,
    TimeSampler1D,
)

from ...common._curves import IdentityCurve1D, Curve
from ...common._integrators import integrator_trapezoid, Integrator

from ...common._estimator_helpers import _validate_and_store_slots
from ...common._cfgs import OptimCfg, SchedCfg, EmaCfg
from ...common._cfgs import make_optim, make_sched, make_ema

from src.methods.common.base import ELDR
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D


class TriangularCTSMV2(ELDR):
    """CTSM with a piecewise-Schroedinger-bridge triangular path."""

    def __init__(
        self,
        input_dim: int,
        *,
        path: Optional[TriangularPath1D] = None,
        time: Optional[TimeSampler1D] = None,
        curve: Curve = IdentityCurve1D(),
        integrator: Integrator = integrator_trapezoid,
        sigma: float = 1.0,
        vertex: float = 0.5,
        inner_eps: float = 0.02,
        gamma_min: float = 0.0,
        test_inner_eps: float = 0.0,
        test_gamma_min: float = 0.0,
        # network and training scalars
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_steps: int = 1000,
        batch_size: int = 512,
        optim: OptimCfg = None,
        sched: SchedCfg = None,
        ema: EmaCfg = None,
        device: Optional[str] = None,
        activation: str = "elu",
        integration_steps: int = 200,
        reweight: bool = False,
    ) -> None:
        """Piecewise-SB CTSM with optional path-aware time sampler.

        If path is None, construct a TriangularPath1D with piecewise-SB weights
        and CTSM piecewise-SB gamma. If time is None and inner_eps > 0, construct
        make_piecewise_sb_sampler tuned to the path's geometry; else use uniform.

        Args:
            input_dim: data dimensionality.
            path: optional TriangularPath1D. If None, auto-construct from
                sigma, vertex, inner_eps, gamma_min (training defaults).
            time: optional TimeSampler1D. If None, auto-construct based on
                inner_eps: path-aware sampler if inner_eps > 0, else uniform.
            curve: Curve protocol instance (default IdentityCurve1D).
            integrator: Integrator callable (default integrator_trapezoid).
            sigma: noise amplitude scale for CTSM variance schedule.
            vertex: triangular path peak (default 0.5).
            inner_eps: coordinate-clamp half-width (default 0.02, legacy V2).
            gamma_min: value floor for gamma (default 0.0, no floor).
            test_inner_eps: coordinate-clamp for inference (default 0.0, unclamped).
            test_gamma_min: value floor for inference (default 0.0, unclamped).
            hidden_dim, n_hidden_layers, n_steps, batch_size, activation,
            integration_steps, reweight: network and training parameters.
            optim, sched, ema: optimizer, scheduler, EMA configs.
            device: torch device (auto-selected if None).
        """
        super().__init__(input_dim)

        # resolve defaults before validation (contract requirement)
        if path is None:
            path = psb(
                sigma=sigma,
                vertex=vertex,
                inner_eps=inner_eps,
                gamma_min=gamma_min,
                eps=1e-3,
            )

        # always build test_path with test defaults
        test_path = psb(
            sigma=sigma,
            vertex=vertex,
            inner_eps=test_inner_eps,
            gamma_min=test_gamma_min,
            eps=1e-3,
        )

        if time is None:
            if inner_eps > 0:
                time = make_piecewise_sb_sampler(
                    vertex=vertex,
                    inner_eps=inner_eps,
                    eps=path.eps,
                )
            else:
                time = make_uniform(eps=path.eps)

        # F2/F3: sampler/path inner_eps consistency
        samp_ie = getattr(time, "inner_eps", 0.0) if time is not None else 0.0
        if (samp_ie > 0) != (inner_eps > 0):
            warnings.warn(
                f"asymmetric inner_eps: sampler={samp_ie}, path={inner_eps}. "
                "Probably unintentional.",
                UserWarning,
                stacklevel=2,
            )
        elif samp_ie > 0 and inner_eps > 0:
            assert abs(samp_ie - inner_eps) < 1e-9, \
                f"sampler/path inner_eps mismatch: {samp_ie} vs {inner_eps}"

        # F4: inactive gamma_min warning (CTSM-family sigma paths only)
        if inner_eps > 0 and gamma_min > 0:
            eff_inner = sigma * (inner_eps * (1 - inner_eps)) ** 0.5
            if gamma_min < eff_inner:
                warnings.warn(
                    f"gamma_min={gamma_min} below coord-clamp effective floor "
                    f"{eff_inner:.4g}; gamma_min is inactive in compose order.",
                    UserWarning,
                    stacklevel=2,
                )

        # validate and store four-slot surface
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

        # store test_path and clamping scalars for HPO introspection
        self.test_path = test_path
        self.inner_eps = inner_eps
        self.gamma_min = gamma_min
        self.test_inner_eps = test_inner_eps
        self.test_gamma_min = test_gamma_min

        # store CTSM-specific scalars for HPO introspection
        self.sigma = sigma
        self.vertex = vertex

        # store training hyperparameters
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.activation = activation
        self.integration_steps = integration_steps
        self.reweight = reweight

        # store configs (field names: optim, sched, ema; not optim_cfg etc)
        if optim is None:
            raise TypeError("optim is required (keyword-only, non-default)")
        self.optim = optim
        self.sched = sched if sched is not None else SchedCfg()
        self.ema = ema if ema is not None else EmaCfg()

        # initialize network placeholders
        self.model: Optional[TimeScoreNetwork1D] = None
        self.ema_obj: Optional[object] = None

    def fit(
        self,
        samples_p0: Tensor,
        samples_p1: Tensor,
        samples_pstar: Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> "TriangularCTSMV2":
        """fit the CTSM model via one-phase continuous-time score matching.

        constructs network, loss function, and calls train_loop with
        the piecewise-SB path and time sampler.

        Args:
            samples_p0, samples_p1, samples_pstar: training sample triplets [N, D].

        Returns:
            self for method chaining.
        """
        from ...common._trainer import train_loop
        from ...common._losses import make_sb_loss

        # initialize network if not already done
        if self.model is None:
            self.model = TimeScoreNetwork1D(
                self.input_dim,
                self.hidden_dim,
                n_hidden_layers=self.n_hidden_layers,
                activation=self.activation,
            ).to(self.device)

        # construct loss function with piecewise-SB path
        loss_fn = make_sb_loss(
            path=self.path,
            reweight=self.reweight,
        )

        # construct optimizer, scheduler, EMA from configs
        optim_obj = make_optim(self.model.parameters(), self.optim)
        sched_obj = make_sched(optim_obj, self.n_steps, self.optim.lr, self.sched)
        self.ema_obj = make_ema(self.model, self.ema)

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

        # run training loop
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

        return self

    def predict_ldr(self, xs: Tensor) -> Tensor:
        """predict log-density ratio via curve integration.

        trapezoid-integrates -model over tau in [eps, 1-eps]; uses EMA shadow if set.

        Args:
            xs: sample points [B, D].

        Returns:
            log-density ratio estimates [B].
        """
        if self.model is None:
            raise RuntimeError("model not fitted; call fit() first.")

        self.model.eval()
        xs = xs.to(self.device)
        n = xs.shape[0]

        if self.ema_obj is not None:
            self.ema_obj.apply_to(self.model)

        try:
            ts = torch.linspace(
                self.test_path.eps,
                1.0 - self.test_path.eps,
                self.integration_steps,
                device=self.device,
            )
            with torch.no_grad():
                # inline CTSM time-score; space-first model(x, tau) matches
                # TimeScoreNetwork.forward and make_sb_loss training
                # stack over tau grid: [n_points, B]
                vals = torch.stack([
                    -self.model(xs, torch.full((n, 1), float(t.item()), device=self.device))
                    for t in ts
                ]).squeeze(-1)  # [n_points, B]
            dt = (1.0 - 2.0 * self.test_path.eps) / (self.integration_steps - 1)
            return torch.trapezoid(vals, dx=dt, dim=0).cpu()  # [B]
        finally:
            if self.ema_obj is not None:
                self.ema_obj.restore(self.model)
