"""TriangularCTSM2D: CTSM DRE with 2D-time stacked interpolant.

Trains a 2-vector score network on the closed-form regression target;
predicts log(p0/p1) by integrating -score along a 1D curve in the (t1, t2) square.
"""
from typing import Optional, Callable
import warnings

import torch

from ....common.base import ELDR
from ...common._cfgs import (
    OptimCfg, SchedCfg, EmaCfg,
    make_optim, make_sched, make_ema,
)
from ...common._estimator_helpers import _validate_and_store_slots
from ...common._weighting import resolve_outer_lambda
from ...common._paradigm_funcs import ctsm_regression_target_2d
from ...common._predict_ldr import predict_ldr_via_curve
from ....common._report import _make_report
from src.models.time_score_matching.score_network_2d import ScoreNetwork2D
from src.waypoints.dataclass_paths import TriangularPath2D
from src.methods.reg.common._curves import Curve
from src.waypoints.path_builders import rect_ctsm
from src.methods.reg.common._curves import LowArcCurve2D
from src.methods.reg.common._integrators import Integrator, integrator_trapezoid
from src.methods.reg.common._time_samplers import (
    TimeSampler2D, make_uniform, make_uniform_scaled, make_product,
)


class TriangularCTSM2D(ELDR):
    """CTSM with 2D-time stacked interpolant path.

    Trains a 2-vector score network on closed-form regression target via
    importance-weighted MSE loss. Constructor uses four-slot surface (path,
    time, curve, integrator) with full defaults.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        path: Optional[TriangularPath2D] = None,
        time: Optional[TimeSampler2D] = None,
        curve: Optional[Curve] = None,
        integrator: Integrator = integrator_trapezoid,
        inner_eps: float = 0.02,
        gamma_min: float = 0.0,
        test_inner_eps: float = 0.0,
        test_gamma_min: float = 0.0,
        test_path: Optional[TriangularPath2D] = None,
        # network / training scalars
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_epochs: int = 1000,
        batch_size: int = 512,
        optim: Optional[OptimCfg] = None,
        sched: Optional[SchedCfg] = None,
        ema: Optional[EmaCfg] = None,
        device: Optional[str] = None,
        integration_steps: int = 200,
        activation: str = "elu",
        reweight: bool = False,
    ) -> None:
        """Construct TriangularCTSM2D with four-slot surface.

        Args:
            input_dim: dimensionality of data.
            path: optional TriangularPath2D; defaults to stacked 2D CTSM path.
            time: optional TimeSampler2D; defaults to product of uniforms.
            curve: optional Curve; defaults to LowArcCurve2D(path_height=1.0).
            integrator: integration scheme for predict_ldr; defaults to trapezoid.
            hidden_dim, n_hidden_layers, n_epochs, batch_size: network and training hyperparams.
            optim: required OptimCfg (optimizer config with lr, grad_clip_norm, etc.).
            sched: SchedCfg for learning rate schedule; defaults to SchedCfg().
            ema: EmaCfg for exponential moving average; defaults to EmaCfg().
            device: torch device; defaults to cuda if available, else cpu.
            integration_steps: number of tau quadrature points for predict_ldr.
            activation: score network activation in {'elu', 'gelu', 'silu'}.
            reweight: whether to apply outer path-variance weight to loss.
        """
        super().__init__(input_dim)

        # validate activation
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )

        # store cfg objects and scalars
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optim = optim if optim is not None else OptimCfg(lr=1e-3)
        self.sched = sched if sched is not None else SchedCfg()
        self.ema = ema if ema is not None else EmaCfg()
        self.integration_steps = integration_steps
        self.activation = activation
        self.reweight = reweight

        # resolve device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # resolve defaults before validation. caller-provided `time=` owns
        # the t2-domain bound; otherwise we fall back to a hardcoded default.
        if path is None:
            path = rect_ctsm(sigma=1.0, inner_eps=inner_eps, gamma_min=gamma_min, eps=1e-3)

        if test_path is None:
            test_path = rect_ctsm(sigma=1.0, inner_eps=test_inner_eps, gamma_min=test_gamma_min, eps=1e-3)

        if time is None:
            time = make_product(
                make_uniform(eps=path.eps),
                make_uniform_scaled(eps=path.eps, max=0.3),
            )

        if curve is None:
            curve = LowArcCurve2D(path_height=1.0)

        # F4 inactive gamma_min warning (CTSM-style sigma paths only)
        if inner_eps > 0 and gamma_min > 0:
            eff_inner = 1.0 * (inner_eps * (1.0 - inner_eps)) ** 0.5
            if gamma_min < eff_inner:
                warnings.warn(
                    f"gamma_min={gamma_min} below coord-clamp effective floor "
                    f"{eff_inner:.4g}; gamma_min is inactive in compose order.",
                    UserWarning, stacklevel=2,
                )

        # validate and store the four slots
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


        # store test path and clamping scalars
        self.test_path = test_path
        self.inner_eps = inner_eps
        self.gamma_min = gamma_min
        self.test_inner_eps = test_inner_eps
        self.test_gamma_min = test_gamma_min

        # lazy initialization placeholder
        self.model = None
        self.ema_obj = None

    def fit(
        self,
        samples_p0: torch.Tensor,  # [N0, D]
        samples_p1: torch.Tensor,  # [N1, D]
        samples_pstar: torch.Tensor,  # [Nstar, D]
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """Train single score network via importance-weighted CTSM regression.

        Args:
            samples_p0: [N0, D] samples from p_0.
            samples_p1: [N1, D] samples from p_1.
            samples_pstar: [Nstar, D] samples from p_*.
            step_cb: optional callback Callable[[int, float], None] invoked at intervals.
                Receives (step_number, eval_mae) during training for pruning/monitoring.
            eval_data: optional dict with keys "pstar" and "true_ldrs" (both torch.Tensor).
                If provided along with step_cb, enables held-out evaluation.
            step_cb_interval: frequency (in steps) at which to invoke step_cb; default 50.
        """
        # move to device and cast to float
        samples_p0 = samples_p0.to(self.device).float()  # [N0, D]
        samples_p1 = samples_p1.to(self.device).float()  # [N1, D]
        samples_pstar = samples_pstar.to(self.device).float()  # [Nstar, D]

        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        # build single 2-output network
        self.model = ScoreNetwork2D(
            self.input_dim,
            self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
        ).to(self.device)

        # build optimizer, scheduler, ema from cfgs
        optimizer = make_optim(self.model.parameters(), self.optim)
        scheduler = make_sched(optimizer, self.n_epochs, self.optim.lr, self.sched)
        ema_obj = make_ema(self.model, self.ema)

        self.model.train()

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

        do_report = _make_report(step_cb, step_cb_interval, eval_fn, self.model, self.model)

        # extract references for closure
        device = self.device
        path = self.path
        model = self.model
        reweight = self.reweight
        time_sampler = self.time  # TimeSampler2D callable

        # bind post-step callbacks once
        grad_clip = self.optim.grad_clip_norm
        if grad_clip is not None and grad_clip > 0:
            param_list = list(model.parameters())
            def do_clip():
                torch.nn.utils.clip_grad_norm_(param_list, max_norm=grad_clip)
        else:
            def do_clip():
                pass

        do_sched = scheduler.step if scheduler is not None else (lambda: None)
        do_ema = (lambda: ema_obj.update(model)) if ema_obj is not None else (lambda: None)

        # training loop
        for epoch in range(self.n_epochs):
            # bootstrap minibatches
            idx0 = torch.randint(0, n0, (self.batch_size,), device=device)  # [B]
            idx1 = torch.randint(0, n1, (self.batch_size,), device=device)  # [B]
            idx_star = torch.randint(0, n_star, (self.batch_size,), device=device)  # [B]

            x0 = samples_p0[idx0]  # [B, D]
            x1 = samples_p1[idx1]  # [B, D]
            xstar = samples_pstar[idx_star]  # [B, D]

            # sample (t1, t2) via time sampler
            t1, t2, iw = time_sampler(self.batch_size, device)  # [B, 1], [B, 1], [B]

            # sample noise
            epsilon = torch.randn_like(x0)  # [B, D]

            # closed-form 2-vector target
            x_t, target, lambda_t = ctsm_regression_target_2d(
                path, x0, x1, xstar, t1, t2, epsilon,
            )  # x_t [B, D], target [B, 2], lambda_t [B, 2]

            # forward
            pred = model(x_t, t1, t2)  # [B, 2]

            # loss: weighted mse per-axis, aggregated
            err = target - lambda_t * pred  # [B, 2]
            outer = resolve_outer_lambda(reweight, t1)  # [B]
            loss = (iw.squeeze(-1) * outer * (err ** 2).sum(dim=1)).mean()  # scalar

            # backward + step
            optimizer.zero_grad()
            loss.backward()
            do_clip()
            optimizer.step()
            do_sched()
            do_ema()
            do_report()

        self.model.eval()
        self.ema_obj = ema_obj

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """Predict log density ratios via curve-integrated time-score.

        Args:
            xs: [n, D] test points.

        Returns:
            [n] tensor on CPU.
        """
        if self.model is None:
            raise RuntimeError("TriangularCTSM2D not trained. Call fit() before predict_ldr().")

        self.model.eval()
        xs = xs.to(self.device).float()  # [n, D]

        if self.ema_obj is not None:
            self.ema_obj.apply_to(self.model)

        try:
            # define time-score callable
            def time_score_fn(path, ts, samples):
                """Unpack (t1, t2) from ts [n, 2] and return scores [n, m, 2].

                Args:
                    path: unused (for consistency with signature).
                    ts: [n, 2] curve points (t1, t2 per column).
                    samples: [m, D] test points.

                Returns:
                    [n, m, 2] scores; axis 2 is (score_t1, score_t2).
                """
                m = samples.shape[0]
                out = []
                for i in range(ts.shape[0]):
                    t1_i = ts[i:i+1, 0:1].expand(m, 1)
                    t2_i = ts[i:i+1, 1:2].expand(m, 1)
                    out.append(self.model(samples, t1_i, t2_i))
                return torch.stack(out, dim=0)  # [n, m, 2]

            result = predict_ldr_via_curve(
                time_score_fn=time_score_fn,
                path=self.test_path,
                curve=self.curve,
                integrator=self.integrator,
                n_points=self.integration_steps,
                samples=xs,
            )  # [n] on CPU

            return result
        finally:
            if self.ema_obj is not None:
                self.ema_obj.restore(self.model)
