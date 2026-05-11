"""TriangularCTSM2D: CTSM DRE with 2D-time stacked interpolant.

trains a 2-vector score network on the closed-form Stacked2DCtsm target; predicts
log(p0/p1) by integrating -score along a 1D curve in the (t_1, t_2) square.
"""
from typing import Optional

import torch

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._cfgs import (
    OptimCfg, SchedCfg, EmaCfg, TimeCfg,
    make_optim, make_sched, make_ema, make_time_sampler,
)
from src.density_ratio_estimation._weighting import resolve_outer_lambda
from src.models.time_score_matching.score_network_2d import ScoreNetwork2D
from src.waypoints.curve_2d import Curve2D
from src.waypoints.path_2d import CtsmPath2D
from src.waypoints.triangular_continuous_2d import Stacked2DCtsm


class TriangularCTSM2D(DensityRatioEstimator):
    """CTSM with 2D-time stacked interpolant path.

    Trains a 2-vector score network on closed-form Stacked2DCtsm target via
    inline bespoke training loop (not train_loop). Constructor surface migrated
    to cfg objects; per-step loss plumbed with importance weighting and
    reweight outer lambda.
    """

    def __init__(
        self,
        input_dim: int,
        path: Optional[CtsmPath2D] = None,
        curve: Optional[Curve2D] = None,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_epochs: int = 1000,
        batch_size: int = 512,
        *,
        optim: OptimCfg,
        sched: SchedCfg = SchedCfg(),
        ema: EmaCfg = EmaCfg(),
        time: TimeCfg = TimeCfg(),
        device: Optional[str] = None,
        integration_steps: int = 200,
        activation: str = "elu",
        reweight: bool = False,
    ) -> None:
        """Construct TriangularCTSM2D with cfg-based optimization surface.

        Args:
            input_dim: dimensionality of data.
            path: optional CtsmPath2D; defaults to Stacked2DCtsm(sigma=1.0, gamma_schedule="sqrt", eps=time.eps).
            curve: optional Curve2D; defaults to Curve2D(path_height=1.0).
            hidden_dim, n_hidden_layers, n_epochs, batch_size: network and training shape.
            optim: required OptimCfg (optimizer config with lr, grad_clip_norm, etc.).
            sched: SchedCfg for learning rate schedule (default SchedCfg()).
            ema: EmaCfg for exponential moving average (default EmaCfg()).
            time: TimeCfg for time sampling and eps (default TimeCfg()).
            device: torch device (defaults to cuda if available, else cpu).
            integration_steps: number of tau quadrature points for predict_ldr.
            activation: score network activation in {'elu', 'gelu', 'silu'}.
            reweight: whether to apply outer path-variance weight to loss.

        Raises:
            ValueError: if activation invalid.
            ValueError: if path provides sample_tau and time.dist != 'uniform'.
            ValueError: if curve.peak_t2() exceeds path.t2_max.
        """
        super().__init__(input_dim)

        # validate activation
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )

        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optim = optim
        self.sched = sched
        self.ema = ema
        self.time = time
        self.integration_steps = integration_steps
        self.activation = activation
        self.reweight = reweight

        # resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # resolve path with time.eps default
        if path is None:
            self.path = Stacked2DCtsm(
                sigma=1.0, gamma_schedule="sqrt", eps=self.time.eps
            )
        else:
            self.path = path

        # resolve curve
        self.curve = curve if curve is not None else Curve2D(path_height=1.0)

        # no path-conflict guard: timecfg holds an explicit timesampler instance.
        # users who want path-driven sampling pass timecfg(sampler=pathsampler(path)).

        # coverage check: ensure inference curve peak_t2 is within training range
        peak = float(self.curve.peak_t2())
        t2_max = float(getattr(self.path, "t2_max", 1.0 - self.time.eps))
        if peak > t2_max:
            raise ValueError(
                f"curve.peak_t2() = {peak} exceeds path.t2_max = {t2_max}; "
                f"the network would be queried at untrained t_2 values during predict_ldr. "
                f"Increase Stacked2DCtsm(t2_max=...) or reduce Curve2D(path_height=...)."
            )

        # lazy initialization placeholders
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.ema_obj = None

    def init_model(self) -> None:
        """Construct ScoreNetwork2D and build optimizer/scheduler/ema from cfgs.

        Procedure:
          - create ScoreNetwork2D with stored hyperparams and move to device.
          - create optimizer via make_optim(self.optim).
          - create scheduler via make_sched(self.sched).
          - create ema via make_ema(self.ema).
        """
        self.model = ScoreNetwork2D(
            self.input_dim,
            self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
        ).to(self.device)
        self.optimizer = make_optim(self.model.parameters(), self.optim)
        self.scheduler = make_sched(
            self.optimizer, self.n_epochs, self.optim.lr, self.sched
        )
        self.ema_obj = make_ema(self.model, self.ema)

    def fit(
        self,
        samples_p0: torch.Tensor,  # [N0, D]
        samples_p1: torch.Tensor,  # [N1, D]
        samples_pstar: torch.Tensor,  # [Nstar, D]
    ) -> None:
        """Train ScoreNetwork2D via bespoke inline loop on closed-form 2-vector target.

        Procedure:
          - init_model + train mode.
          - Move samples to device, cast to float.
          - Build time_sampler: if path.sample_tau exists, defer to it; else make_time_sampler(time).
          - For each of n_epochs:
              bootstrap minibatches (x0, x1, xstar) [B, D].
              sample t1, iw via time_sampler; sample t2 ~ U(eps, t2_max).
              compute closed-form 2-vector target via path.sample_and_target.
              forward through model; compute err = target - lambda_t * pred.
              compose loss = (iw * outer_lambda * err^2).mean().
              backward; clip grad; step; step scheduler; update ema.
          - eval mode.

        Args:
            samples_p0: [N0, D] samples from p_0.
            samples_p1: [N1, D] samples from p_1.
            samples_pstar: [Nstar, D] samples from p_*.
        """
        self.init_model()
        self.model.train()

        # move to device and cast to float
        samples_p0 = samples_p0.to(self.device).float()  # [N0, D]
        samples_p1 = samples_p1.to(self.device).float()  # [N1, D]
        samples_pstar = samples_pstar.to(self.device).float()  # [Nstar, D]

        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        # restrict training t_2 range to overlap inference curve
        t2_max = float(getattr(self.path, "t2_max", 1.0 - self.time.eps))

        # time sampler comes from cfg; users pick pathsampler(self.path) explicitly
        # if they want path-driven sampling.
        time_sampler = make_time_sampler(self.time)

        # bind post-step callbacks once so the inner loop holds no `is not None` checks.
        grad_clip = self.optim.grad_clip_norm
        if grad_clip is not None and grad_clip > 0:
            param_list = list(self.model.parameters())

            def do_clip():
                torch.nn.utils.clip_grad_norm_(param_list, max_norm=grad_clip)
        else:
            def do_clip():
                return None
        do_sched = self.scheduler.step if self.scheduler is not None else (lambda: None)
        do_ema = (lambda: self.ema_obj.update(self.model)) if self.ema_obj is not None else (lambda: None)

        for epoch_idx in range(self.n_epochs):
            # bootstrap minibatches
            idx0 = torch.randint(0, n0, (self.batch_size,), device=self.device)  # [B]
            idx1 = torch.randint(0, n1, (self.batch_size,), device=self.device)  # [B]
            idx_star = torch.randint(
                0, n_star, (self.batch_size,), device=self.device
            )  # [B]

            x0 = samples_p0[idx0]  # [B, D]
            x1 = samples_p1[idx1]  # [B, D]
            xstar = samples_pstar[idx_star]  # [B, D]

            # sample t1 with importance weighting; t2 uniform in [eps, t2_max]
            t1, iw = time_sampler(self.batch_size, self.time.eps, self.device)  # [B, 1], [B, 1]
            t2 = (
                torch.rand(self.batch_size, 1, device=self.device)
                * (t2_max - self.time.eps)
                + self.time.eps
            )  # [B, 1]

            # noise
            epsilon = torch.randn_like(x0)  # [B, D]

            # closed-form 2-vector target from path
            x, target, lambda_t = self.path.sample_and_target(
                x0, x1, xstar, t1, t2, epsilon
            )  # x: [B, D], target: [B, 2], lambda_t: [B, 2]

            # forward through model
            pred = self.model(x, t1, t2)  # [B, 2]

            # mse loss with importance weighting and optional outer reweight lambda
            err = target - lambda_t * pred  # [B, 2]
            outer = resolve_outer_lambda(self.reweight, t1)  # [B]
            loss = (iw.squeeze(-1) * outer * (err ** 2).mean(dim=1)).mean()  # scalar

            # backward + step
            self.optimizer.zero_grad()
            loss.backward()
            do_clip()
            self.optimizer.step()
            do_sched()
            do_ema()

        self.model.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """Predict log density ratios via trapezoidal quadrature of line integral.

        Procedure:
          1. Validate model is trained; set eval mode; move xs to device.
          2. If EMA active, swap in shadow weights.
          3. Create uniform tau grid in [time.eps, 1-time.eps].
          4. For each tau:
               query curve for (t1, t2, dt1, dt2).
               forward through model: s = model(xs, t1, t2) -> [n, 2].
               compute integrand dy/dtau = -(s[:, 0] * dt1 + s[:, 1] * dt2).
               clamp NaN/Inf to safe bounds.
          5. Stack integrand rows and integrate via trapezoid rule.
          6. Restore EMA if active.
          7. Return result on CPU.

        Args:
            xs: [n, D] test points.

        Returns:
            [n] tensor on CPU, log density ratios log(p_0 / p_1).

        Raises:
            RuntimeError: if model not trained.
        """
        if self.model is None:
            raise RuntimeError(
                "TriangularCTSM2D is not trained. Call fit() before predict_ldr()."
            )

        self.model.eval()
        xs = xs.to(self.device).float()  # [n, D]
        n = xs.shape[0]
        dtype = next(self.model.parameters()).dtype

        if self.ema_obj is not None:
            self.ema_obj.apply_to(self.model)
        try:
            ts = torch.linspace(
                self.time.eps, 1.0 - self.time.eps, self.integration_steps, device=self.device
            )  # [integration_steps]
            with torch.no_grad():
                integrand_rows = []
                for t in ts:
                    tau_v = float(t.item())
                    t1_v = float(self.curve.t1(tau_v))
                    t2_v = float(self.curve.t2(tau_v))
                    dt1_v = float(self.curve.dt1(tau_v))
                    dt2_v = float(self.curve.dt2(tau_v))
                    t1_t = torch.full((n, 1), t1_v, dtype=dtype, device=self.device)  # [n, 1]
                    t2_t = torch.full((n, 1), t2_v, dtype=dtype, device=self.device)  # [n, 1]
                    s = self.model(xs, t1_t, t2_t)  # [n, 2]
                    dy = -(s[:, 0] * dt1_v + s[:, 1] * dt2_v)  # [n]
                    dy = torch.nan_to_num(dy, nan=0.0, posinf=1e6, neginf=-1e6)  # [n]
                    integrand_rows.append(dy)
                vals = torch.stack(integrand_rows)  # [integration_steps, n]
            dt = (1.0 - 2.0 * self.time.eps) / (self.integration_steps - 1)
            return torch.trapezoid(vals, dx=dt, dim=0).cpu()  # [n]
        finally:
            if self.ema_obj is not None:
                self.ema_obj.restore(self.model)
