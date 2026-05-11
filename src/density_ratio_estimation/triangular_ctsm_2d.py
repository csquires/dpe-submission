"""TriangularCTSM2D: V3 (2D-time stacked-interpolant) triangular CTSM density-ratio estimator.

Trains a 2-vector score network on the closed-form regression target from
Stacked2DCtsm, then performs density-ratio estimation via line integral along
a 1D curve in the (t_1, t_2) square (default LowArc curve).

Contract: fit(samples_p0, samples_p1, samples_pstar) with three [N, D] tensors.
predict_ldr(xs) returns log density ratios as a 1D CPU tensor [N], following
V2's sign convention dy/dtau = -score so the integral yields log(p_0 / p_1).

Mirrors V2's TriangularCTSM in src/density_ratio_estimation/triangular_ctsm.py.
"""
from typing import Optional

import torch
import torch.optim as optim

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._ema import EMA, maybe_clip_grad, sample_time_and_iw
from src.models.time_score_matching.score_network_2d import ScoreNetwork2D
from src.waypoints.curve_2d import Curve2D
from src.waypoints.path_2d import CtsmPath2D
from src.waypoints.triangular_continuous_2d import Stacked2DCtsm


class TriangularCTSM2D(DensityRatioEstimator):
    """V3 triangular CTSM with 2D-time stacked interpolant.

    Constructor:
      input_dim: int, feature dimension D.
      path: optional CtsmPath2D; defaults to Stacked2DCtsm(sigma=1.0, gamma_schedule="sqrt", eps=eps).
      curve: optional Curve2D; defaults to Curve2D(path_height=1.0).
      hidden_dim: int, score network hidden width.
      n_hidden_layers: int, number of hidden layers in score network backbone (default 3).
      n_epochs: int, training epochs (one minibatch step per epoch).
      batch_size: int, minibatch size.
      lr: float, Adam learning rate.
      eps: float, boundary epsilon for tau and (t_1, t_2) sampling.
      device: optional str; auto-resolves to cuda or cpu.
      integration_steps: int, number of tau quadrature points for predict_ldr.
      log_every: int, log per-head losses every N epochs (0 = disabled).
      ema_decay: optional float, exponential moving average decay (e.g. 0.999).
      grad_clip_norm: optional float, max gradient norm for clipping.
      time_dist: str in {"uniform", "beta_2_2", "beta_5_5"}, time sampling distribution.
      activation: str in {"elu", "gelu", "silu"}, score network activation.
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
        lr: float = 1e-3,
        eps: float = 1e-3,
        device: Optional[str] = None,
        integration_steps: int = 200,
        log_every: int = 100,
        ema_decay: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        time_dist: str = "uniform",
        activation: str = "elu",
    ) -> None:
        super().__init__(input_dim)

        # validate and store hyperparameters
        if time_dist not in {"uniform", "beta_2_2", "beta_5_5"}:
            raise ValueError(
                f"time_dist must be in {{'uniform', 'beta_2_2', 'beta_5_5'}}; "
                f"got {time_dist!r}"
            )
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )

        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.integration_steps = integration_steps
        self.log_every = log_every
        self.ema_decay = ema_decay
        self.grad_clip_norm = grad_clip_norm
        self.time_dist = time_dist
        self.activation = activation

        # device resolution
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # initialize path and curve with defaults
        self.path = path if path is not None else Stacked2DCtsm(
            sigma=1.0, gamma_schedule="sqrt", eps=eps
        )
        self.curve = curve if curve is not None else Curve2D(path_height=1.0)

        # coverage check: ensure inference curve peak_t2 is within training range
        peak = float(self.curve.peak_t2())
        t2_max = float(getattr(self.path, "t2_max", 1.0 - self.eps))
        if peak > t2_max:
            raise ValueError(
                f"curve.peak_t2() = {peak} exceeds path.t2_max = {t2_max}; "
                f"the network would be queried at untrained t_2 values during predict_ldr. "
                f"Increase Stacked2DCtsm(t2_max=...) or reduce Curve2D(path_height=...)."
            )

        # lazy initialization placeholders
        self.model = None
        self.optimizer = None
        self.ema: Optional[EMA] = None

    def init_model(self) -> None:
        """Construct ScoreNetwork2D + Adam optimizer on self.device.

        Procedure:
          - create ScoreNetwork2D with stored hyperparams.
          - move to device.
          - create Adam optimizer.
          - if ema_decay is set, create EMA; else set self.ema = None.
        """
        self.model = ScoreNetwork2D(
            self.input_dim,
            self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.ema = (
            EMA(self.model, self.ema_decay)
            if self.ema_decay is not None
            else None
        )

    def fit(
        self,
        samples_p0: torch.Tensor,  # [N0, D]
        samples_p1: torch.Tensor,  # [N1, D]
        samples_pstar: torch.Tensor,  # [Nstar, D]
    ) -> None:
        """Train ScoreNetwork2D on the closed-form 2-vector target from self.path.

        Procedure:
          - init_model + train mode.
          - Move samples to device, cast to float.
          - Read t2_max from self.path (defaults to 1 - eps if path lacks the attribute).
          - For each of n_epochs:
              bootstrap minibatches (x0, x1, xstar) [B, D].
              sample t1 ~ time_dist with importance weights, t2 ~ U(eps, t2_max).
              sample noise epsilon ~ N(0, I).
              compute closed-form 2-vector target via self.path.sample_and_target.
              forward through model: pred = model(x, t1, t2) -> [B, 2].
              compute mse loss with importance weighting over both heads.
              backward; step; optionally update EMA.
              log per-head losses every log_every epochs (if > 0).
          - eval mode.

        Args:
            samples_p0: [N0, D] samples from p_0.
            samples_p1: [N1, D] samples from p_1.
            samples_pstar: [Nstar, D] samples from p_*.
        """
        self.init_model()
        self.model.train()

        # move to device and cast to float
        samples_p0 = samples_p0.to(self.device).float()
        samples_p1 = samples_p1.to(self.device).float()
        samples_pstar = samples_pstar.to(self.device).float()

        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        # restrict training t_2 range to overlap inference curve
        t2_max = float(getattr(self.path, "t2_max", 1.0 - self.eps))

        for epoch_idx in range(self.n_epochs):
            # bootstrap minibatches
            idx0 = torch.randint(0, n0, (self.batch_size,), device=self.device)
            idx1 = torch.randint(0, n1, (self.batch_size,), device=self.device)
            idx_star = torch.randint(
                0, n_star, (self.batch_size,), device=self.device
            )

            x0 = samples_p0[idx0]  # [B, D]
            x1 = samples_p1[idx1]  # [B, D]
            xstar = samples_pstar[idx_star]  # [B, D]

            # sample t1 with importance weighting; t2 uniform in [eps, t2_max]
            t1, iw = sample_time_and_iw(
                self.time_dist, self.batch_size, self.eps, self.device
            )  # [B, 1], [B, 1]
            t2 = (
                torch.rand(self.batch_size, 1, device=self.device)
                * (t2_max - self.eps)
                + self.eps
            )  # [B, 1]

            # noise
            epsilon = torch.randn_like(x0)  # [B, D]

            # closed-form 2-vector target from path
            x, target, lambda_t = self.path.sample_and_target(
                x0, x1, xstar, t1, t2, epsilon
            )
            # x: [B, D], target: [B, 2], lambda_t: [B, 2]

            # forward through model
            pred = self.model(x, t1, t2)  # [B, 2]

            # mse loss with importance weighting; mean over batch and both heads
            err = target - lambda_t * pred  # [B, 2]
            loss = (iw * (err ** 2)).mean()  # broadcast iw [B, 1] over [B, 2]

            # diagnostic: per-head losses (if logging enabled)
            if self.log_every > 0 and (epoch_idx + 1) % self.log_every == 0:
                with torch.no_grad():
                    loss_h1 = (err[:, 0] ** 2).mean().item()
                    loss_h2 = (err[:, 1] ** 2).mean().item()
                print(
                    f"epoch {epoch_idx + 1}: loss={loss.item():.4f} "
                    f"(head1={loss_h1:.4f}, head2={loss_h2:.4f})"
                )

            # backward + step
            self.optimizer.zero_grad()
            loss.backward()
            maybe_clip_grad(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model)

        self.model.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """Predict log density ratios via trapezoidal quadrature of line integral.

        Procedure:
          1. Validate model is trained; set eval mode; move xs to device.
          2. If EMA active, swap in shadow weights.
          3. Create uniform tau grid in [eps, 1-eps].
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
        xs = xs.to(self.device).float()
        n = xs.shape[0]
        dtype = next(self.model.parameters()).dtype

        if self.ema is not None:
            self.ema.apply_to(self.model)
        try:
            ts = torch.linspace(
                self.eps, 1.0 - self.eps, self.integration_steps, device=self.device
            )
            with torch.no_grad():
                integrand_rows = []
                for t in ts:
                    tau_v = float(t.item())
                    t1_v = float(self.curve.t1(tau_v))
                    t2_v = float(self.curve.t2(tau_v))
                    dt1_v = float(self.curve.dt1(tau_v))
                    dt2_v = float(self.curve.dt2(tau_v))
                    t1_t = torch.full((n, 1), t1_v, dtype=dtype, device=self.device)
                    t2_t = torch.full((n, 1), t2_v, dtype=dtype, device=self.device)
                    s = self.model(xs, t1_t, t2_t)  # [n, 2]
                    dy = -(s[:, 0] * dt1_v + s[:, 1] * dt2_v)  # [n]
                    dy = torch.nan_to_num(dy, nan=0.0, posinf=1e6, neginf=-1e6)
                    integrand_rows.append(dy)
                vals = torch.stack(integrand_rows)  # [integration_steps, n]
            dt = (1.0 - 2.0 * self.eps) / (self.integration_steps - 1)
            return torch.trapezoid(vals, dx=dt, dim=0).cpu()
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
