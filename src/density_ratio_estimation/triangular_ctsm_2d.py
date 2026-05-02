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

import numpy as np
import torch
import torch.optim as optim
from scipy import integrate

from src.density_ratio_estimation.base import DensityRatioEstimator
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
      rtol, atol: scipy ODE solver tolerances.
      log_every: int, log per-head losses every N epochs (0 = disabled).
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
        rtol: float = 1e-6,
        atol: float = 1e-6,
        log_every: int = 100,
    ) -> None:
        super().__init__(input_dim)

        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.rtol = rtol
        self.atol = atol
        self.log_every = log_every

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.path = path if path is not None else Stacked2DCtsm(
            sigma=1.0, gamma_schedule="sqrt", eps=eps
        )
        self.curve = curve if curve is not None else Curve2D(path_height=1.0)

        # coverage check: training t_2 range must contain the inference curve's t_2 peak.
        # missing this allows silent extrapolation at predict_ldr.
        peak = float(self.curve.peak_t2())
        t2_max = float(getattr(self.path, "t2_max", 1.0 - self.eps))
        if peak > t2_max:
            raise ValueError(
                f"curve.peak_t2() = {peak} exceeds path.t2_max = {t2_max}; "
                f"the network would be queried at untrained t_2 values during predict_ldr. "
                f"Increase Stacked2DCtsm(t2_max=...) or reduce Curve2D(path_height=...)."
            )

        self.model = None
        self.optimizer = None

    def init_model(self) -> None:
        """Construct ScoreNetwork2D + Adam optimizer on self.device."""
        self.model = ScoreNetwork2D(self.input_dim, self.hidden_dim, n_hidden_layers=self.n_hidden_layers).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
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
              sample t1 ~ U(eps, 1-eps), t2 ~ U(eps, t2_max), epsilon ~ N(0, I).
              (x, target [B, 2], lambda_t [B, 2]) = self.path.sample_and_target(...).
              pred = self.model(x, t1, t2)  -> [B, 2].
              loss = mean((target - lambda_t * pred)^2)  over batch and over both heads.
              backward; step.
              every log_every epochs (if > 0): log per-head losses.
          - eval mode.
        """
        self.init_model()
        self.model.train()

        samples_p0 = samples_p0.to(self.device).float()
        samples_p1 = samples_p1.to(self.device).float()
        samples_pstar = samples_pstar.to(self.device).float()

        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        # restrict training t_2 range to overlap the inference curve.
        # paths without a t2_max attribute fall back to the standard upper bound.
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

            # sample t1 ~ U(eps, 1-eps) and t2 ~ U(eps, t2_max)
            t1 = (
                torch.rand(self.batch_size, 1, device=self.device)
                * (1.0 - 2.0 * self.eps)
                + self.eps
            )  # [B, 1]
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

            # forward
            pred = self.model(x, t1, t2)  # [B, 2]

            # mse loss; mean over batch and over both heads
            err = target - lambda_t * pred  # [B, 2]
            loss = (err ** 2).mean()

            # diagnostic: per-head losses
            if self.log_every > 0 and (epoch_idx + 1) % self.log_every == 0:
                with torch.no_grad():
                    loss_h1 = (err[:, 0] ** 2).mean().item()
                    loss_h2 = (err[:, 1] ** 2).mean().item()
                print(
                    f"epoch {epoch_idx + 1}: loss={loss.item():.4f} "
                    f"(head1={loss_h1:.4f}, head2={loss_h2:.4f})"
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.model.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """Predict log density ratios via line integral along self.curve.

        Procedure:
          - Verify trained.
          - Move xs to device.
          - Define ode_func(tau_scalar, y):
              tau_v = float(tau_scalar)
              (t1_v, t2_v) = (curve.t1(tau_v), curve.t2(tau_v))
              (dt1_v, dt2_v) = (curve.dt1(tau_v), curve.dt2(tau_v))
              allocate t1_t, t2_t = full((n, 1), <value>, dtype=model dtype, device=self.device).
              with no_grad: s = self.model(xs, t1_t, t2_t)  -> [n, 2]
              dy = -(s[:, 0] * dt1_v + s[:, 1] * dt2_v)   # mirror V2 sign convention
              dy = torch.nan_to_num(dy, nan=0.0, posinf=1e6, neginf=-1e6)
              return dy.cpu().numpy()
          - Integrate from tau=eps to 1-eps via solve_ivp RK45 with y0 = zeros(n).
          - Return solution.y[:, -1] as a CPU torch tensor [n].
        """
        if self.model is None:
            raise RuntimeError(
                "TriangularCTSM2D is not trained. Call fit() before predict_ldr()."
            )

        self.model.eval()
        xs = xs.to(self.device).float()
        n = xs.shape[0]

        # cache the model's parameter dtype/device for the ode_func tensor allocation
        dtype = next(self.model.parameters()).dtype
        device = self.device

        with torch.no_grad():

            def ode_func(tau_scalar, y):
                tau_v = float(tau_scalar)
                t1_v = float(self.curve.t1(tau_v))
                t2_v = float(self.curve.t2(tau_v))
                dt1_v = float(self.curve.dt1(tau_v))
                dt2_v = float(self.curve.dt2(tau_v))

                # build (n, 1) constant tensors with the model's dtype/device
                t1_t = torch.full((n, 1), t1_v, dtype=dtype, device=device)
                t2_t = torch.full((n, 1), t2_v, dtype=dtype, device=device)

                # forward
                s = self.model(xs, t1_t, t2_t)  # [n, 2]

                # line-integral integrand; mirror V2 sign convention
                dy = -(s[:, 0] * dt1_v + s[:, 1] * dt2_v)  # [n]
                dy = torch.nan_to_num(
                    dy, nan=0.0, posinf=1e6, neginf=-1e6
                )
                return dy.cpu().numpy()

            solution = integrate.solve_ivp(
                ode_func,
                (self.eps, 1.0 - self.eps),
                np.zeros(n),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
            )

            if not solution.success:
                raise RuntimeError(
                    f"solve_ivp failed: {solution.message}. "
                    f"Refusing to return solution.y[:, -1] which may be the last attempted "
                    f"step rather than the integrated value."
                )

            log_ratios = solution.y[:, -1]  # [n]

        return torch.from_numpy(log_ratios)


if __name__ == "__main__":
    from torch.distributions import MultivariateNormal

    from experiments.utils.prescribed_kls import create_two_gaussians_kl

    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 10
    KL_DIVERGENCE = 5.0

    # gaussian pair with controlled KL divergence
    gp = create_two_gaussians_kl(dim=DIM, k=KL_DIVERGENCE, beta=0.5)
    mu0, Sigma0 = gp["mu0"], gp["Sigma0"]
    mu1, Sigma1 = gp["mu1"], gp["Sigma1"]

    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

    # p_* (anchor): midpoint in mean and covariance
    mu_star = (mu0 + mu1) / 2.0
    Sigma_star = (Sigma0 + Sigma1) / 2.0
    pstar = MultivariateNormal(mu_star, covariance_matrix=Sigma_star)

    # sample
    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))
    samples_pstar = pstar.sample((NSAMPLES_TRAIN,))
    samples_test = p0.sample((NSAMPLES_TEST,))

    # train + evaluate
    estimator = TriangularCTSM2D(input_dim=DIM)
    estimator.fit(samples_p0, samples_p1, samples_pstar)

    est_ldrs = estimator.predict_ldr(samples_test)
    true_ldrs = p0.log_prob(samples_test) - p1.log_prob(samples_test)
    mae = torch.mean(torch.abs(est_ldrs - true_ldrs))

    print(f"MAE: {mae}")
