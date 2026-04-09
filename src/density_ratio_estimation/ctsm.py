from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from scipy import integrate

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D


class CTSM(DensityRatioEstimator):
    """
    Conditional Time Score Matching for density ratio estimation.

    Uses Schrodinger Bridge path with closed-form conditional time score target.
    Replaces TSM's Hyvarinen loss with direct MSE regression to avoid autograd
    w.r.t. t. Training: sample t, epsilon, construct SB path x_t, compute
    closed-form target and weight via _epsilon_target, then MSE(target - weight*pred).
    Inference: ODE integration from t=eps to t=1-eps.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1e-3,
        sigma: float = 1.0,
        eps: float = 1e-3,
        device: Optional[str] = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        """
        Initialize CTSM estimator.

        Store hyperparameters, set device, initialize model and optimizer to None.
        """
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.sigma = sigma
        self.eps = eps
        self.rtol = rtol
        self.atol = atol
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = None
        self.optimizer = None

    def init_model(self) -> None:
        """
        Initialize neural network and optimizer.

        Instantiate TimeScoreNetwork1D on device, create Adam optimizer.
        """
        self.model = TimeScoreNetwork1D(self.input_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )

    def _epsilon_target(
        self,
        epsilon: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute closed-form CTSM target and weight for Schrodinger Bridge path.

        SB path: x_t = (1-t)*x0 + t*x1 + \sigma*\sqrt{t(1-t)}*\epsilon

        All operations element-wise, outputs shape [B, 1]:
        - var = \sigma^2 * t * (1-t)
        - std = \sigma * \sqrt{t * (1-t)}
        - d_var = \sigma^2 * (1 - 2*t)
        - delta = x1 - x0  (shape [B, dim])
        - delta_sq = sum(delta^2, dim=-1, keepdim=True)
        - eps_sq = sum(\epsilon^2, dim=-1, keepdim=True)
        - delta_dot_eps = sum(delta * \epsilon, dim=-1, keepdim=True)
        - temp = \sqrt{2 * delta_sq + 1e-8}  (numerical stability)
        - lambda_t = var / temp
        - target = (d_var * (eps_sq - dim) / 2 + std * delta_dot_eps) / temp

        Returns: tuple (lambda_t, target) both of shape [B, 1].
        """
        # [B, 1]
        var = self.sigma**2 * t * (1 - t)
        std = self.sigma * torch.sqrt(t * (1 - t))
        d_var = self.sigma**2 * (1 - 2 * t)

        # [B, dim]
        delta = x1 - x0
        # [B, 1]
        delta_sq = torch.sum(delta**2, dim=-1, keepdim=True)
        eps_sq = torch.sum(epsilon**2, dim=-1, keepdim=True)
        delta_dot_eps = torch.sum(delta * epsilon, dim=-1, keepdim=True)

        dim = epsilon.shape[-1]
        # [B, 1]
        temp = torch.sqrt(2 * delta_sq + 1e-8)
        lambda_t = var / temp
        target = (d_var * (eps_sq - dim) / 2 + std * delta_dot_eps) / temp

        return lambda_t, target

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """
        Train CTSM on paired samples from p0 and p1.

        Initialize model, loop epochs: sample batch indices, extract x0 and x1,
        sample t and epsilon, construct SB path, compute target via _epsilon_target,
        forward pass, MSE loss, backprop.
        """
        self.init_model()
        self.model.train()

        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()
        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]

        samples_p0 = samples_p0.to(self.device)
        samples_p1 = samples_p1.to(self.device)

        for _ in range(self.n_epochs):
            # sample batch indices with replacement
            idx0 = torch.randint(0, n0, (self.batch_size,))
            idx1 = torch.randint(0, n1, (self.batch_size,))

            x0 = samples_p0[idx0]  # [B, dim]
            x1 = samples_p1[idx1]  # [B, dim]

            # sample t and epsilon
            t = torch.rand(self.batch_size, 1, device=self.device) * (1 - 2 * self.eps) + self.eps  # [B, 1]
            epsilon = torch.randn_like(x0)  # [B, dim]

            # construct SB path
            x_t = (1 - t) * x0 + t * x1 + self.sigma * torch.sqrt(t * (1 - t)) * epsilon  # [B, dim]

            # compute target
            lambda_t, target = self._epsilon_target(epsilon, x0, x1, t)  # [B, 1], [B, 1]

            # forward pass
            pred = self.model(x_t, t)  # [B, 1]

            # loss
            mse_loss = torch.mean((target - lambda_t * pred) ** 2)

            # backprop
            self.optimizer.zero_grad()
            mse_loss.backward()
            self.optimizer.step()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Estimate log density ratio via ODE integration.

        Verify model is initialized, set eval mode, integrate from t=eps to t=1-eps
        using learned score via ODE45. Return log ratios on xs device.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        xs = xs.to(self.device)
        n = xs.shape[0]

        def ode_func(t, y):
            """ODE function for scipy integration: return -score(x_t, t) as numpy."""
            t_tensor = torch.full((n, 1), t, device=self.device)
            with torch.no_grad():
                score = self.model(xs, t_tensor)  # [B, 1]
            return (-score).squeeze().cpu().numpy()

        solution = integrate.solve_ivp(
            ode_func,
            (self.eps, 1.0 - self.eps),
            np.zeros(n),
            method="RK45",
            rtol=self.rtol,
            atol=self.atol,
        )

        log_ratios = solution.y[:, -1]
        return torch.from_numpy(log_ratios)


if __name__ == "__main__":
    from torch.distributions import MultivariateNormal

    from experiments.utils.prescribed_kls import create_two_gaussians_kl

    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 10
    KL_DIVERGENCE = 5

    # === CREATE SYNTHETIC DATA ===
    gaussian_pair = create_two_gaussians_kl(DIM, KL_DIVERGENCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair["mu0"], gaussian_pair["Sigma0"]
    mu1, Sigma1 = gaussian_pair["mu1"], gaussian_pair["Sigma1"]
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))
    samples_test = p0.sample((NSAMPLES_TEST,))

    # === TRAIN AND EVALUATE ===
    ctsm = CTSM(DIM)
    ctsm.fit(samples_p0, samples_p1)

    est_ldrs = ctsm.predict_ldr(samples_test)
    true_ldrs = p0.log_prob(samples_test) - p1.log_prob(samples_test)
    mae = torch.mean(torch.abs(est_ldrs - true_ldrs))
    print(f"MAE: {mae}")
