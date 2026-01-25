from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import integrate

from src.density_ratio_estimation.base import DensityRatioEstimator


class VelocityScoreNetwork(nn.Module):
    """
    Network that outputs both velocity field v and spatial score s.
    Output is [v, s] concatenated (2 * input_dim).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * input_dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [batch, 1]
            x: Spatial values [batch, dim]
        Returns:
            [v, s] concatenated [batch, 2*dim]
        """
        tx = torch.cat([t, x], dim=-1)
        return self.net(tx)


def compute_divergence(output: torch.Tensor, x: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    """
    Hutchinson's trace estimator for divergence.

    Args:
        output: Vector field [batch, dim]
        x: Input points [batch, dim] (must have requires_grad=True)
        epsilon: Random probe vectors [batch, dim]

    Returns:
        Divergence estimates [batch]
    """
    grad_outputs = torch.autograd.grad(
        outputs=(output * epsilon).sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
    )[0]
    return (grad_outputs * epsilon).sum(dim=-1)


class InterpolantScore(DensityRatioEstimator):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1e-3,
        k: float = 0.5,
        n_t: int = 50,
        eps: float = 0.1,
        device: Optional[str] = None,
        integration_steps: int = 100,
    ):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.k = k
        self.n_t = n_t
        self.eps = eps
        self.integration_steps = integration_steps
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = None
        self.optimizer = None

    def init_model(self) -> None:
        self.model = VelocityScoreNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """gamma(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t)))"""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        return (1 - torch.exp(-self.k * t)) * (1 - torch.exp(-self.k * (1 - t)))

    def dgamma_dt(self, t: torch.Tensor) -> torch.Tensor:
        """gamma'(t) = k*exp(-k*t)*(1-exp(-k*(1-t))) - k*exp(-k*(1-t))*(1-exp(-k*t))"""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        exp_kt = torch.exp(-self.k * t)
        exp_k1t = torch.exp(-self.k * (1 - t))
        return self.k * exp_kt * (1 - exp_k1t) - self.k * exp_k1t * (1 - exp_kt)

    def compute_time_score(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        gamma_t: torch.Tensor,
        gamma_dot_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute partial_t log rho(t, x).

        Args:
            t: Time tensor [batch, 1]
            x: Data tensor [batch, dim]
            gamma_t: Value of gamma at time t [batch, 1] or scalar
            gamma_dot_t: Time derivative of gamma at time t [batch, 1] or scalar

        Returns:
            time_score: [batch, 1] estimate of the time score
        """
        x = x.clone().requires_grad_(True)

        outputs = self.model(t, x)
        v_pred, s_pred = torch.chunk(outputs, chunks=2, dim=1)

        # Divergences via Hutchinson estimator
        epsilon = torch.randn_like(x)
        div_v = compute_divergence(v_pred, x, epsilon)
        div_s = compute_divergence(s_pred, x, epsilon)

        # Scalar terms
        v_dot_s = (v_pred * s_pred).sum(dim=-1, keepdim=True)
        s_norm_sq = (s_pred ** 2).sum(dim=-1, keepdim=True)

        # Assemble: time_score = -div(v) - v·s + gamma_dot*gamma*(div(s) + ||s||²)
        term_signal = -div_v.view(-1, 1) - v_dot_s
        c_t = gamma_dot_t * gamma_t
        term_noise = c_t * (div_s.view(-1, 1) + s_norm_sq)

        return term_signal + term_noise

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        self.init_model()
        self.model.train()

        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()
        n_p0 = samples_p0.shape[0]
        n_p1 = samples_p1.shape[0]

        # Pre-sample time grid
        t_grid = torch.linspace(self.eps, 1 - self.eps, self.n_t, device=self.device)

        for epoch in range(self.n_epochs):
            # Sample batches
            p0_idx = torch.randint(0, n_p0, (self.batch_size,))
            p1_idx = torch.randint(0, n_p1, (self.batch_size,))
            x0 = samples_p0[p0_idx].to(self.device)
            x1 = samples_p1[p1_idx].to(self.device)

            # Sample noise (same for all t in this batch for variance reduction)
            z = torch.randn_like(x0)

            total_loss = 0.0
            for t_val in t_grid:
                t_batch = torch.full((self.batch_size, 1), t_val.item(), device=self.device)
                gamma_t = self.gamma(t_val)

                # Construct interpolant: x_t = (1-t)*x0 + t*x1 + gamma(t)*z
                x_t = (1 - t_val) * x0 + t_val * x1 + gamma_t * z

                # Forward pass
                outputs = self.model(t_batch, x_t)
                v_pred, s_pred = torch.chunk(outputs, chunks=2, dim=1)

                # Velocity loss: 0.5*||v||² - (x1 - x0)·v
                v_norm_sq = (v_pred ** 2).sum(dim=-1)
                target_dot_v = ((x1 - x0) * v_pred).sum(dim=-1)
                loss_v = (0.5 * v_norm_sq - target_dot_v).mean()

                # Score loss: 0.5*||s||² + (1/gamma)*z·s
                s_norm_sq = (s_pred ** 2).sum(dim=-1)
                z_dot_s = (z * s_pred).sum(dim=-1)
                loss_s = (0.5 * s_norm_sq + (1.0 / gamma_t) * z_dot_s).mean()

                total_loss = total_loss + loss_v + loss_s

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("InterpolantScore model is not trained. Call fit() before predict_ldr().")

        self.model.eval()
        samples = xs.float().to(self.device)
        n_samples = samples.shape[0]

        # Integration grid
        n_points = self.integration_steps
        if n_points % 2 == 0:
            n_points += 1
        t_vals = torch.linspace(self.eps, 1 - self.eps, n_points, device=self.device)

        # Compute time scores at each t for all samples
        time_scores = []
        for t_val in t_vals:
            t_batch = torch.full((n_samples, 1), t_val.item(), device=self.device)
            gamma_t = self.gamma(t_val)
            gamma_dot_t = self.dgamma_dt(t_val)

            time_score = self.compute_time_score(t_batch, samples, gamma_t, gamma_dot_t)
            time_scores.append(time_score.detach())

        time_scores = torch.stack(time_scores, dim=0)  # [n_points, n_samples, 1]
        time_scores = time_scores.squeeze(-1)  # [n_points, n_samples]

        # Simpson's rule integration
        t_np = t_vals.cpu().numpy()
        h = (t_np[-1] - t_np[0]) / (n_points - 1)

        integrand = time_scores.cpu().numpy()  # [n_points, n_samples]
        integral = integrand[0] + integrand[-1]
        for i in range(1, n_points - 1):
            if i % 2 == 0:
                integral += 2 * integrand[i]
            else:
                integral += 4 * integrand[i]
        integral *= h / 3

        return torch.from_numpy(integral)


if __name__ == '__main__':
    from torch.distributions import MultivariateNormal
    from experiments.utils.two_gaussians_kl import create_two_gaussians_kl

    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 10
    KL_DISTANCE = 5

    # === CREATE SYNTHETIC DATA ===
    gaussian_pair = create_two_gaussians_kl(DIM, KL_DISTANCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
    mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)
    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))
    samples_pstar1 = p0.sample((NSAMPLES_TEST,))

    # === DENSITY RATIO ESTIMATION ===
    estimator = InterpolantScore(DIM, n_epochs=2000)
    estimator.fit(samples_p0, samples_p1)

    # === EVALUATION ===
    est_ldrs = estimator.predict_ldr(samples_pstar1)
    true_ldrs = p0.log_prob(samples_pstar1) - p1.log_prob(samples_pstar1)
    mae = torch.mean(torch.abs(est_ldrs - true_ldrs))
    print(f'MAE: {mae}')
