"""
Direct ELDR Estimation using Stochastic Interpolants (Version 4)

Ports the spatial velocity denoiser approach to ELDR estimation using
separate networks for velocity (b) and denoiser (eta), with the denoiser-only
parameterization.

Key idea: Instead of sampling z ~ N(0,I), we compute z_equiv deterministically
from x ~ p_* using: z_equiv = (x - I_t) / gamma_t

This allows us to estimate ELDR = E_{p_*}[log(p_0(x)/p_1(x))] directly.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.eldr_estimation.base import ELDREstimator


class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron for estimating vector fields.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [batch, 1]
            x: Spatial values [batch, dim]
        Returns:
            Vector field [batch, output_dim]
        """
        tx = torch.cat([t, x], dim=-1)
        return self.net(tx)


def compute_divergence(output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes exact divergence (trace of Jacobian).

    Args:
        output: Vector field [batch, dim]
        x: Input points [batch, dim] (must have requires_grad=True)

    Returns:
        Divergence estimates [batch]
    """
    batch_size, dim = x.shape
    divergence = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

    for i in range(dim):
        grad_i = torch.autograd.grad(
            outputs=output[:, i].sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
        )[0]
        divergence = divergence + grad_i[:, i]

    return divergence


class DirectELDREstimator4(ELDREstimator):
    """
    ELDR estimator using spatial velocity-denoiser approach with z replacement.

    Instead of sampling z ~ N(0,I), we compute z_equiv = (x - I_t) / gamma_t
    where x comes from the base distribution p_*.

    The time score is: dt_log_rho = -div(b) + b.eta/gamma

    ELDR is estimated by integrating the negative time score over t.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 2e-3,
        k: float = 0.5,
        eps: float = 0.01,
        integration_steps: int = 10000,
        verbose: bool = False,
        log_every: int = 100,
        device: Optional[str] = None,
    ):
        """
        Args:
            input_dim: Dimensionality of input samples
            hidden_dim: Hidden layer dimension for networks
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            k: Parameter for gamma(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t)))
            eps: Boundary epsilon (t in [eps, 1-eps])
            integration_steps: Number of points for numerical integration
            verbose: Print training progress
            log_every: Print every N epochs when verbose
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.k = k
        self.eps = eps
        self.integration_steps = integration_steps
        self.verbose = verbose
        self.log_every = log_every

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.net_b = None
        self.net_eta = None

    def init_model(self) -> None:
        """Initialize two completely separate networks for b and eta."""
        self.net_b = MLP(self.input_dim, self.hidden_dim, output_dim=self.input_dim).to(self.device)
        self.net_eta = MLP(self.input_dim, self.hidden_dim, output_dim=self.input_dim).to(self.device)

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
    ) -> torch.Tensor:
        """
        Compute partial_t log rho(t, x) using:
        dt_log_rho = -div(b) + b.eta/gamma

        Args:
            t: Time tensor [batch, 1]
            x: Data tensor [batch, dim]
            gamma_t: Value of gamma at time t (scalar or [batch, 1])

        Returns:
            time_score: [batch, 1] estimate of the time score
        """
        x = x.clone().requires_grad_(True)

        # Run separate forward passes
        b_pred = self.net_b(t, x)
        eta_pred = self.net_eta(t, x)

        # Exact divergence (trace of Jacobian)
        div_b = compute_divergence(b_pred, x)

        # Scalar terms
        b_dot_eta = (b_pred * eta_pred).sum(dim=-1, keepdim=True)

        return -div_b.view(-1, 1) + b_dot_eta / gamma_t

    def fit(
        self,
        samples_base: torch.Tensor,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
    ) -> None:
        """
        Train the velocity and denoiser networks.

        Key difference from standard spatial_velo: instead of sampling z ~ N(0,I),
        we compute z_equiv = (x - I_t) / gamma_t where x ~ p_*.

        Args:
            samples_base: Samples from the base distribution p_* [n_base, dim]
            samples_p0: Samples from p_0 [n_p0, dim]
            samples_p1: Samples from p_1 [n_p1, dim]
        """
        self.init_model()

        samples_base = samples_base.float().to(self.device)
        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)

        n_base = samples_base.shape[0]
        n_p0 = samples_p0.shape[0]
        n_p1 = samples_p1.shape[0]

        if self.verbose:
            print(f"[DirectELDREstimator4] Starting Sequential Training")
            print(f"[DirectELDREstimator4] gamma range: [{self.gamma(torch.tensor(self.eps)).item():.4f}, {self.gamma(torch.tensor(0.5)).item():.4f}]")

        # ==========================================
        # PHASE 1: Train Velocity Network (b)
        # ==========================================
        if self.verbose:
            print(f"\n[Phase 1/2] Training Velocity Network (b) for {self.n_epochs} epochs...")

        self.net_b.train()
        self.net_eta.eval()
        optimizer_b = optim.Adam(self.net_b.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        for epoch in range(self.n_epochs):
            # Sample batches from all three distributions
            base_idx = torch.randint(0, n_base, (self.batch_size,))
            p0_idx = torch.randint(0, n_p0, (self.batch_size,))
            p1_idx = torch.randint(0, n_p1, (self.batch_size,))

            x = samples_base[base_idx]  # from p_*
            x0 = samples_p0[p0_idx]     # from p0
            x1 = samples_p1[p1_idx]     # from p1

            # Sample time uniformly from [eps, 1-eps] per sample
            t = torch.rand(self.batch_size, device=self.device) * (1 - 2 * self.eps) + self.eps
            t_batch = t.unsqueeze(-1)  # [B, 1]

            # Compute gamma and gamma' for each t
            gamma_t = self.gamma(t).unsqueeze(-1)  # [B, 1]
            gamma_prime_t = self.dgamma_dt(t).unsqueeze(-1)  # [B, 1]

            # Interpolant mean I_t = (1-t)*x0 + t*x1
            I_t = (1 - t_batch) * x0 + t_batch * x1

            # Compute z_equiv = (x - I_t) / gamma_t (deterministic given x, x0, x1, t)
            z_equiv = (x - I_t) / gamma_t

            # Target velocity: (x1 - x0) + gamma' * z_equiv
            # Equivalently: (x1 - x0) + (gamma'/gamma) * (x - I_t)
            dtIt = x1 - x0  # derivative of I_t w.r.t. t
            target_v = dtIt + gamma_prime_t * z_equiv

            # Forward pass: network takes (t, x) NOT (t, x_t)
            b_pred = self.net_b(t_batch, x)

            # Velocity loss: 0.5*||b||^2 - target_v.b
            b_norm_sq = (b_pred ** 2).sum(dim=-1)
            target_dot_b = (target_v * b_pred).sum(dim=-1)
            loss_b = (0.5 * b_norm_sq - target_dot_b).mean()

            optimizer_b.zero_grad()
            loss_b.backward()
            optimizer_b.step()

            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"  [Epoch {epoch+1}] loss_b={loss_b.item():.4f}")

        # ==========================================
        # PHASE 2: Train Denoiser Network (eta)
        # ==========================================
        if self.verbose:
            print(f"\n[Phase 2/2] Training Denoiser Network (eta) for {self.n_epochs} epochs...")

        self.net_b.eval()
        self.net_eta.train()
        optimizer_eta = optim.Adam(self.net_eta.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        for epoch in range(self.n_epochs):
            # Sample batches from all three distributions
            base_idx = torch.randint(0, n_base, (self.batch_size,))
            p0_idx = torch.randint(0, n_p0, (self.batch_size,))
            p1_idx = torch.randint(0, n_p1, (self.batch_size,))

            x = samples_base[base_idx]  # from p_*
            x0 = samples_p0[p0_idx]     # from p0
            x1 = samples_p1[p1_idx]     # from p1

            # Sample time uniformly from [eps, 1-eps] per sample
            t = torch.rand(self.batch_size, device=self.device) * (1 - 2 * self.eps) + self.eps
            t_batch = t.unsqueeze(-1)  # [B, 1]

            # Compute gamma for each t
            gamma_t = self.gamma(t).unsqueeze(-1)  # [B, 1]

            # Interpolant mean I_t = (1-t)*x0 + t*x1
            I_t = (1 - t_batch) * x0 + t_batch * x1

            # Compute z_equiv = (x - I_t) / gamma_t (target for denoiser)
            z_equiv = (x - I_t) / gamma_t

            # Forward pass: network takes (t, x)
            eta_pred = self.net_eta(t_batch, x)

            # Denoiser loss: 0.5*||eta||^2 - z_equiv.eta
            eta_norm_sq = (eta_pred ** 2).sum(dim=-1)
            z_dot_eta = (z_equiv * eta_pred).sum(dim=-1)
            loss_eta = (0.5 * eta_norm_sq - z_dot_eta).mean()

            optimizer_eta.zero_grad()
            loss_eta.backward()
            optimizer_eta.step()

            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"  [Epoch {epoch+1}] loss_eta={loss_eta.item():.4f}")

        if self.verbose:
            print(f"[DirectELDREstimator4] Training complete")

        self.net_eta.eval()

    def _integrate_time_score(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Integrate the time score over t for each sample.

        Args:
            samples: Points at which to evaluate [n_samples, dim]

        Returns:
            Integral values [n_samples]
        """
        self.net_b.eval()
        self.net_eta.eval()

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

            time_score = self.compute_time_score(t_batch, samples, gamma_t)
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

    def estimate_eldr(
        self,
        samples_pstar: torch.Tensor,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
    ) -> float:
        """
        Estimate the Expected Log Density Ratio E_{p_*}[log(p_0(x)/p_1(x))].

        Args:
            samples_pstar: Samples from the base distribution p_*
            samples_p0: Samples from p_0
            samples_p1: Samples from p_1

        Returns:
            Scalar ELDR estimate
        """
        # Convert to tensors if numpy arrays
        if isinstance(samples_pstar, np.ndarray):
            samples_pstar = torch.from_numpy(samples_pstar).float()
        if isinstance(samples_p0, np.ndarray):
            samples_p0 = torch.from_numpy(samples_p0).float()
        if isinstance(samples_p1, np.ndarray):
            samples_p1 = torch.from_numpy(samples_p1).float()

        # Train the model
        self.fit(samples_pstar, samples_p0, samples_p1)

        # Compute time_score integral for each x in samples_pstar
        samples_pstar = samples_pstar.float().to(self.device)
        integrals = self._integrate_time_score(samples_pstar)

        # ELDR = -E[integral of time_score]
        eldr = -integrals.mean().item()

        return eldr


if __name__ == '__main__':
    from torch.distributions import MultivariateNormal, kl_divergence
    from experiments.utils.prescribed_kls import create_two_gaussians_kl

    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 1000
    KL_DISTANCE = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {DEVICE}")
    print(f"Dimension: {DIM}")
    print(f"KL distance: {KL_DISTANCE}")
    print()

    # === CREATE SYNTHETIC DATA ===
    gaussian_pair = create_two_gaussians_kl(DIM, KL_DISTANCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair['mu0'].to(DEVICE), gaussian_pair['Sigma0'].to(DEVICE)
    mu1, Sigma1 = gaussian_pair['mu1'].to(DEVICE), gaussian_pair['Sigma1'].to(DEVICE)
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

    # Use p0 as the base distribution p_*
    samples_base = p0.sample((NSAMPLES_TRAIN,))
    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))

    # === TRUE ELDR (= KL divergence when p_* = p0) ===
    true_eldr = kl_divergence(p0, p1).item()
    print(f"True ELDR (KL(p0||p1)): {true_eldr:.4f}")
    print()

    # === DIRECT4 ESTIMATOR ===
    print("=" * 50)
    print("DirectELDREstimator4")
    print("=" * 50)

    estimator = DirectELDREstimator4(
        input_dim=DIM,
        hidden_dim=256,
        n_epochs=2000,
        batch_size=512,
        lr=2e-3,
        k=0.5,
        eps=0.01,
        integration_steps=10001,
        verbose=True,
        log_every=200,
        device=DEVICE,
    )

    eldr_estimate = estimator.estimate_eldr(samples_base, samples_p0, samples_p1)

    print()
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"True ELDR:      {true_eldr:.4f}")
    print(f"Estimated ELDR: {eldr_estimate:.4f}")
    print(f"Absolute Error: {abs(eldr_estimate - true_eldr):.4f}")
    print(f"Relative Error: {abs(eldr_estimate - true_eldr) / true_eldr * 100:.2f}%")
