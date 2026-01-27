from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import integrate

from src.density_ratio_estimation.base import DensityRatioEstimator


class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron for estimating vector fields.
    Used in sequential training mode.
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


class SeparateNetworks(nn.Module):
    """
    sharing='none': Two completely separate networks with half width each.
    Output is [v, s] concatenated (2 * input_dim).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        half_hidden = hidden_dim // 2
        self.b_net = nn.Sequential(
            nn.Linear(input_dim + 1, half_hidden),
            nn.ELU(),
            nn.Linear(half_hidden, half_hidden),
            nn.ELU(),
            nn.Linear(half_hidden, half_hidden),
            nn.ELU(),
            nn.Linear(half_hidden, input_dim),
        )
        self.s_net = nn.Sequential(
            nn.Linear(input_dim + 1, half_hidden),
            nn.ELU(),
            nn.Linear(half_hidden, half_hidden),
            nn.ELU(),
            nn.Linear(half_hidden, half_hidden),
            nn.ELU(),
            nn.Linear(half_hidden, input_dim),
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
        v = self.b_net(tx)
        s = self.s_net(tx)
        return torch.cat([v, s], dim=-1)


class SharedBackboneNetwork(nn.Module):
    """
    sharing='embeddings': Shared backbone with separate output heads.
    Output is [v, s] concatenated (2 * input_dim).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.b_head = nn.Linear(hidden_dim, input_dim)
        self.s_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [batch, 1]
            x: Spatial values [batch, dim]
        Returns:
            [v, s] concatenated [batch, 2*dim]
        """
        tx = torch.cat([t, x], dim=-1)
        h = self.backbone(tx)
        v = self.b_head(h)
        s = self.s_head(h)
        return torch.cat([v, s], dim=-1)


class FullSharingNetwork(nn.Module):
    """
    sharing='full': Single network with combined output split as [v, s].
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


def compute_divergence(output: torch.Tensor, x: torch.Tensor, epsilon: torch.Tensor = None) -> torch.Tensor:
    """
    Computes exact divergence (trace of Jacobian).

    Args:
        output: Vector field [batch, dim]
        x: Input points [batch, dim] (must have requires_grad=True)
        epsilon: Ignored (kept for API compatibility)

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


class SpatialVeloScore(DensityRatioEstimator):
    """
    Score-based variant of the interpolant estimator.

    The interpolant is: x_t = (1-t)*x0 + t*x1 + gamma(t)*z where z ~ N(0,I)

    Score interpretation: The score network predicts the spatial score s = -z/gamma

    Supports two training modes:
    - 'sequential': Two-phase training (trains velocity first, then score) with separate MLPs
    - 'simultaneous': Joint training with shared/separate networks (original approach)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 2e-3,
        k: float = 0.5,
        n_t: int = 50,
        eps: float = 0.01,
        device: Optional[str] = None,
        integration_steps: int = 10000,
        integration_type: Literal['1', '2', '3'] = '1',
        verbose: bool = False,
        log_every: int = 100,
        training_mode: Literal['sequential', 'simultaneous'] = 'sequential',
        sharing: str = 'full',
        antithetic: bool = False,
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
        self.integration_type = integration_type
        self.verbose = verbose
        self.log_every = log_every
        self.training_mode = training_mode
        if sharing not in {'none', 'embeddings', 'full'}:
            raise ValueError(f"Unknown sharing mode: {sharing}. Expected 'none', 'embeddings', or 'full'.")
        self.sharing = sharing
        self.antithetic = antithetic
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # For simultaneous mode
        self.model = None
        self.optimizer = None

        # For sequential mode
        self.net_b = None
        self.net_s = None

    def init_model(self) -> None:
        if self.training_mode == 'sequential':
            # Initialize two completely separate networks with full hidden_dim
            self.net_b = MLP(self.input_dim, self.hidden_dim, output_dim=self.input_dim).to(self.device)
            self.net_s = MLP(self.input_dim, self.hidden_dim, output_dim=self.input_dim).to(self.device)
        else:
            # Simultaneous mode: use sharing networks
            if self.sharing == 'none':
                self.model = SeparateNetworks(self.input_dim, self.hidden_dim).to(self.device)
            elif self.sharing == 'embeddings':
                self.model = SharedBackboneNetwork(self.input_dim, self.hidden_dim).to(self.device)
            elif self.sharing == 'full':
                self.model = FullSharingNetwork(self.input_dim, self.hidden_dim).to(self.device)
            else:
                raise ValueError(f"Unknown sharing mode: {self.sharing}")
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

        if self.training_mode == 'sequential':
            b_pred = self.net_b(t, x)
            s_pred = self.net_s(t, x)
        else:
            outputs = self.model(t, x)
            b_pred, s_pred = torch.chunk(outputs, chunks=2, dim=1)

        # Exact divergence (trace of Jacobian)
        div_b = compute_divergence(b_pred, x)

        # Scalar terms
        b_dot_s = (b_pred * s_pred).sum(dim=-1, keepdim=True)

        # Assemble: time_score = -div(b) - b·s
        return -div_b.view(-1, 1) - b_dot_s

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        if self.training_mode == 'sequential':
            self._fit_sequential(samples_p0, samples_p1)
        else:
            self._fit_simultaneous(samples_p0, samples_p1)

    def _fit_sequential(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """Sequential two-phase training: velocity first, then score."""
        self.init_model()

        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()
        n_p0 = samples_p0.shape[0]
        n_p1 = samples_p1.shape[0]

        if self.verbose:
            print(f"[SpatialVeloScore] Starting Sequential Training with batch-based time sampling.")
            print(f"[SpatialVeloScore] gamma range: [{self.gamma(torch.tensor(self.eps)).item():.4f}, {self.gamma(torch.tensor(0.5)).item():.4f}]")

        # ==========================================
        # PHASE 1: Train Velocity Network (b)
        # ==========================================
        if self.verbose:
            print(f"\n[Phase 1/2] Training Velocity Network (b) for {self.n_epochs} epochs...")

        self.net_b.train()
        self.net_s.eval()
        optimizer_b = optim.Adam(self.net_b.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        for epoch in range(self.n_epochs):
            p0_idx = torch.randint(0, n_p0, (self.batch_size,))
            p1_idx = torch.randint(0, n_p1, (self.batch_size,))
            x0 = samples_p0[p0_idx].to(self.device)
            x1 = samples_p1[p1_idx].to(self.device)

            t = torch.rand(self.batch_size, device=self.device) * (1 - 2 * self.eps) + self.eps
            t_batch = t.unsqueeze(-1)

            z = torch.randn_like(x0)

            gamma_t = self.gamma(t).unsqueeze(-1)
            gamma_prime_t = self.dgamma_dt(t).unsqueeze(-1)

            I_t = (1 - t_batch) * x0 + t_batch * x1
            dtIt = x1 - x0

            if self.antithetic:
                x_t_plus = I_t + gamma_t * z
                x_t_minus = I_t - gamma_t * z

                b_plus = self.net_b(t_batch, x_t_plus)
                b_minus = self.net_b(t_batch, x_t_minus)

                b_norm_sq_plus = (b_plus ** 2).sum(dim=-1)
                b_norm_sq_minus = (b_minus ** 2).sum(dim=-1)
                target_dot_b_plus = ((dtIt + gamma_prime_t * z) * b_plus).sum(dim=-1)
                target_dot_b_minus = ((dtIt - gamma_prime_t * z) * b_minus).sum(dim=-1)
                loss_b = (0.25 * b_norm_sq_plus - 0.5 * target_dot_b_plus
                        + 0.25 * b_norm_sq_minus - 0.5 * target_dot_b_minus).mean()
            else:
                x_t = I_t + gamma_t * z
                b_pred = self.net_b(t_batch, x_t)

                b_norm_sq = (b_pred ** 2).sum(dim=-1)
                target_dot_b = ((dtIt + gamma_prime_t * z) * b_pred).sum(dim=-1)
                loss_b = (0.5 * b_norm_sq - target_dot_b).mean()

            optimizer_b.zero_grad()
            loss_b.backward()
            optimizer_b.step()

            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"  [Epoch {epoch+1}] loss_b={loss_b.item():.4f}")

        # ==========================================
        # PHASE 2: Train Score Network (s)
        # ==========================================
        if self.verbose:
            print(f"\n[Phase 2/2] Training Score Network (s) for {self.n_epochs} epochs...")

        self.net_b.eval()
        self.net_s.train()
        optimizer_s = optim.Adam(self.net_s.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        for epoch in range(self.n_epochs):
            p0_idx = torch.randint(0, n_p0, (self.batch_size,))
            p1_idx = torch.randint(0, n_p1, (self.batch_size,))
            x0 = samples_p0[p0_idx].to(self.device)
            x1 = samples_p1[p1_idx].to(self.device)

            # Sample time from [0, 1] for score training
            t = torch.rand(self.batch_size, device=self.device)
            t_batch = t.unsqueeze(-1)

            z = torch.randn_like(x0)

            gamma_t = self.gamma(t).unsqueeze(-1)

            x_t = (1 - t_batch) * x0 + t_batch * x1 + gamma_t * z

            s_pred = self.net_s(t_batch, x_t)

            # Score loss reformulated to avoid division by gamma:
            # s = -z/gamma => gamma*s = -z
            # Minimize: 0.5*||gamma*s||² + z·(gamma*s)
            # = 0.5*gamma²*||s||² + gamma*(z·s)
            gamma_s = gamma_t * s_pred
            gamma_s_norm_sq = (gamma_s ** 2).sum(dim=-1)
            z_dot_gamma_s = (z * gamma_s).sum(dim=-1)
            loss_s = (0.5 * gamma_s_norm_sq + z_dot_gamma_s).mean()

            optimizer_s.zero_grad()
            loss_s.backward()
            optimizer_s.step()

            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"  [Epoch {epoch+1}] loss_s={loss_s.item():.4f}")

        if self.verbose:
            print(f"[SpatialVeloScore] Training complete")

        self.net_s.eval()

    def _fit_simultaneous(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """Simultaneous joint training (original approach)."""
        self.init_model()
        self.model.train()

        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()
        n_p0 = samples_p0.shape[0]
        n_p1 = samples_p1.shape[0]

        if self.verbose:
            print(f"[SpatialVeloScore] Training with {self.n_epochs} epochs, batch-based time sampling")
            print(f"[SpatialVeloScore] gamma range: [{self.gamma(torch.tensor(self.eps)).item():.4f}, {self.gamma(torch.tensor(0.5)).item():.4f}]")

        for epoch in range(self.n_epochs):
            # Sample batches
            p0_idx = torch.randint(0, n_p0, (self.batch_size,))
            p1_idx = torch.randint(0, n_p1, (self.batch_size,))
            x0 = samples_p0[p0_idx].to(self.device)
            x1 = samples_p1[p1_idx].to(self.device)

            # Sample time uniformly from [eps, 1-eps] per sample
            t = torch.rand(self.batch_size, device=self.device) * (1 - 2 * self.eps) + self.eps
            t_batch = t.unsqueeze(-1)  # [B, 1]

            # Sample noise (independent per sample)
            z = torch.randn_like(x0)

            # Compute gamma and gamma' for each t
            gamma_t = self.gamma(t).unsqueeze(-1)  # [B, 1]
            gamma_prime_t = self.dgamma_dt(t).unsqueeze(-1)  # [B, 1]

            # Interpolant
            I_t = (1 - t_batch) * x0 + t_batch * x1
            dtIt = x1 - x0  # derivative of I_t w.r.t. t
            eta_t = gamma_t * z
            x_t_plus = I_t + eta_t

            # Forward pass
            outputs_plus = self.model(t_batch, x_t_plus)
            b_plus, s_plus = torch.chunk(outputs_plus, chunks=2, dim=1)

            if self.antithetic:
                # Antithetic sampling: compute x_t+ and x_t-
                x_t_minus = I_t - eta_t
                outputs_minus = self.model(t_batch, x_t_minus)
                b_minus, s_minus = torch.chunk(outputs_minus, chunks=2, dim=1)

                # Velocity loss with antithetic sampling
                # loss_b+ = 0.5*||b+||² - (dtIt + γ'z)·b+
                # loss_b- = 0.5*||b-||² - (dtIt - γ'z)·b-
                b_norm_sq_plus = (b_plus ** 2).sum(dim=-1)
                b_norm_sq_minus = (b_minus ** 2).sum(dim=-1)
                target_dot_b_plus = ((dtIt + gamma_prime_t * z) * b_plus).sum(dim=-1)
                target_dot_b_minus = ((dtIt - gamma_prime_t * z) * b_minus).sum(dim=-1)
                loss_b = (0.25 * b_norm_sq_plus - 0.5 * target_dot_b_plus
                        + 0.25 * b_norm_sq_minus - 0.5 * target_dot_b_minus).mean()

                # Score loss with antithetic sampling (reformulated to avoid 1/gamma)
                gamma_s_plus = gamma_t * s_plus
                gamma_s_minus = gamma_t * s_minus
                gamma_s_norm_sq_plus = (gamma_s_plus ** 2).sum(dim=-1)
                gamma_s_norm_sq_minus = (gamma_s_minus ** 2).sum(dim=-1)
                loss_s = (0.25 * gamma_s_norm_sq_plus + 0.25 * gamma_s_norm_sq_minus
                        + 0.5 * (z * (gamma_s_plus - gamma_s_minus)).sum(dim=-1)).mean()
            else:
                # Standard (non-antithetic) training
                b_norm_sq = (b_plus ** 2).sum(dim=-1)
                target_dot_b = ((dtIt + gamma_prime_t * z) * b_plus).sum(dim=-1)
                loss_b = (0.5 * b_norm_sq - target_dot_b).mean()

                # Reformulated to avoid 1/gamma
                gamma_s = gamma_t * s_plus
                gamma_s_norm_sq = (gamma_s ** 2).sum(dim=-1)
                z_dot_gamma_s = (z * gamma_s).sum(dim=-1)
                loss_s = (0.5 * gamma_s_norm_sq + z_dot_gamma_s).mean()

            total_loss = loss_b + loss_s

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"[Epoch {epoch+1}/{self.n_epochs}] total_loss={total_loss.item():.4f}, loss_b={loss_b.item():.4f}, loss_s={loss_s.item():.4f}")

        if self.verbose:
            print(f"[SpatialVeloScore] Training complete")

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        if self.training_mode == 'sequential':
            if self.net_b is None or self.net_s is None:
                raise RuntimeError("SpatialVeloScore model is not trained. Call fit() before predict_ldr().")
            self.net_b.eval()
            self.net_s.eval()
        else:
            if self.model is None:
                raise RuntimeError("SpatialVeloScore model is not trained. Call fit() before predict_ldr().")
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

        if self.integration_type == '3':
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
            out = -torch.from_numpy(integral)
        elif self.integration_type == '1':
            # Mean approximation
            out = -time_scores.mean(dim=0).cpu()
        elif self.integration_type == '2':
            # Trapezoidal rule
            out = -torch.trapz(time_scores, t_vals, dim=0).cpu()
        else:
            raise ValueError(f"Unknown integration_type: {self.integration_type}")

        return out


if __name__ == '__main__':
    from torch.distributions import MultivariateNormal
    from experiments.utils.prescribed_kls import create_two_gaussians_kl
    from src.density_ratio_estimation.bdre import BDRE
    from src.models.binary_classification import make_binary_classifier

    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 100
    KL_DISTANCE = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # === CREATE SYNTHETIC DATA ===
    gaussian_pair = create_two_gaussians_kl(DIM, KL_DISTANCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair['mu0'].to(DEVICE), gaussian_pair['Sigma0'].to(DEVICE)
    mu1, Sigma1 = gaussian_pair['mu1'].to(DEVICE), gaussian_pair['Sigma1'].to(DEVICE)
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)
    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))
    samples_test = p0.sample((NSAMPLES_TEST,))

    # === TRUE LDR ===
    true_ldrs = p0.log_prob(samples_test) - p1.log_prob(samples_test)
    print(f"True LDR range: [{true_ldrs.min().item():.4f}, {true_ldrs.max().item():.4f}]")
    print(f"True LDR mean: {true_ldrs.mean().item():.4f}")
    print()

    # === BDRE BASELINE ===
    print("=" * 50)
    print("BDRE (Baseline)")
    print("=" * 50)
    classifier = make_binary_classifier("default", input_dim=DIM)
    bdre = BDRE(classifier, device=DEVICE)
    bdre.fit(samples_p0.to(DEVICE), samples_p1.to(DEVICE))
    bdre_ldrs = bdre.predict_ldr(samples_test.to(DEVICE))
    bdre_mae = torch.mean(torch.abs(bdre_ldrs.cpu() - true_ldrs.cpu()))
    print(f"BDRE MAE: {bdre_mae.item():.4f}")
    print(f"BDRE LDR range: [{bdre_ldrs.min().item():.4f}, {bdre_ldrs.max().item():.4f}]")
    print()

    # === SPATIAL VELO SCORE (Sequential Mode - Default) ===
    print("=" * 50)
    print("SpatialVeloScore (Sequential Training - Default)")
    print("=" * 50)
    estimator = SpatialVeloScore(
        DIM,
        n_epochs=1000,
        verbose=True,
        log_every=200,
        device=DEVICE,
        training_mode='sequential',
    )
    estimator.fit(samples_p0, samples_p1)

    print("\nEvaluating...")
    est_ldrs = estimator.predict_ldr(samples_test)
    sequential_mae = torch.mean(torch.abs(est_ldrs.cpu() - true_ldrs.cpu()))
    print(f"SpatialVeloScore (sequential) MAE: {sequential_mae.item():.4f}")
    print(f"SpatialVeloScore LDR range: [{est_ldrs.min().item():.4f}, {est_ldrs.max().item():.4f}]")
    print()

    # === SPATIAL VELO SCORE (Simultaneous Mode - Backward Compat) ===
    sharing_modes = ['none', 'embeddings', 'full']
    maes = {}

    for sharing in sharing_modes:
        print("=" * 50)
        print(f"SpatialVeloScore (simultaneous, sharing='{sharing}')")
        print("=" * 50)
        estimator = SpatialVeloScore(
            DIM,
            n_epochs=2000,
            verbose=True,
            log_every=400,
            device=DEVICE,
            training_mode='simultaneous',
            sharing=sharing,
            # Use old defaults for simultaneous mode comparison
            eps=0.1,
            lr=1e-3,
            integration_steps=100,
            integration_type='3',
        )
        estimator.fit(samples_p0, samples_p1)

        print("\nEvaluating...")
        est_ldrs = estimator.predict_ldr(samples_test)
        mae = torch.mean(torch.abs(est_ldrs.cpu() - true_ldrs.cpu()))
        maes[sharing] = mae.item()
        print(f"SpatialVeloScore (simultaneous, sharing='{sharing}') MAE: {mae.item():.4f}")
        print(f"SpatialVeloScore LDR range: [{est_ldrs.min().item():.4f}, {est_ldrs.max().item():.4f}]")
        print()

    # === COMPARISON ===
    print("=" * 50)
    print("Comparison")
    print("=" * 50)
    print(f"BDRE MAE:                                    {bdre_mae.item():.4f}")
    print(f"SpatialVeloScore (sequential):              {sequential_mae.item():.4f}")
    for sharing in sharing_modes:
        print(f"SpatialVeloScore (simultaneous, '{sharing}'): {maes[sharing]:.4f}")
