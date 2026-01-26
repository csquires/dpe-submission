from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import integrate

from src.density_ratio_estimation.base import DensityRatioEstimator


class SeparateNetworks(nn.Module):
    """
    sharing='none': Two completely separate networks with half width each.
    Output is [v, d] concatenated (2 * input_dim).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        half_hidden = hidden_dim // 2
        self.v_net = nn.Sequential(
            nn.Linear(input_dim + 1, half_hidden),
            nn.ELU(),
            nn.Linear(half_hidden, half_hidden),
            nn.ELU(),
            nn.Linear(half_hidden, half_hidden),
            nn.ELU(),
            nn.Linear(half_hidden, input_dim),
        )
        self.d_net = nn.Sequential(
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
            [v, d] concatenated [batch, 2*dim]
        """
        tx = torch.cat([t, x], dim=-1)
        v = self.v_net(tx)
        d = self.d_net(tx)
        return torch.cat([v, d], dim=-1)


class SharedBackboneNetwork(nn.Module):
    """
    sharing='embeddings': Shared backbone with separate output heads.
    Output is [v, d] concatenated (2 * input_dim).
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
        self.v_head = nn.Linear(hidden_dim, input_dim)
        self.d_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [batch, 1]
            x: Spatial values [batch, dim]
        Returns:
            [v, d] concatenated [batch, 2*dim]
        """
        tx = torch.cat([t, x], dim=-1)
        h = self.backbone(tx)
        v = self.v_head(h)
        d = self.d_head(h)
        return torch.cat([v, d], dim=-1)


class FullSharingNetwork(nn.Module):
    """
    sharing='full': Single network with combined output split as [v, d].
    Output is [v, d] concatenated (2 * input_dim).
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
            [v, d] concatenated [batch, 2*dim]
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


class SpatialVeloDenoiser(DensityRatioEstimator):
    """
    Denoiser-based variant of the interpolant estimator.

    The interpolant is: x_t = (1-t)*x0 + t*x1 + gamma(t)*z where z ~ N(0,I)

    Denoiser interpretation: The denoiser predicts the normalized noise: d ≈ z

    Relationship to score: s = -d / gamma, so d = -gamma * s
    """
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
        verbose: bool = False,
        log_every: int = 100,
        sharing: str = 'full',
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
        self.verbose = verbose
        self.log_every = log_every
        if sharing not in {'none', 'embeddings', 'full'}:
            raise ValueError(f"Unknown sharing mode: {sharing}. Expected 'none', 'embeddings', or 'full'.")
        self.sharing = sharing
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = None
        self.optimizer = None

    def init_model(self) -> None:
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
        Compute partial_t log rho(t, x) using denoiser parameterization.

        The relationship s = -d/gamma is used to transform the time score formula.

        Original formula (in terms of score s):
            time_score = -div(v) - v·s + gamma_dot*gamma*(div(s) + ||s||²)

        Transformed formula (in terms of denoiser d):
            term_signal = -div(v) + v·d/gamma
            term_noise = -gamma_dot*div(d) + gamma_dot*||d||²/gamma

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
        v_pred, d_pred = torch.chunk(outputs, chunks=2, dim=1)

        # Divergences via Hutchinson estimator
        epsilon = torch.randn_like(x)
        div_v = compute_divergence(v_pred, x, epsilon)
        div_d = compute_divergence(d_pred, x, epsilon)

        # Scalar terms
        v_dot_d = (v_pred * d_pred).sum(dim=-1, keepdim=True)
        d_norm_sq = (d_pred ** 2).sum(dim=-1, keepdim=True)

        # Assemble using transformed formula:
        # term_signal = -div(v) + v·d/gamma
        # term_noise = -gamma_dot*div(d) + gamma_dot*||d||²/gamma
        term_signal = -div_v.view(-1, 1) + v_dot_d / gamma_t
        term_noise = -gamma_dot_t * div_d.view(-1, 1) + (gamma_dot_t / gamma_t) * d_norm_sq

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

        if self.verbose:
            print(f"[SpatialVeloDenoiser] Training with {self.n_epochs} epochs, {self.n_t} time points")
            print(f"[SpatialVeloDenoiser] gamma range: [{self.gamma(torch.tensor(self.eps)).item():.4f}, {self.gamma(torch.tensor(0.5)).item():.4f}]")

        for epoch in range(self.n_epochs):
            # Sample batches
            p0_idx = torch.randint(0, n_p0, (self.batch_size,))
            p1_idx = torch.randint(0, n_p1, (self.batch_size,))
            x0 = samples_p0[p0_idx].to(self.device)
            x1 = samples_p1[p1_idx].to(self.device)

            # Sample noise (same for all t in this batch for variance reduction)
            z = torch.randn_like(x0)

            total_loss = 0.0
            total_loss_v = 0.0
            total_loss_d = 0.0
            for t_val in t_grid:
                t_batch = torch.full((self.batch_size, 1), t_val.item(), device=self.device)
                gamma_t = self.gamma(t_val)

                # Construct interpolant and forward pass
                x_t = (1 - t_val) * x0 + t_val * x1 + gamma_t * z
                outputs = self.model(t_batch, x_t)
                v_pred, d_pred = torch.chunk(outputs, chunks=2, dim=1)

                # Velocity loss: 0.5*||v||² - (x1 - x0)·v
                v_norm_sq = (v_pred ** 2).sum(dim=-1)
                target_dot_v = ((x1 - x0) * v_pred).sum(dim=-1)
                loss_v = (0.5 * v_norm_sq - target_dot_v).mean()

                # Denoiser loss: 0.5*||d||² - z·d
                # (MSE: ||d-z||²/2 = 0.5||d||² - z·d + const)
                d_norm_sq = (d_pred ** 2).sum(dim=-1)
                z_dot_d = (z * d_pred).sum(dim=-1)
                loss_d = (0.5 * d_norm_sq - z_dot_d).mean()

                total_loss = total_loss + loss_v + loss_d
                total_loss_v = total_loss_v + loss_v.item()
                total_loss_d = total_loss_d + loss_d.item()

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"[Epoch {epoch+1}/{self.n_epochs}] total_loss={total_loss.item():.4f}, loss_v={total_loss_v:.4f}, loss_d={total_loss_d:.4f}")

        if self.verbose:
            print(f"[SpatialVeloDenoiser] Training complete")

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("SpatialVeloDenoiser model is not trained. Call fit() before predict_ldr().")

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

    # === SPATIAL VELO DENOISER (all sharing modes) ===
    sharing_modes = ['none', 'embeddings', 'full']
    maes = {}

    for sharing in sharing_modes:
        print("=" * 50)
        print(f"SpatialVeloDenoiser (sharing='{sharing}')")
        print("=" * 50)
        estimator = SpatialVeloDenoiser(
            DIM,
            n_epochs=2000,
            verbose=True,
            log_every=200,
            device=DEVICE,
            sharing=sharing,
        )
        estimator.fit(samples_p0, samples_p1)

        print("\nEvaluating...")
        est_ldrs = estimator.predict_ldr(samples_test)
        mae = torch.mean(torch.abs(est_ldrs.cpu() - true_ldrs.cpu()))
        maes[sharing] = mae.item()
        print(f"SpatialVeloDenoiser (sharing='{sharing}') MAE: {mae.item():.4f}")
        print(f"SpatialVeloDenoiser LDR range: [{est_ldrs.min().item():.4f}, {est_ldrs.max().item():.4f}]")
        print()

    # === COMPARISON ===
    print("=" * 50)
    print("Comparison")
    print("=" * 50)
    print(f"BDRE MAE:                                 {bdre_mae.item():.4f}")
    for sharing in sharing_modes:
        print(f"SpatialVeloDenoiser (sharing='{sharing}'): {maes[sharing]:.4f}")
