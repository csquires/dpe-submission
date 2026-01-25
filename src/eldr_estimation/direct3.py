"""
Direct ELDR Estimation using Stochastic Interpolants (Version 3)

Estimates E_{p_*}[log(p_0(x)/p_1(x))] by learning the expected time-derivative
of log p_t along a stochastic interpolant path, then integrating over [0, 1].

Key differences from direct2.py:
- New estimand: d^T * mu_t' * gamma + ||d||^2 where d = x - mu_t
- Uniform sampling on [eps, 1-eps] instead of importance sampling
- Loss weighting: 1 / (g(t) + eps) instead of gamma(t) / f(t)
- Integration: v(t) * NN(t) / gamma^2 (extra gamma division)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from src.eldr_estimation.base import ELDREstimator


class FourierTimeEmbedding(nn.Module):
    """
    Gaussian Fourier feature embedding for time values.

    Maps scalar time t to high-dimensional representation using random Fourier features:
        [sin(2π * t * B), cos(2π * t * B)]

    where B is a fixed (non-learnable) random frequency matrix.
    """

    def __init__(self, mapping_size: int = 64, scale: float = 10.0):
        """
        Args:
            mapping_size: Number of random frequencies (output dimension = 2 * mapping_size)
            scale: Scale factor for the random frequencies (controls frequency range)
        """
        super().__init__()
        self.B = nn.Parameter(
            torch.randn(1, mapping_size) * scale,
            requires_grad=False
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature mapping to time values.

        Args:
            t: Time values of shape [batch, 1]

        Returns:
            Fourier features of shape [batch, 2 * mapping_size]
        """
        x_proj = 2 * np.pi * t @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class NoiserNetwork(nn.Module):
    """
    Neural network that maps time t to the expected noiser eta_*(t).

    Uses Gaussian Fourier feature embeddings for time representation.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout_p: float = 0.1,
        time_embed_size: int = 64,
        time_embed_scale: float = 10.0,
    ):
        """
        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout_p: Dropout probability
            time_embed_size: Size of Fourier embedding (output will be 2*time_embed_size)
            time_embed_scale: Scale parameter for random Fourier frequencies
        """
        super().__init__()

        self.time_embed = FourierTimeEmbedding(
            mapping_size=time_embed_size,
            scale=time_embed_scale
        )

        layers = []
        layers.append(nn.Linear(2 * time_embed_size, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values of shape [batch] or [batch, 1]

        Returns:
            Noiser estimates of shape [batch]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t_embed = self.time_embed(t)
        return self.net(t_embed).squeeze(-1)

    def _reset_parameters(self) -> None:
        """Reset model parameters to random initialization."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class DirectELDREstimator3(ELDREstimator):
    def __init__(
        self,
        input_dim: int,
        # Network hyperparameters
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout_p: float = 0.1,
        time_embed_size: int = 64,
        time_embed_scale: float = 10.0,
        # Interpolant hyperparameters
        k: float = 0.5,
        eps: float = 0.1,
        # Training hyperparameters
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 1000,
        batch_size: int = 256,
        # Integration hyperparameters
        integration_steps: int = 100,
        # Convergence hyperparameters
        convergence_threshold: float = 1e-4,
        patience: int = 200,
        # Misc
        verbose: bool = False,
        device: Optional[str] = None,
    ):
        """
        Args:
            input_dim: Dimensionality of input samples
            hidden_dim: Hidden layer dimension for noiser network
            num_layers: Number of layers in noiser network
            dropout_p: Dropout probability
            time_embed_size: Size of Fourier time embedding (output will be 2*time_embed_size)
            time_embed_scale: Scale parameter for random Fourier frequencies
            k: Parameter for g(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t)))
            eps: Boundary epsilon (t in [eps, 1-eps])
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            num_epochs: Maximum training epochs
            batch_size: Batch size for training
            integration_steps: Number of points for numerical integration
            convergence_threshold: Stop if loss change < threshold for patience steps
            patience: Number of steps to wait for convergence
            verbose: Print training progress
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        super().__init__(input_dim)

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device

        self.k = k
        self.eps = eps

        self.noiser_network = NoiserNetwork(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_p=dropout_p,
            time_embed_size=time_embed_size,
            time_embed_scale=time_embed_scale,
        ).to(self.device)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.integration_steps = integration_steps
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.verbose = verbose

        # Sample statistics for variance computation
        self._mean_x = None
        self._E = None  # Var(x) summed across dims

    def _sample_t_uniform(self, batch_size: int) -> torch.Tensor:
        """
        Sample t uniformly on [eps, 1-eps].

        Args:
            batch_size: Number of samples to generate

        Returns:
            Tensor of t values of shape [batch_size]
        """
        t_samples = torch.rand(batch_size, device=self.device) * (1 - 2 * self.eps) + self.eps
        return t_samples

    def _compute_statistics(
        self,
        samples_base: torch.Tensor,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
    ) -> None:
        """
        Compute sample statistics needed for variance computation.

        Stores:
        - _mean_x: Mean of base distribution samples
        - _E: Variance of base distribution samples (summed across dims)
        """
        self._mean_x = samples_base.mean(dim=0)
        self._E = samples_base.var(dim=0).sum()

    def mu(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """mu(t) = (1-t)*x0 + t*x1"""
        t_exp = t.unsqueeze(-1)
        return (1 - t_exp) * x0 + t_exp * x1

    def dmudt(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """mu'(t) = x1 - x0 (constant in t)"""
        return x1 - x0

    def g(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t)))"""
        return (1 - torch.exp(-self.k * t)) * (1 - torch.exp(-self.k * (1 - t)))

    def dgdt(self, t: torch.Tensor) -> torch.Tensor:
        """
        g'(t) = k*exp(-k*t)*(1-exp(-k*(1-t))) - k*exp(-k*(1-t))*(1-exp(-k*t))
        """
        exp_kt = torch.exp(-self.k * t)
        exp_k1t = torch.exp(-self.k * (1 - t))
        return self.k * exp_kt * (1 - exp_k1t) - self.k * exp_k1t * (1 - exp_kt)

    def noiser(
        self,
        t: torch.Tensor,      # [batch]
        x: torch.Tensor,      # [batch, dim]
        x0: torch.Tensor,     # [batch, dim]
        x1: torch.Tensor,     # [batch, dim]
    ) -> torch.Tensor:
        """
        Compute raw estimand: d^T * mu_t' * gamma + ||d||^2

        where d = x - mu(t) and mu' = x1 - x0

        Returns:
            Scalar estimand values of shape [batch]
        """
        mu_t = self.mu(t, x0, x1)            # [batch, dim]
        mu_prime = self.dmudt(x0, x1)        # [batch, dim]
        gamma_t = self.g(t)                   # [batch]

        d = x - mu_t  # [batch, dim]
        d_dot_mu_prime = (d * mu_prime).sum(dim=-1)  # [batch]
        d_norm_sq = (d ** 2).sum(dim=-1)  # [batch]

        return d_dot_mu_prime * gamma_t + d_norm_sq  # [batch]

    def _compute_v(
        self,
        t: torch.Tensor,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sample variance v(t) of the averaged estimand.

        v(t) = Var_{x0, x1}[E_x[d^T * mu_t' * gamma + ||d||^2]]

        where d = x - mu(t)

        For fixed (x0, x1), the averaged estimand (over x ~ p_*) is:
            Y(x0, x1) = m^T * (x1 - x0) * gamma + ||m||^2 + E
        where:
            m = mean_x - alpha*x0 - beta*x1
            E = sum of Var(x) across dims
            gamma = g(t)

        Args:
            t: Time value (scalar tensor)
            samples_p0: Samples from p0 [n_p0, dim]
            samples_p1: Samples from p1 [n_p1, dim]

        Returns:
            Scalar variance v(t)
        """
        alpha = 1 - t
        beta = t
        gamma_t = self.g(t)

        # m = mean_x - alpha*x0 - beta*x1 for all (x0, x1) pairs
        m = self._mean_x - alpha * samples_p0 - beta * samples_p1  # [n_samples, dim]
        d = samples_p1 - samples_p0  # x1 - x0

        # Y = m^T * d * gamma + ||m||^2 + E
        m_dot_d = (m * d).sum(dim=-1)  # [n_samples]
        m_norm_sq = (m ** 2).sum(dim=-1)  # [n_samples]
        Y = m_dot_d * gamma_t + m_norm_sq + self._E

        return Y.var() + 1e-8

    def _interpolate_v(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate v(t) from precomputed grid.

        Args:
            t: Time values [batch] or scalar

        Returns:
            Interpolated v(t) values matching input shape
        """
        t_clamped = torch.clamp(t, self.eps, 1 - self.eps)

        grid_range = 1 - 2 * self.eps
        idx_float = (t_clamped - self.eps) / grid_range * (len(self._v_grid_t) - 1)
        idx_low = torch.floor(idx_float).long()
        idx_high = torch.clamp(idx_low + 1, max=len(self._v_grid_t) - 1)
        frac = idx_float - idx_low.float()

        v_low = self._v_grid_vals[idx_low]
        v_high = self._v_grid_vals[idx_high]
        return v_low + frac * (v_high - v_low)

    def _compute_marginal_score(
        self,
        t: float,
        x: torch.Tensor,      # [batch, dim]
        mu0: torch.Tensor,    # [dim]
        Sigma0: torch.Tensor, # [dim, dim]
        mu1: torch.Tensor,    # [dim]
        Sigma1: torch.Tensor, # [dim, dim]
    ) -> torch.Tensor:
        """
        Compute d/dt log p_t(x) for the MARGINAL Gaussian interpolant.

        p_t(x) = N(x; μ_t, Σ_t)
        μ_t = (1-t)*μ₀ + t*μ₁
        Σ_t = (1-t)²*Σ₀ + t²*Σ₁ + γ²(t)*I

        d/dt log p_t(x) = -1/2 tr(Σ_t⁻¹ Σ'_t)
                         + (x-μ_t)ᵀ Σ_t⁻¹ μ'_t
                         + 1/2 (x-μ_t)ᵀ Σ_t⁻¹ Σ'_t Σ_t⁻¹ (x-μ_t)

        Returns:
            Score values of shape [batch]
        """
        dim = x.shape[-1]

        alpha = 1 - t
        beta = t
        alpha_prime = -1.0
        beta_prime = 1.0

        exp_kt = np.exp(-self.k * t)
        exp_k1t = np.exp(-self.k * (1 - t))
        gamma = (1 - exp_kt) * (1 - exp_k1t)
        gamma_prime = self.k * exp_kt * (1 - exp_k1t) - self.k * exp_k1t * (1 - exp_kt)

        mu_t = alpha * mu0 + beta * mu1
        mu_prime_t = alpha_prime * mu0 + beta_prime * mu1

        I_d = torch.eye(dim, dtype=Sigma0.dtype, device=Sigma0.device)
        Sigma_t = alpha**2 * Sigma0 + beta**2 * Sigma1 + gamma**2 * I_d
        Sigma_prime_t = 2*alpha*alpha_prime * Sigma0 + 2*beta*beta_prime * Sigma1 + 2*gamma*gamma_prime * I_d

        Sigma_t_inv = torch.linalg.inv(Sigma_t)

        r = x - mu_t

        term1 = -0.5 * torch.trace(Sigma_t_inv @ Sigma_prime_t)
        term2 = r @ Sigma_t_inv @ mu_prime_t
        M = Sigma_t_inv @ Sigma_prime_t @ Sigma_t_inv
        term3 = 0.5 * torch.einsum('bi,ij,bj->b', r, M, r)

        return term1 + term2 + term3

    def _fit_noiser_network(
        self,
        samples_base: torch.Tensor,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        mu0: Optional[torch.Tensor] = None,
        Sigma0: Optional[torch.Tensor] = None,
        mu1: Optional[torch.Tensor] = None,
        Sigma1: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Train the noiser network using weighted MSE loss.

        Loss = E_{t~Uniform, x0~p0, x1~p1}[w(t) * ||NN(t) - E_{x~p_*}[estimand]||^2]

        where:
        - t is sampled uniformly from [eps, 1-eps]
        - x0, x1 are sampled from p0, p1 respectively
        - The target is the AVERAGE noiser over all x from base distribution p_*
        - weight = 1 / (g(t) + eps)
        """
        self.noiser_network._reset_parameters()
        self.noiser_network.train()

        samples_base = samples_base.to(self.device)
        samples_p0 = samples_p0.to(self.device)
        samples_p1 = samples_p1.to(self.device)

        self._compute_statistics(samples_base, samples_p0, samples_p1)

        n_base = samples_base.shape[0]
        n_p0 = samples_p0.shape[0]
        n_p1 = samples_p1.shape[0]
        dim = samples_base.shape[-1]

        # Precompute v(t) on a grid for variance normalization
        n_v_grid = 100
        self._v_grid_t = torch.linspace(self.eps, 1 - self.eps, n_v_grid, device=self.device)
        self._v_grid_vals = []
        with torch.no_grad():
            for t_val in self._v_grid_t:
                v_t = self._compute_v(t_val, samples_p0, samples_p1)
                self._v_grid_vals.append(v_t)
        self._v_grid_vals = torch.stack(self._v_grid_vals)

        if self.verbose:
            print(f"v(t) range: [{self._v_grid_vals.min().item():.4f}, {self._v_grid_vals.max().item():.4f}]")

        optimizer = torch.optim.AdamW(
            self.noiser_network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        t_eval = torch.linspace(self.eps, 1 - self.eps, self.integration_steps, device=self.device)

        sanity_check_targets = None
        if mu0 is not None and Sigma0 is not None and mu1 is not None and Sigma1 is not None:
            mu0 = mu0.to(self.device)
            Sigma0 = Sigma0.to(self.device)
            mu1 = mu1.to(self.device)
            Sigma1 = Sigma1.to(self.device)

            with torch.no_grad():
                sanity_check_targets = []
                for i, t_val in enumerate(t_eval):
                    noisers = self._compute_marginal_score(
                        t_val.item(), samples_base, mu0, Sigma0, mu1, Sigma1
                    )
                    sanity_check_targets.append(noisers.mean())
                sanity_check_targets = torch.stack(sanity_check_targets)
                print(f'Sanity Check: ELDR {-sanity_check_targets.mean()}')

                weight_eval = 1.0 / (self.g(t_eval) + self.eps)**2

        best_loss = float('inf')
        patience_counter = 0
        global_iter = 0

        num_iters = max(1, min(n_p0, n_p1) // self.batch_size)

        for epoch in range(self.num_epochs):
            perm_p0 = torch.randperm(n_p0, device=self.device)
            perm_p1 = torch.randperm(n_p1, device=self.device)
            shuffled_p0 = samples_p0[perm_p0]
            shuffled_p1 = samples_p1[perm_p1]

            for iter_idx in range(num_iters):
                global_iter += 1

                start = iter_idx * self.batch_size
                end = start + self.batch_size
                x0 = shuffled_p0[start:end]
                x1 = shuffled_p1[start:end]

                # Sample t uniformly from [eps, 1-eps]
                t = self._sample_t_uniform(self.batch_size)

                # For each (t, x0, x1), compute average estimand over ALL x from base distribution
                avg_estimands = []
                for i in range(self.batch_size):
                    t_i = t[i].expand(n_base)
                    x0_i = x0[i].unsqueeze(0).expand(n_base, -1)
                    x1_i = x1[i].unsqueeze(0).expand(n_base, -1)

                    estimands_i = self.noiser(t_i, samples_base, x0_i, x1_i)
                    avg_estimands.append(estimands_i.mean())

                avg_estimands = torch.stack(avg_estimands)

                # Scale by 1/v(t) for variance normalization
                v_t = self._interpolate_v(t)
                scaled_targets = avg_estimands / v_t

                predictions = self.noiser_network(t)

                mse_per_sample = (predictions - scaled_targets.detach()) ** 2

                # Compute weights: 1 / (g(t) + eps)
                g_t = self.g(t)
                weights = 1.0 / (g_t + self.eps)**2

                loss = (mse_per_sample * weights).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.noiser_network.parameters(), max_norm=10.0)
                optimizer.step()

                loss_val = loss.item()

                is_best = loss_val < best_loss

                if is_best:
                    best_loss = loss_val
                    patience_counter = 0
                elif loss_val >= best_loss - self.convergence_threshold:
                    patience_counter += 1

                should_log = is_best or (global_iter % 100 == 0)

                if should_log:
                    with torch.no_grad():
                        nn_predictions = self.noiser_network(t_eval)

                        v_eval = self._interpolate_v(t_eval)
                        gamma_log = self.g(t_eval)
                        gamma_prime_log = self.dgdt(t_eval)

                        # Full integrand: -dim * g'/g + v(t) * NN(t) / gamma^3
                        centering_term = -dim * gamma_prime_log / gamma_log
                        noiser_term = v_eval * nn_predictions / (gamma_log ** 3)
                        full_integrand = centering_term + noiser_term

                        weighted_error = None
                        unweighted_error = None
                        if sanity_check_targets is not None:
                            sq_errors = (full_integrand - sanity_check_targets) ** 2
                            weighted_error = (weight_eval * sq_errors).mean().item()
                            unweighted_error = sq_errors.mean().item()

                        avg_nn = -full_integrand.mean().item()

                    if self.verbose:
                        log_msg = f"[Iter {global_iter}] loss={loss_val:.6f}"
                        if weighted_error is not None:
                            log_msg += f", weighted_err={weighted_error:.6f}, unweighted_err={unweighted_error:.6f}"
                        log_msg += f", eldr_est={avg_nn:.6f}"
                        if is_best:
                            log_msg += " *best*"
                        print(log_msg)

                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Converged at iteration {global_iter}")
                    return

    def _integrate_noiser(self, dim: int) -> float:
        """
        Integrate the full noiser from t=eps to t=1-eps.

        The full integrand is:
            -dim * g'/g + v(t) * NN(t) / gamma^3

        where:
        - The first term is the centering term
        - The second term rescales the network prediction (which predicts target/v(t))
          by v(t) and divides by gamma^3

        Uses Simpson's rule for numerical integration.

        Args:
            dim: Dimensionality of the data

        Returns:
            Integral value (scalar)
        """
        self.noiser_network.eval()

        n_points = self.integration_steps
        if n_points % 2 == 0:
            n_points += 1

        t_vals = torch.linspace(self.eps, 1 - self.eps, n_points, device=self.device)

        with torch.no_grad():
            nn_predictions = self.noiser_network(t_vals)
            v_vals = self._interpolate_v(t_vals)

            gamma_vals = self.g(t_vals)
            gamma_prime_vals = self.dgdt(t_vals)

            # Full integrand: -dim * g'/g + v(t) * NN(t) / gamma^3
            centering_term = -dim * gamma_prime_vals / gamma_vals
            noiser_term = v_vals * nn_predictions / (gamma_vals ** 3)
            integrand = (centering_term + noiser_term).cpu().numpy()

        t_np = t_vals.cpu().numpy()

        # Simpson's rule integration
        h = (t_np[-1] - t_np[0]) / (n_points - 1)
        integral = integrand[0] + integrand[-1]
        for i in range(1, n_points - 1):
            if i % 2 == 0:
                integral += 2 * integrand[i]
            else:
                integral += 4 * integrand[i]
        integral *= h / 3

        return float(integral)

    def estimate_eldr(
        self,
        samples_base: np.ndarray,
        samples_p0: np.ndarray,
        samples_p1: np.ndarray,
        mu0: Optional[torch.Tensor] = None,
        Sigma0: Optional[torch.Tensor] = None,
        mu1: Optional[torch.Tensor] = None,
        Sigma1: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Estimate the Expected Log Density Ratio E_{p_*}[log(p_0(x)/p_1(x))].

        Args:
            samples_base: Samples from the base distribution p_*
            samples_p0: Samples from p_0
            samples_p1: Samples from p_1
            mu0: Optional mean of p_0 (for sanity check logging)
            Sigma0: Optional covariance of p_0 (for sanity check logging)
            mu1: Optional mean of p_1 (for sanity check logging)
            Sigma1: Optional covariance of p_1 (for sanity check logging)

        Returns:
            Scalar ELDR estimate
        """
        samples_base_t = torch.from_numpy(samples_base).float()
        samples_p0_t = torch.from_numpy(samples_p0).float()
        samples_p1_t = torch.from_numpy(samples_p1).float()

        self._fit_noiser_network(
            samples_base_t, samples_p0_t, samples_p1_t,
            mu0=mu0, Sigma0=Sigma0, mu1=mu1, Sigma1=Sigma1
        )

        dim = samples_base.shape[-1]
        integral = self._integrate_noiser(dim)

        return -integral


if __name__ == '__main__':
    from torch.distributions import MultivariateNormal, kl_divergence
    from experiments.utils.two_gaussians_kl import create_two_gaussians_kl

    DIM = 1
    NSAMPLES = 2048
    KL_DISTANCE = 10

    # === CREATE SYNTHETIC DATA ===
    gaussian_pair = create_two_gaussians_kl(DIM, KL_DISTANCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
    mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

    # Use p0 as the base distribution p_*
    samples_base = p0.sample((NSAMPLES,)).numpy()
    samples_p0 = p0.sample((NSAMPLES,)).numpy()
    samples_p1 = p1.sample((NSAMPLES,)).numpy()

    # === ESTIMATE ELDR ===
    estimator = DirectELDREstimator3(
        input_dim=DIM,
        # Network architecture
        hidden_dim=64,
        num_layers=3,
        time_embed_size=64,
        # Interpolant parameters
        k=8.0,
        eps=0.1,
        # Training parameters
        learning_rate=1e-2,
        weight_decay=1e-4,
        num_epochs=200000,
        batch_size=NSAMPLES,
        # Convergence
        patience=32000,
        verbose=True,
        integration_steps=NSAMPLES*2,
        convergence_threshold=1e-8
    )

    # Pass true distribution parameters for sanity check
    eldr_estimate = estimator.estimate_eldr(
        samples_base, samples_p0, samples_p1,
        mu0=mu0, Sigma0=Sigma0, mu1=mu1, Sigma1=Sigma1
    )

    # === COMPUTE TRUE ELDR ===
    # True ELDR = E_{p0}[log p0(x) - log p1(x)] = KL(p0 || p1)
    true_eldr = kl_divergence(p0, p1).item()

    print(f"\nEstimated ELDR: {eldr_estimate:.4f}")
    print(f"True ELDR (KL): {true_eldr:.4f}")
    print(f"Error: {abs(eldr_estimate - true_eldr):.4f}")
