"""
Direct ELDR Estimation using Stochastic Interpolants (Version 3)

Estimates E_{p_*}[log(p_0(x)/p_1(x))] by learning the expected time-derivative
of log p_t along a stochastic interpolant path, then integrating over [0, 1].
"""

import builtins
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Literal

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
        layers.append(nn.GELU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
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
        """Reset model parameters with scaled initialization for stability."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # Scale down weights to start near zero output but with gradients
                module.weight.data *= 0.2
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
        k: float = 16.0,
        eps_train: float = 0.1,  # Training eps for clamping gamma in grad norm clipping / loss weighting
        eps_eval: float = 0.01,  # Integration eps for bounds [eps_eval, 1-eps_eval]
        # Training hyperparameters
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 1000,
        batch_size: int = 256,
        gamma_weight_exp: float = 0.0,  # Loss weighting exponent: 0 = uniform, 3 = match integration importance
        # Integration hyperparameters
        integration_steps: int = 100,
        integration_type: Literal['1', '2', '3'] = '1',  # '1': MC, '2': trapz, '3': Simpson
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
            eps_train: Training epsilon for clamping gamma in grad norm clipping / loss weighting
            eps_eval: Integration epsilon for bounds [eps_eval, 1-eps_eval]
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            num_epochs: Maximum training epochs
            batch_size: Batch size for training
            gamma_weight_exp: Loss weighting exponent for importance sampling (0 = uniform, 3 = match integration)
            integration_steps: Number of points for numerical integration
            integration_type: '1' for MC, '2' for trapz, '3' for Simpson
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
        self.eps_train = eps_train
        self.eps_eval = eps_eval
        self.integration_type = integration_type

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
        self.gamma_weight_exp = gamma_weight_exp
        self.integration_steps = integration_steps
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.verbose = verbose

        # Sample statistics for variance computation
        self._mean_x = None
        self._E = None  # Var(x) summed across dims

    def _sample_t_uniform(self, batch_size: int) -> torch.Tensor:
        """
        Sample t uniformly on [0, 1] (full range).

        Args:
            batch_size: Number of samples to generate

        Returns:
            Tensor of t values of shape [batch_size]
        """
        return torch.rand(batch_size, device=self.device)

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
        Compute raw estimand: d^T * mu_t' * gamma + ||d||^2 * gamma'

        where d = x - mu(t) and mu' = x1 - x0

        Returns:
            Scalar estimand values of shape [batch]
        """
        mu_t        = self.mu(t, x0, x1)            # [batch, dim]
        mu_prime    = self.dmudt(x0, x1)        # [batch, dim]
        gamma_t     = self.g(t)                   # [batch]
        gamma_prime = self.dgdt(t)

        d = x - mu_t  # [batch, dim]
        d_dot_mu_prime = (d * mu_prime).sum(dim=-1)  # [batch]
        d_norm_sq = (d ** 2).sum(dim=-1)  # [batch]

        return d_dot_mu_prime * gamma_t + d_norm_sq * gamma_prime # [batch]

    def _compute_estimand_stats(
        self,
        t: torch.Tensor,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sample mean and std of the averaged estimand.

        For fixed (x0, x1), the averaged estimand (over x ~ p_*) is:
            Y(x0, x1) = m^T * (x1 - x0) * gamma + (||m||^2 + E) * gamma'
        where:
            m = mean_x - alpha*x0 - beta*x1
            E = sum of Var(x) across dims
            gamma = g(t)
            gamma' = dg/dt

        Args:
            t: Time value (scalar tensor)
            samples_p0: Samples from p0 [n_p0, dim]
            samples_p1: Samples from p1 [n_p1, dim]

        Returns:
            (mean, std) tuple
        """
        alpha = 1 - t
        beta = t
        gamma_t = self.g(t)
        gamma_prime_t = self.dgdt(t)

        # m = mean_x - alpha*x0 - beta*x1 for all (x0, x1) pairs
        m = self._mean_x - alpha * samples_p0 - beta * samples_p1  # [n_samples, dim]
        d = samples_p1 - samples_p0  # x1 - x0

        # Y = m^T * d * gamma + (||m||^2 + E) * gamma'
        m_dot_d = (m * d).sum(dim=-1)  # [n_samples]
        m_norm_sq = (m ** 2).sum(dim=-1)  # [n_samples]
        Y = m_dot_d * gamma_t + (m_norm_sq + self._E) * gamma_prime_t

        mean_Y = Y.mean()
        std_Y = Y.std() + 1e-8
        return mean_Y, std_Y

    def _interpolate_E_eta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate E[η](t) (expected estimand) from precomputed grid.

        NOTE: E[η](t) should NEVER use clamped t.

        Args:
            t: Time values [batch] or scalar (NOT clamped)

        Returns:
            Interpolated mean values matching input shape
        """
        grid_range = 1
        idx_float = t / grid_range * (len(self._v_grid_t) - 1)
        idx_low = torch.floor(idx_float).long().clamp(0, len(self._v_grid_t) - 1)
        idx_high = torch.clamp(idx_low + 1, max=len(self._v_grid_t) - 1)
        frac = idx_float - idx_low.float()

        mean_low = self._E_eta_grid_vals[idx_low]
        mean_high = self._E_eta_grid_vals[idx_high]
        return mean_low + frac * (mean_high - mean_low)

    def _interpolate_std(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate std(t) from precomputed grid.

        NOTE: t is ALWAYS clamped with eps_train internally, whether during
        training or integration. This ensures consistency between how the NN
        is trained and how it is evaluated.

        Args:
            t: Time values [batch] or scalar

        Returns:
            Interpolated std values matching input shape
        """
        # Always clamp t for std(t) to ensure training/eval consistency
        t_clamped = torch.clamp(t, self.eps_train, 1 - self.eps_train)

        grid_range = 1
        idx_float = t_clamped / grid_range * (len(self._v_grid_t) - 1)
        idx_low = torch.floor(idx_float).long().clamp(0, len(self._v_grid_t) - 1)
        idx_high = torch.clamp(idx_low + 1, max=len(self._v_grid_t) - 1)
        frac = idx_float - idx_low.float()

        std_low = self._std_grid_vals[idx_low]
        std_high = self._std_grid_vals[idx_high]
        return std_low + frac * (std_high - std_low)

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

    def _compute_decentered_marginal_score(
        self,
        t: float,
        x: torch.Tensor,      # [batch, dim]
        mu0: torch.Tensor,    # [dim]
        Sigma0: torch.Tensor, # [dim, dim]
        mu1: torch.Tensor,    # [dim]
        Sigma1: torch.Tensor, # [dim, dim]
    ) -> torch.Tensor:
        """
        Compute decentered marginal score (without the -dim*γ'/γ centering term).

        This is the marginal score but using Σ'_t_no_gamma = 2αα'Σ₀ + 2ββ'Σ₁
        instead of the full Σ'_t = 2αα'Σ₀ + 2ββ'Σ₁ + 2γγ'I.

        This avoids computing γ'/γ which is numerically unstable near t=0 and t=1.

        Returns:
            Decentered score values of shape [batch]
        """
        dim = x.shape[-1]

        alpha = 1 - t
        beta = t
        alpha_prime = -1.0
        beta_prime = 1.0

        exp_kt = np.exp(-self.k * t)
        exp_k1t = np.exp(-self.k * (1 - t))
        gamma = (1 - exp_kt) * (1 - exp_k1t)

        mu_t = alpha * mu0 + beta * mu1
        mu_prime_t = alpha_prime * mu0 + beta_prime * mu1

        I_d = torch.eye(dim, dtype=Sigma0.dtype, device=Sigma0.device)
        Sigma_t = alpha**2 * Sigma0 + beta**2 * Sigma1 + gamma**2 * I_d
        # Exclude the 2*gamma*gamma'*I term
        Sigma_prime_t_no_gamma = 2*alpha*alpha_prime * Sigma0 + 2*beta*beta_prime * Sigma1

        Sigma_t_inv = torch.linalg.inv(Sigma_t)

        r = x - mu_t

        term1 = -0.5 * torch.trace(Sigma_t_inv @ Sigma_prime_t_no_gamma)
        term2 = r @ Sigma_t_inv @ mu_prime_t
        M = Sigma_t_inv @ Sigma_prime_t_no_gamma @ Sigma_t_inv
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

        # Precompute E[η](t) and std(t) on a grid for normalization
        n_v_grid = 100
        self._v_grid_t = torch.linspace(0, 1, n_v_grid, device=self.device)
        self._E_eta_grid_vals = []
        self._std_grid_vals = []
        with torch.no_grad():
            for t_val in self._v_grid_t:
                E_eta_t, std_t = self._compute_estimand_stats(t_val, samples_p0, samples_p1)
                self._E_eta_grid_vals.append(E_eta_t)
                self._std_grid_vals.append(std_t)
        self._E_eta_grid_vals = torch.stack(self._E_eta_grid_vals)
        self._std_grid_vals = torch.stack(self._std_grid_vals)

        if self.verbose:
            print(f"E[η](t) range: [{self._E_eta_grid_vals.min().item():.4f}, {self._E_eta_grid_vals.max().item():.4f}]")
            print(f"std(t) range: [{self._std_grid_vals.min().item():.4f}, {self._std_grid_vals.max().item():.4f}]")

        optimizer = torch.optim.AdamW(
            self.noiser_network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # LR scheduler: warmup + cosine annealing
        warmup_iters = 50
        def lr_lambda(iter):
            if iter < warmup_iters:
                return iter / warmup_iters  # Linear warmup
            else:
                # Cosine decay after warmup
                progress = (iter - warmup_iters) / (self.num_epochs * num_iters - warmup_iters)
                return 0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        t_eval = torch.linspace(self.eps_eval, 1 - self.eps_eval, self.integration_steps, device=self.device)

        sanity_check_targets = None
        decentered_targets = None
        if mu0 is not None and Sigma0 is not None and mu1 is not None and Sigma1 is not None:
            mu0 = mu0.to(self.device)
            Sigma0 = Sigma0.to(self.device)
            mu1 = mu1.to(self.device)
            Sigma1 = Sigma1.to(self.device)

            with torch.no_grad():
                sanity_check_targets = []
                decentered_targets = []
                for i, t_val in enumerate(t_eval):
                    scores = self._compute_marginal_score(
                        t_val.item(), samples_base, mu0, Sigma0, mu1, Sigma1
                    )
                    sanity_check_targets.append(scores.mean())

                    # Compute decentered targets (without γ'/γ term) for stable comparison
                    decentered_scores = self._compute_decentered_marginal_score(
                        t_val.item(), samples_base, mu0, Sigma0, mu1, Sigma1
                    )
                    decentered_targets.append(decentered_scores.mean())

                sanity_check_targets = torch.stack(sanity_check_targets)
                decentered_targets = torch.stack(decentered_targets)
                # MC integration: ∫_ε^{1-ε} f dt ≈ (1-2ε) * mean(f)
                integration_range = (1 - self.eps_eval) - self.eps_eval
                true_eldr = -integration_range * sanity_check_targets.mean().item()
                true_eldr_no_range = -sanity_check_targets.mean().item()  # For comparison
                self._sanity_eldr = true_eldr  # Store for later access
                self._sanity_eldr_no_range = true_eldr_no_range
                print(f'Sanity Check: ELDR {true_eldr:.4f} (w/o range factor: {true_eldr_no_range:.4f})')

                weight_eval = 1.0 / (self.g(t_eval) + self.eps_train)**2

                # Baseline ELDR: Riemann sum of Monte Carlo decentered scores
                dt = t_eval[1] - t_eval[0]
                baseline_eldr = -(decentered_targets.sum() * dt).item()
                self._baseline_eldr = baseline_eldr
                print(f'Baseline ELDR (MC Riemann): {baseline_eldr:.4f}')

        best_loss = float('inf')
        patience_counter = 0
        global_iter = 0
        best_model_state = None
        best_eldr_model_state = None  # Track model with best ELDR error

        # Collect stats for plotting (when verbose=False)
        self._stats_history = []

        # Initialize best tracking dict
        best_stats = {
            'weighted_err': float('inf'),
            'weighted_rel_err': float('inf'),
            'unweighted_err': float('inf'),
            'unweighted_rel_err': float('inf'),
            'decentered_err': float('inf'),
            'decentered_rel_err': float('inf'),
            'eldr_err': float('inf'),
            'eldr_rel_err': float('inf'),
            'best_eldr_est': None,  # Track the estimate with lowest ELDR error
        }

        baseline_eldr = None

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

                # Monte Carlo: sample one x_base per (x0, x1, t) tuple
                x_base_idx = torch.randint(0, n_base, (self.batch_size,), device=self.device)
                x_base = samples_base[x_base_idx]  # [B, dim]

                # Compute estimands in one batched call
                avg_estimands = self.noiser(t, x_base, x0, x1)  # [B]

                # Standard normalization: target / std (no mean centering)
                # std(t) ALWAYS clamped internally
                std_t = self._interpolate_std(t)  # clamps internally with eps_train
                scaled_targets = avg_estimands / std_t

                # Clamp t for gamma computation in weighting/clipping only
                t_clamped = torch.clamp(t, self.eps_train, 1 - self.eps_train)

                predictions = self.noiser_network(t)

                mse_per_sample = (predictions - scaled_targets.detach()) ** 2

                # Compute importance weights to emphasize boundary regions
                # t_clamped is already clamped to [eps_train, 1-eps_train] above
                g_t = self.g(t_clamped)
                if self.gamma_weight_exp == 0:
                    weights = torch.ones_like(mse_per_sample)
                else:
                    # Weight by 1/γ^p to match integration importance (which has 1/γ³ in denominator)
                    weights = 1.0 / (g_t + self.eps_train)**self.gamma_weight_exp
                    # Normalize to prevent loss scale issues
                    weights = weights / weights.mean()

                loss = (mse_per_sample * weights).mean()

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (configurable) - use global config if available
                CLIP_BASE = 10.0
                CLIP_GAMMA_EXP = getattr(builtins, '_CFG_CLIP_GAMMA_EXP', -2)
                gamma_min = g_t.min().item()  # Now uses clamped gamma
                clip_norm = CLIP_BASE * (gamma_min + self.eps_train)**CLIP_GAMMA_EXP
                torch.nn.utils.clip_grad_norm_(self.noiser_network.parameters(), max_norm=clip_norm)
                optimizer.step()
                scheduler.step()  # Step per iteration for warmup

                loss_val = loss.item()

                is_best = loss_val < best_loss

                if is_best:
                    best_loss = loss_val
                    patience_counter = 0
                    # Save best model checkpoint
                    # import copy
                    # best_model_state = copy.deepcopy(self.noiser_network.state_dict())
                elif loss_val >= best_loss - self.convergence_threshold:
                    patience_counter += 1

                should_log = (global_iter % 10 == 0) or (global_iter <= 5)  # Log every 10th iter

                if should_log:
                    with torch.no_grad():
                        nn_predictions = self.noiser_network(t_eval)

                        # std(t) ALWAYS clamped internally
                        std_eval = self._interpolate_std(t_eval)  # clamps internally
                        gamma_log = self.g(t_eval)
                        gamma_prime_log = self.dgdt(t_eval)

                        # Full integrand: (std * NN(t)) / gamma^3
                        # NOTE: centering term -d*gamma'/gamma is antisymmetric on [0,1], integrates to 0
                        noiser_term = (std_eval * nn_predictions) / (gamma_log ** 3)
                        full_integrand = -(noiser_term)  # Negate to fix sign

                        weighted_error = None
                        unweighted_error = None
                        weighted_rel_err = None
                        unweighted_rel_err = None
                        eldr_err = None
                        eldr_rel_err = None
                        if sanity_check_targets is not None:
                            sq_errors = (full_integrand - sanity_check_targets) ** 2
                            weighted_error = (weight_eval * sq_errors).mean().item()
                            unweighted_error = sq_errors.mean().item()
                            abs_errors = abs(full_integrand - sanity_check_targets)
                            weighted_abs_error = ((weight_eval).sqrt() * abs_errors).mean().item()
                            unweighted_abs_error = abs_errors.mean().item()

                            # Compute relative errors
                            target_abs_mean = abs(sanity_check_targets).mean().item() + 1e-8
                            weighted_rel_err =  weighted_abs_error/ target_abs_mean
                            unweighted_rel_err = unweighted_abs_error / target_abs_mean

                            # Update best tracking
                            if weighted_error < best_stats['weighted_err']:
                                best_stats['weighted_err'] = weighted_error
                            if weighted_rel_err < best_stats['weighted_rel_err']:
                                best_stats['weighted_rel_err'] = weighted_rel_err
                            if unweighted_error < best_stats['unweighted_err']:
                                best_stats['unweighted_err'] = unweighted_error
                            if unweighted_rel_err < best_stats['unweighted_rel_err']:
                                best_stats['unweighted_rel_err'] = unweighted_rel_err

                        decentered_err = None
                        decentered_rel_err = None
                        if decentered_targets is not None:
                            # Compare noiser_term directly to decentered targets
                            # (computed without γ'/γ for numerical stability)
                            decentered_sq_errors = (noiser_term - decentered_targets) ** 2
                            decentered_err = decentered_sq_errors.mean().item()

                            # Compute relative error
                            decentered_abs_mean = abs(decentered_targets).mean().item() + 1e-8
                            decentered_rel_err = abs(noiser_term-decentered_targets).mean().item() / decentered_abs_mean

                            # Update best tracking
                            if decentered_err < best_stats['decentered_err']:
                                best_stats['decentered_err'] = decentered_err
                            if decentered_rel_err < best_stats['decentered_rel_err']:
                                best_stats['decentered_rel_err'] = decentered_rel_err

                        # Compute actual ELDR estimate via integration (same as final)
                        # Note: estimate_eldr returns -integral, so negate here too
                        avg_nn = -self._integrate_noiser(dim)

                        # Compute ELDR error if true_eldr is available
                        if sanity_check_targets is not None:
                            eldr_err = abs(avg_nn - true_eldr)
                            eldr_rel_err = eldr_err / (abs(true_eldr) + 1e-8)

                            # Update best tracking and save model with best ELDR
                            if eldr_err < best_stats['eldr_err']:
                                best_stats['eldr_err'] = eldr_err
                                best_stats['best_eldr_est'] = avg_nn  # Save best estimate
                                # Save model state with best ELDR error
                                import copy
                                best_eldr_model_state = copy.deepcopy(self.noiser_network.state_dict())
                            if eldr_rel_err < best_stats['eldr_rel_err']:
                                best_stats['eldr_rel_err'] = eldr_rel_err

                        # Collect stats for plotting
                        self._stats_history.append({
                            'iter': global_iter,
                            'loss': loss_val,
                            'eldr_err': eldr_err if eldr_err is not None else float('nan'),
                            'eldr_est': avg_nn,
                            'dec_err': decentered_err if decentered_err is not None else float('nan'),
                            'baseline_eldr': baseline_eldr if baseline_eldr is not None else float('nan'),
                        })

                    if self.verbose:
                        log_msg = f"[Iter {global_iter}] loss={loss_val:.6f}"
                        if is_best:
                            log_msg += " *best*"
                        if weighted_error is not None:
                            log_msg += f", wt_err={weighted_error:.4f}(best:{best_stats['weighted_err']:.4f})"
                            log_msg += f", wt_rel={weighted_rel_err:.4f}(best:{best_stats['weighted_rel_err']:.4f})"
                            log_msg += f", unwt_err={unweighted_error:.4f}(best:{best_stats['unweighted_err']:.4f})"
                            log_msg += f", unwt_rel={unweighted_rel_err:.4f}(best:{best_stats['unweighted_rel_err']:.4f})"
                        if decentered_err is not None:
                            log_msg += f", dec_err={decentered_err:.4f}(best:{best_stats['decentered_err']:.4f})"
                            log_msg += f", dec_rel={decentered_rel_err:.4f}(best:{best_stats['decentered_rel_err']:.4f})"
                        if eldr_err is not None:
                            log_msg += f", eldr_err={eldr_err:.4f}(best:{best_stats['eldr_err']:.4f})"
                            log_msg += f", eldr_rel={eldr_rel_err:.4f}(best:{best_stats['eldr_rel_err']:.4f})"
                        log_msg += f", eldr_est={avg_nn:.4f}"
                        print(log_msg)

                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Converged at iteration {global_iter}")
                    # Restore best ELDR model (not best loss model)
                    # if best_eldr_model_state is not None:
                    #     self.noiser_network.load_state_dict(best_eldr_model_state)
                    # elif best_model_state is not None:
                    #     self.noiser_network.load_state_dict(best_model_state)
                    self._best_stats = best_stats
                    return

            # LR scheduler is stepped per iteration (see above)

        # Don't restore - use final model state
        self._best_stats = best_stats

    def _integrate_noiser(self, dim: int) -> float:
        """
        Integrate the full noiser from t=eps_eval to t=1-eps_eval.

        The full integrand is:
            (mean + std * NN(t)) / gamma^3

        Uses integration method specified by self.integration_type:
            '1': Monte Carlo (mean * range)
            '2': Trapezoidal (torch.trapz)
            '3': Simpson's rule

        Args:
            dim: Dimensionality of the data

        Returns:
            Integral value (scalar)
        """
        self.noiser_network.eval()

        n_points = self.integration_steps
        # Ensure odd number of points for Simpson's rule
        if self.integration_type == '3' and n_points % 2 == 0:
            n_points += 1
        t_vals = torch.linspace(self.eps_eval, 1 - self.eps_eval, n_points, device=self.device)

        with torch.no_grad():
            nn_predictions = self.noiser_network(t_vals)
            # std(t) ALWAYS clamped internally
            std_vals = self._interpolate_std(t_vals)  # clamps internally

            gamma_vals = self.g(t_vals)

            # Full integrand: (std * NN(t)) / gamma^3
            # NOTE: centering term -d*gamma'/gamma is antisymmetric on [0,1], integrates to 0
            noiser_term = (std_vals * nn_predictions) / (gamma_vals ** 3)
            integrand = -noiser_term  # Negate to fix sign

        # Integration based on type
        if self.integration_type == '3':
            # Simpson's rule integration
            h = (t_vals[-1] - t_vals[0]) / (n_points - 1)
            integral_val = integrand[0] + integrand[-1]
            for i in range(1, n_points - 1):
                if i % 2 == 0:
                    integral_val = integral_val + 2 * integrand[i]
                else:
                    integral_val = integral_val + 4 * integrand[i]
            integral = float((h / 3 * integral_val).item())
        elif self.integration_type == '2':
            # Trapezoidal integration
            integral = float(torch.trapz(integrand, t_vals).item())
        else:  # '1' - Monte Carlo
            # Monte Carlo integration: integral ≈ (b - a) * mean(f)
            integration_range = (1 - self.eps_eval) - self.eps_eval  # = 1 - 2*eps_eval
            integral = float(integration_range * integrand.mean().item())

        # Debug prints
        if hasattr(self, '_debug_integrate') and self._debug_integrate:
            print(f"  [integrate] t range: [{t_vals[0].item():.4f}, {t_vals[-1].item():.4f}]")
            print(f"  [integrate] gamma range: [{gamma_vals.min().item():.6f}, {gamma_vals.max().item():.6f}]")
            print(f"  [integrate] mean_vals range: [{mean_vals.min().item():.4f}, {mean_vals.max().item():.4f}]")
            print(f"  [integrate] std_vals range: [{std_vals.min().item():.4f}, {std_vals.max().item():.4f}]")
            print(f"  [integrate] nn_predictions range: [{nn_predictions.min().item():.4f}, {nn_predictions.max().item():.4f}]")
            print(f"  [integrate] noiser_term range: [{noiser_term.min().item():.4f}, {noiser_term.max().item():.4f}]")
            print(f"  [integrate] integrand mean: {integrand.mean().item():.4f}")
            print(f"  [integrate] integration_type: {self.integration_type}, integral: {integral:.4f}")

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
    import matplotlib.pyplot as plt
    from torch.distributions import MultivariateNormal, kl_divergence
    from experiments.utils.prescribed_kls import create_two_gaussians_kl

    DIM = 1
    NSAMPLES = 8192
    KL_DISTANCE = 10

    # === CREATE SYNTHETIC DATA ===
    gaussian_pair = create_two_gaussians_kl(DIM, KL_DISTANCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
    mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

    samples_base = p0.sample((NSAMPLES,)).numpy()
    samples_p0 = p0.sample((NSAMPLES,)).numpy()
    samples_p1 = p1.sample((NSAMPLES,)).numpy()

    true_eldr = kl_divergence(p0, p1).item()
    print(f"True ELDR (KL): {true_eldr:.4f}")
    print("="*80)

    # === CONFIGS TO TRY (update based on analysis) ===
    # Test uniform weighting (gamma_weight_exp=0) vs importance weighting (gamma_weight_exp=3)
    # The importance weighting should help NN learn better near boundaries where 1/γ³ is large
    # CONFIGS = [
    #     # Baseline: uniform weighting
    #     {'name': 'k=16 uniform', 'eps_train': 0.03, 'eps_eval': 0.03, 'lr': 2e-6, 'gamma_weight_exp': 0, 'k': 16, 'seed': 0},
    #     {'name': 'k=25 uniform', 'eps_train': 0.03, 'eps_eval': 0.03, 'lr': 2e-6, 'gamma_weight_exp': 0, 'k': 25, 'seed': 0},
    #     # Importance weighted: gamma_weight_exp=3 matches the 1/γ³ in the integrand
    #     {'name': 'k=16 weighted', 'eps_train': 0.03, 'eps_eval': 0.03, 'lr': 2e-6, 'gamma_weight_exp': 3, 'k': 16, 'seed': 0},
    #     {'name': 'k=25 weighted', 'eps_train': 0.03, 'eps_eval': 0.03, 'lr': 2e-6, 'gamma_weight_exp': 3, 'k': 25, 'seed': 0},
    # ]
    CONFIGS = [
        {'name': f's{s}k{k}et{et}ee{ee}lr{elar}e{expo}', 'eps_train': et, 'eps_eval': ee, 'lr': elar, 'gamma_weight_exp': expo, 'k': k, 'seed': s} for
        et in [0.01] for
        ee in [0.03] for
        elar in [3e-7, 9e-7, 3e-6] for
        expo in [0] for
        k in [30] for
        s in [2, 0, 1]
    ]

    all_results = {}

    for cfg in CONFIGS:
        print(f"Running config: {cfg['name']}")
        # Reset seed for reproducibility
        if 'seed' in cfg:
            torch.manual_seed(cfg['seed'])
            np.random.seed(cfg['seed'])

        estimator = DirectELDREstimator3(
            input_dim=DIM,
            hidden_dim=256,
            num_layers=3,
            time_embed_size=128,
            k=cfg.get('k', 16.0),
            eps_train=cfg['eps_train'],
            eps_eval=cfg['eps_eval'],
            learning_rate=cfg['lr'],
            weight_decay=1e-4,
            num_epochs=1000,
            batch_size=256,
            gamma_weight_exp=cfg.get('gamma_weight_exp', 0),
            patience=10000,
            verbose=False,  # Use plots instead
            integration_steps=100,
            convergence_threshold=1e-8
        )

        estimator._debug_integrate = False
        eldr_estimate = estimator.estimate_eldr(
            samples_base, samples_p0, samples_p1,
            mu0=mu0, Sigma0=Sigma0, mu1=mu1, Sigma1=Sigma1
        )

        # Get diagnostic info about the integrand
        with torch.no_grad():
            t_diag = torch.linspace(cfg['eps_eval'], 1 - cfg['eps_eval'], 100, device=estimator.device)
            nn_pred = estimator.noiser_network(t_diag)
            std_diag = estimator._interpolate_std(t_diag)
            E_eta_diag = estimator._interpolate_E_eta(t_diag)
            gamma_diag = estimator.g(t_diag)

            # Our integrand
            noiser_term = (std_diag * nn_pred) / (gamma_diag ** 3)
            our_integrand = -noiser_term

            # True NN target: E[η](t) / std(t)
            # The NN learns h(t) = E[η](t) / std(t)
            true_nn_target = E_eta_diag / std_diag

            # Compute various targets for comparison
            marginal_scores = []
            decentered_scores = []
            E_eta_over_gamma3 = []  # Direct computation of E[η]/gamma^3
            for t_val in t_diag:
                # Full marginal score
                scores = estimator._compute_marginal_score(
                    t_val.item(), torch.from_numpy(samples_base).float().to(estimator.device),
                    mu0.to(estimator.device), Sigma0.to(estimator.device),
                    mu1.to(estimator.device), Sigma1.to(estimator.device)
                )
                marginal_scores.append(scores.mean())

                # Decentered marginal score (without the 2*gamma*gamma'*I term in Sigma')
                dec_scores = estimator._compute_decentered_marginal_score(
                    t_val.item(), torch.from_numpy(samples_base).float().to(estimator.device),
                    mu0.to(estimator.device), Sigma0.to(estimator.device),
                    mu1.to(estimator.device), Sigma1.to(estimator.device)
                )
                decentered_scores.append(dec_scores.mean())

                # Direct computation of E[η](t) / gamma^3
                gamma_val = estimator.g(t_val)
                E_eta_t = estimator._interpolate_E_eta(t_val)
                E_eta_over_gamma3.append(E_eta_t / (gamma_val ** 3))

            marginal_scores = torch.stack(marginal_scores)
            decentered_scores = torch.stack(decentered_scores)
            E_eta_over_gamma3 = torch.stack(E_eta_over_gamma3)

            # NN reconstruction: std(t) * NN(t) / gamma^3
            nn_reconstruction = (std_diag * nn_pred) / (gamma_diag ** 3)

            # Get baseline ELDR if available
            baseline_eldr = getattr(estimator, '_baseline_eldr', None)

        all_results[cfg['name']] = {
            'stats': estimator._stats_history,
            'final_est': eldr_estimate,
            'final_err': abs(eldr_estimate - true_eldr),
            'sanity_eldr': estimator._sanity_eldr,  # Store sanity check ELDR (with range factor)
            'sanity_eldr_no_range': estimator._sanity_eldr_no_range,  # Without range factor
            't_diag': t_diag.cpu().numpy(),
            'our_integrand': our_integrand.cpu().numpy(),
            'marginal_scores': marginal_scores.cpu().numpy(),
            'decentered_scores': decentered_scores.cpu().numpy(),
            'E_eta_over_gamma3': E_eta_over_gamma3.cpu().numpy(),  # Direct E[η]/gamma^3
            'nn_reconstruction': nn_reconstruction.cpu().numpy(),  # std*NN/gamma^3
            'nn_pred': nn_pred.cpu().numpy(),
            'true_nn_target': true_nn_target.cpu().numpy(),  # Add true NN target
            'E_eta_diag': E_eta_diag.cpu().numpy(),  # E[η](t) for analysis
            'std_diag': std_diag.cpu().numpy(),
            'gamma_diag': gamma_diag.cpu().numpy(),
            'baseline_eldr': baseline_eldr,
            'eldr_evolution': [s['eldr_est'] for s in estimator._stats_history],
        }
        print(f"  Final: {eldr_estimate:.4f} (err={abs(eldr_estimate - true_eldr):.4f})")
        if baseline_eldr is not None:
            print(f"  Baseline ELDR (MC Riemann): {baseline_eldr:.4f}")

    # === PLOT RESULTS ===
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for name, result in all_results.items():
        stats = result['stats']
        iters = [s['iter'] for s in stats]
        losses = [s['loss'] for s in stats]
        eldr_errs = [s['eldr_err'] for s in stats]
        eldr_ests = [s['eldr_est'] for s in stats]

        axes[0, 0].plot(iters, losses, label=name)
        axes[0, 1].plot(iters, eldr_errs, label=name)
        axes[0, 2].plot(iters, eldr_ests, label=name)

        # Plot NN learning error: NN(t) - E[η](t)/std(t)
        t_diag = result['t_diag']
        nn_error = result['nn_pred'] - result['true_nn_target']
        axes[0, 3].plot(t_diag, nn_error, label=name)

        # KEY PLOT: NN(t) vs true target (mean/std)
        axes[1, 0].plot(t_diag, result['nn_pred'], label=f'{name} NN(t)')
        axes[1, 0].plot(t_diag, result['true_nn_target'], '--', label=f'{name} target')

        # KEY PLOT: NN reconstruction (std*NN/γ³) vs MC decentered score
        axes[1, 1].plot(t_diag, result['nn_reconstruction'], label=f'{name} std·NN/γ³')
        axes[1, 1].plot(t_diag, result['decentered_scores'], '--', label=f'{name} MC dec')

        # Plot gamma(t) and std(t) for context
        axes[1, 2].plot(t_diag, result['gamma_diag'], label=f'{name} γ(t)')
        axes[1, 2].plot(t_diag, result['std_diag'], '--', label=f'{name} std(t)')

        # Integration cumsum showing convergence
        dt = t_diag[1] - t_diag[0]
        nn_cumsum = np.cumsum(-result['nn_reconstruction']) * dt
        mc_cumsum = np.cumsum(-result['decentered_scores']) * dt
        axes[1, 3].plot(t_diag, nn_cumsum, label=f'{name} NN cumsum')
        axes[1, 3].plot(t_diag, mc_cumsum, '--', label=f'{name} MC cumsum')

    axes[0, 0].set_title('Loss'); axes[0, 0].set_ylabel('Loss')
    axes[0, 1].set_title('ELDR Error'); axes[0, 1].set_ylabel('|est - true|')
    axes[0, 2].set_title('ELDR Estimate'); axes[0, 2].set_ylabel('Estimate')
    axes[0, 2].axhline(y=true_eldr, color='k', linestyle='--', label='True ELDR')
    axes[0, 3].set_title('NN Learning Error: NN(t) - E[η]/std'); axes[0, 3].set_ylabel('Error')
    axes[0, 3].axhline(y=0, color='k', linestyle='--', alpha=0.5)

    axes[1, 0].set_title('NN(t) vs Target E[η](t)/std(t)'); axes[1, 0].set_ylabel('Value')
    axes[1, 1].set_title('NN Recon (std·NN/γ³) vs MC Decentered'); axes[1, 1].set_ylabel('Value')
    axes[1, 2].set_title('γ(t) and std(t)'); axes[1, 2].set_ylabel('Value')
    axes[1, 3].set_title('Integration Cumsum'); axes[1, 3].set_ylabel('Cumulative Integral')
    axes[1, 3].axhline(y=true_eldr, color='k', linestyle='--', label='True ELDR')

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.set_xlabel('t' if ax in axes[1] else 'Iteration')

    plt.tight_layout()
    plt.savefig('hpo_curvesC.png', dpi=150)
    plt.show()
    # Print summary statistics
    errors = [r['final_err'] for r in all_results.values()]
    estimates = [r['final_est'] for r in all_results.values()]
    print(f"\n=== Summary ===")
    print(f"True ELDR: {true_eldr:.4f}")
    print(f"Mean estimate: {np.mean(estimates):.4f} (std={np.std(estimates):.4f})")
    print(f"Mean error: {np.mean(errors):.4f} (std={np.std(errors):.4f})")
    print(f"Best error: {min(errors):.4f}, Worst error: {max(errors):.4f}")

    # Diagnostic: Compare NN estimates to sanity check ELDR
    print(f"\n=== NN Learning Diagnostics ===")
    for name, result in all_results.items():
        nn_target_err = np.abs(result['nn_pred'] - result['true_nn_target']).mean()
        nn_target_max_err = np.abs(result['nn_pred'] - result['true_nn_target']).max()
        sanity_eldr = result['sanity_eldr']
        sanity_no_range = result['sanity_eldr_no_range']
        est = result['final_est']

        # Compare NN reconstruction to MC decentered score
        nn_recon = result['nn_reconstruction']
        dec_scores = result['decentered_scores']
        recon_vs_decentered = np.abs(nn_recon - dec_scores).mean()

        # Compare mean/gamma^3 to decentered score (what we expect to match)
        E_eta_g3 = result['E_eta_over_gamma3']
        E_eta_g3_vs_dec = np.abs(E_eta_g3 - dec_scores).mean()

        print(f"{name}:")
        print(f"  NN target MAE: {nn_target_err:.4f}, Max: {nn_target_max_err:.4f}")
        print(f"  E[η]/γ³ vs MC decentered MAE: {E_eta_g3_vs_dec:.4f}")
        print(f"  NN recon (std*NN/γ³) vs MC decentered MAE: {recon_vs_decentered:.4f}")

        baseline = result['baseline_eldr']
        if baseline is not None:
            print(f"  Baseline ELDR (MC Riemann): {baseline:.4f}, err={abs(baseline - true_eldr):.4f}")

        print(f"  Estimate: {est:.4f}, Sanity(w/range): {sanity_eldr:.4f}, Sanity(w/o): {sanity_no_range:.4f}")
        print(f"  True ELDR: {true_eldr:.4f}, Est-True: {est - true_eldr:.4f}, Sanity-True: {sanity_eldr - true_eldr:.4f}")

    print("\nSaved plot to hpo_curvesC.png")
