"""
Direct ELDR Estimation using Stochastic Interpolants (Version 3) - Normalization Mode Test

Copy of direct3.py with modifications to test 4 normalization modes:
- 'default': Original behavior
- 'sym_mean_asym_std': Symmetric mean (mean(t) - mean(1-t))/2, asymmetric std
- 'sym_mean_sym_std': Symmetric mean and std (std(t) + std(1-t))/2
- 'no_mean_asym_std': Zero mean, asymmetric std
- 'no_mean_no_std': Zero mean, unit std (no normalization)
"""

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
        """Reset model parameters with scaled initialization for stability."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # Scale down weights to start near zero output but with gradients
                module.weight.data *= 0.2
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class DirectELDREstimator3NormTest(ELDREstimator):
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
        # Integration hyperparameters
        integration_steps: int = 100,
        integration_type: Literal['1', '2', '3'] = '1',  # '1': MC, '2': trapz, '3': Simpson
        # Convergence hyperparameters
        convergence_threshold: float = 1e-4,
        patience: int = 200,
        # Normalization mode
        norm_mode: str = 'default',  # 'default', 'sym_mean_asym_std', 'sym_mean_sym_std', 'no_mean_asym_std', 'no_mean_no_std'
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
            integration_steps: Number of points for numerical integration
            integration_type: '1' for MC, '2' for trapz, '3' for Simpson
            convergence_threshold: Stop if loss change < threshold for patience steps
            patience: Number of steps to wait for convergence
            norm_mode: Normalization mode - 'default', 'sym_mean_asym_std', 'sym_mean_sym_std', 'no_mean_asym_std', 'no_mean_no_std'
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
        self.norm_mode = norm_mode

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

    def _interpolate_mean(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate mean(t) from precomputed grid.

        NOTE: mean(t) should NEVER use clamped t.

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

        mean_low = self._mean_grid_vals[idx_low]
        mean_high = self._mean_grid_vals[idx_high]
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

    def _get_mean(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get mean(t) according to normalization mode.

        Modes:
        - 'default': Original mean(t)
        - 'sym_mean_*': (mean(t) - mean(1-t)) / 2 (antisymmetric)
        - 'no_mean_*': Zero

        Args:
            t: Time values [batch] or scalar

        Returns:
            Mean values matching input shape
        """
        if self.norm_mode in ('no_mean_asym_std', 'no_mean_no_std'):
            return torch.zeros_like(t)
        elif self.norm_mode in ('sym_mean_asym_std', 'sym_mean_sym_std'):
            mean_t = self._interpolate_mean(t)
            mean_1_minus_t = self._interpolate_mean(1 - t)
            return (mean_t - mean_1_minus_t) / 2
        else:  # 'default'
            return self._interpolate_mean(t)

    def _get_std(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get std(t) according to normalization mode.

        Modes:
        - 'default', 'sym_mean_asym_std', 'no_mean_asym_std': Original std(t)
        - 'sym_mean_sym_std': (std(t) + std(1-t)) / 2 (symmetric)
        - 'no_mean_no_std': 1 (no normalization)

        Args:
            t: Time values [batch] or scalar

        Returns:
            Std values matching input shape
        """
        if self.norm_mode == 'no_mean_no_std':
            return torch.ones_like(t)
        elif self.norm_mode == 'sym_mean_sym_std':
            std_t = self._interpolate_std(t)
            std_1_minus_t = self._interpolate_std(1 - t)
            return (std_t + std_1_minus_t) / 2
        else:  # 'default', 'sym_mean_asym_std', 'no_mean_asym_std'
            return self._interpolate_std(t)

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

        # Precompute mean(t) and std(t) on a grid for normalization
        n_v_grid = 100
        self._v_grid_t = torch.linspace(0, 1, n_v_grid, device=self.device)
        self._mean_grid_vals = []
        self._std_grid_vals = []
        with torch.no_grad():
            for t_val in self._v_grid_t:
                mean_t, std_t = self._compute_estimand_stats(t_val, samples_p0, samples_p1)
                self._mean_grid_vals.append(mean_t)
                self._std_grid_vals.append(std_t)
        self._mean_grid_vals = torch.stack(self._mean_grid_vals)
        self._std_grid_vals = torch.stack(self._std_grid_vals)

        if self.verbose:
            print(f"mean(t) range: [{self._mean_grid_vals.min().item():.4f}, {self._mean_grid_vals.max().item():.4f}]")
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
                true_eldr = -sanity_check_targets.mean().item()
                self._sanity_eldr = true_eldr  # Store for later access
                if self.verbose:
                    print(f'Sanity Check: ELDR {true_eldr}')

                weight_eval = 1.0 / (self.g(t_eval) + self.eps_train)**2

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

                # Standard normalization using _get_mean and _get_std
                mean_t = self._get_mean(t)
                std_t = self._get_std(t)
                scaled_targets = (avg_estimands - mean_t) / std_t

                # Clamp t for gamma computation in weighting/clipping only
                t_clamped = torch.clamp(t, self.eps_train, 1 - self.eps_train)

                predictions = self.noiser_network(t)

                mse_per_sample = (predictions - scaled_targets.detach()) ** 2

                # Compute weights - use global config if available
                # t_clamped is already clamped to [eps_train, 1-eps_train] above
                g_t = self.g(t_clamped)
                import builtins
                WEIGHT_EXP = getattr(builtins, '_CFG_WEIGHT_EXP', 0)
                if WEIGHT_EXP == 0:
                    weights = torch.ones_like(mse_per_sample)
                else:
                    weights = std_t**2 / (g_t + self.eps_train)**WEIGHT_EXP

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
                elif loss_val >= best_loss - self.convergence_threshold:
                    patience_counter += 1

                should_log = (global_iter % 10 == 0) or (global_iter <= 5)  # Log every 10th iter

                if should_log:
                    with torch.no_grad():
                        nn_predictions = self.noiser_network(t_eval)

                        # Use _get_mean and _get_std for consistent normalization
                        mean_eval = self._get_mean(t_eval)
                        std_eval = self._get_std(t_eval)
                        gamma_log = self.g(t_eval)
                        gamma_prime_log = self.dgdt(t_eval)

                        # Full integrand: (mean + std * NN(t)) / gamma^3
                        # NOTE: centering term -d*gamma'/gamma is antisymmetric on [0,1], integrates to 0
                        noiser_term = (mean_eval + std_eval * nn_predictions) / (gamma_log ** 3)
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
            # Use _get_mean and _get_std for consistent normalization
            mean_vals = self._get_mean(t_vals)
            std_vals = self._get_std(t_vals)

            gamma_vals = self.g(t_vals)

            # Full integrand: (mean + std * NN(t)) / gamma^3
            # NOTE: centering term -d*gamma'/gamma is antisymmetric on [0,1], integrates to 0
            noiser_term = (mean_vals + std_vals * nn_predictions) / (gamma_vals ** 3)
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
    import builtins

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

    # === NORMALIZATION MODES TO TEST ===
    NORM_MODES = ['default', 'sym_mean_asym_std', 'sym_mean_sym_std', 'no_mean_asym_std', 'no_mean_no_std']

    # === BASE CONFIGS (representative set) ===
    BASE_CONFIGS = [
        {'eps_train': 0.01, 'eps_eval': 0.025, 'k': 16},
        {'eps_train': 0.01, 'eps_eval': 0.03, 'k': 16},
        {'eps_train': 0.03, 'eps_eval': 0.03, 'k': 16},
    ]

    # Default training settings
    builtins._CFG_WEIGHT_EXP = 0
    builtins._CFG_CLIP_GAMMA_EXP = -2

    all_results = {}

    for base_cfg in BASE_CONFIGS:
        for norm_mode in NORM_MODES:
            cfg_name = f"{norm_mode}, eps {base_cfg['eps_train']}/{base_cfg['eps_eval']}, k={base_cfg['k']}"
            print(f"\nRunning config: {cfg_name}")

            # Reset seed for reproducibility
            torch.manual_seed(0)
            np.random.seed(0)

            estimator = DirectELDREstimator3NormTest(
                input_dim=DIM,
                hidden_dim=256,
                num_layers=3,
                time_embed_size=128,
                k=base_cfg['k'],
                eps_train=base_cfg['eps_train'],
                eps_eval=base_cfg['eps_eval'],
                learning_rate=2e-6,
                weight_decay=1e-4,
                num_epochs=1000,
                batch_size=256,
                patience=100000,
                verbose=False,
                integration_steps=100,
                convergence_threshold=1e-8,
                norm_mode=norm_mode,
            )

            estimator._debug_integrate = False
            eldr_estimate = estimator.estimate_eldr(
                samples_base, samples_p0, samples_p1,
                mu0=mu0, Sigma0=Sigma0, mu1=mu1, Sigma1=Sigma1
            )

            # Get diagnostic info about the integrand
            with torch.no_grad():
                t_diag = torch.linspace(base_cfg['eps_eval'], 1 - base_cfg['eps_eval'], 100, device=estimator.device)
                nn_pred = estimator.noiser_network(t_diag)
                mean_diag = estimator._get_mean(t_diag)
                std_diag = estimator._get_std(t_diag)
                gamma_diag = estimator.g(t_diag)

                # Our integrand
                noiser_term = (mean_diag + std_diag * nn_pred) / (gamma_diag ** 3)
                our_integrand = -noiser_term

                # True targets (decentered)
                true_targets = []
                for t_val in t_diag:
                    scores = estimator._compute_decentered_marginal_score(
                        t_val.item(), torch.from_numpy(samples_base).float().to(estimator.device),
                        mu0.to(estimator.device), Sigma0.to(estimator.device),
                        mu1.to(estimator.device), Sigma1.to(estimator.device)
                    )
                    true_targets.append(scores.mean())
                true_targets = torch.stack(true_targets)

            all_results[cfg_name] = {
                'norm_mode': norm_mode,
                'base_cfg': base_cfg,
                'stats': estimator._stats_history,
                'final_est': eldr_estimate,
                'final_err': abs(eldr_estimate - true_eldr),
                't_diag': t_diag.cpu().numpy(),
                'our_integrand': our_integrand.cpu().numpy(),
                'true_targets': true_targets.cpu().numpy(),
                'nn_pred': nn_pred.cpu().numpy(),
                'mean_diag': mean_diag.cpu().numpy(),
                'std_diag': std_diag.cpu().numpy(),
                'gamma_diag': gamma_diag.cpu().numpy(),
            }
            print(f"  Final: {eldr_estimate:.4f} (err={abs(eldr_estimate - true_eldr):.4f})")

    # === PLOT RESULTS ===
    # Create a subplot grid: rows = base configs, cols = metrics
    n_base_cfgs = len(BASE_CONFIGS)
    fig, axes = plt.subplots(n_base_cfgs, 4, figsize=(20, 5 * n_base_cfgs))
    if n_base_cfgs == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.tab10(np.linspace(0, 1, len(NORM_MODES)))
    color_map = {mode: colors[i] for i, mode in enumerate(NORM_MODES)}

    for row_idx, base_cfg in enumerate(BASE_CONFIGS):
        cfg_label = f"eps {base_cfg['eps_train']}/{base_cfg['eps_eval']}, k={base_cfg['k']}"

        for norm_mode in NORM_MODES:
            cfg_name = f"{norm_mode}, {cfg_label}"
            result = all_results[cfg_name]
            color = color_map[norm_mode]

            stats = result['stats']
            iters = [s['iter'] for s in stats]
            losses = [s['loss'] for s in stats]
            eldr_errs = [s['eldr_err'] for s in stats]
            eldr_ests = [s['eldr_est'] for s in stats]

            # Col 0: Loss
            axes[row_idx, 0].plot(iters, losses, label=norm_mode, color=color)
            axes[row_idx, 0].set_title(f'Loss ({cfg_label})')
            axes[row_idx, 0].set_ylabel('Loss')
            axes[row_idx, 0].set_xlabel('Iteration')

            # Col 1: ELDR Error
            axes[row_idx, 1].plot(iters, eldr_errs, label=norm_mode, color=color)
            axes[row_idx, 1].set_title(f'ELDR Error ({cfg_label})')
            axes[row_idx, 1].set_ylabel('|est - true|')
            axes[row_idx, 1].set_xlabel('Iteration')

            # Col 2: ELDR Estimate
            axes[row_idx, 2].plot(iters, eldr_ests, label=norm_mode, color=color)
            axes[row_idx, 2].set_title(f'ELDR Estimate ({cfg_label})')
            axes[row_idx, 2].set_ylabel('Estimate')
            axes[row_idx, 2].set_xlabel('Iteration')

            # Col 3: Integrand
            t_diag = result['t_diag']
            axes[row_idx, 3].plot(t_diag, result['our_integrand'], label=f'{norm_mode}', color=color)

        # Add true ELDR line to col 2
        axes[row_idx, 2].axhline(y=true_eldr, color='k', linestyle='--', label='True ELDR')

        # Add true integrand to col 3 (same for all norm modes)
        first_cfg = f"{NORM_MODES[0]}, {cfg_label}"
        axes[row_idx, 3].plot(
            all_results[first_cfg]['t_diag'],
            all_results[first_cfg]['true_targets'],
            'k--', label='True', linewidth=2
        )
        axes[row_idx, 3].set_title(f'Integrand ({cfg_label})')
        axes[row_idx, 3].set_ylabel('Integrand')
        axes[row_idx, 3].set_xlabel('t')

        # Add legends
        for col in range(4):
            axes[row_idx, col].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('hpo_curves_normtest.png', dpi=150)
    print(f"\nSaved plot to hpo_curves_normtest.png")

    # === SUMMARY TABLE ===
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Norm Mode':<25} {'Config':<30} {'Final Est':>12} {'Error':>12}")
    print("-"*80)

    for base_cfg in BASE_CONFIGS:
        cfg_label = f"eps {base_cfg['eps_train']}/{base_cfg['eps_eval']}, k={base_cfg['k']}"
        for norm_mode in NORM_MODES:
            cfg_name = f"{norm_mode}, {cfg_label}"
            result = all_results[cfg_name]
            print(f"{norm_mode:<25} {cfg_label:<30} {result['final_est']:>12.4f} {result['final_err']:>12.4f}")
        print("-"*80)

    # Find best mode per config
    print("\nBest norm mode per config:")
    for base_cfg in BASE_CONFIGS:
        cfg_label = f"eps {base_cfg['eps_train']}/{base_cfg['eps_eval']}, k={base_cfg['k']}"
        best_mode = None
        best_err = float('inf')
        for norm_mode in NORM_MODES:
            cfg_name = f"{norm_mode}, {cfg_label}"
            if all_results[cfg_name]['final_err'] < best_err:
                best_err = all_results[cfg_name]['final_err']
                best_mode = norm_mode
        print(f"  {cfg_label}: {best_mode} (err={best_err:.4f})")

    # Overall best
    print("\nOverall stats by norm mode:")
    for norm_mode in NORM_MODES:
        errors = [all_results[f"{norm_mode}, eps {cfg['eps_train']}/{cfg['eps_eval']}, k={cfg['k']}"]['final_err']
                  for cfg in BASE_CONFIGS]
        print(f"  {norm_mode:<25}: mean={np.mean(errors):.4f}, min={min(errors):.4f}, max={max(errors):.4f}")
