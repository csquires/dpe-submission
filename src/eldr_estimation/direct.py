"""
Direct ELDR Estimation using Stochastic Interpolants

Estimates E_{p_*}[log(p_0(x)/p_1(x))] by learning the expected time-derivative
of log p_t along a stochastic interpolant path, then integrating over [0, 1].

Reference:
    The estimand is E_{p_*}[integral_1^0 d/dt log p_t(x) dt]
    which equals E_{p_*}[log p_0(x) - log p_1(x)] = E_{p_*}[log(p_0(x)/p_1(x))]
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

    This is inspired by positional encodings in transformers and is commonly used
    in score-based diffusion models to better represent continuous time values.
    """

    def __init__(self, mapping_size: int = 64, scale: float = 10.0):
        """
        Args:
            mapping_size: Number of random frequencies (output dimension = 2 * mapping_size)
            scale: Scale factor for the random frequencies (controls frequency range)
        """
        super().__init__()
        # B is not learnable; it is fixed random frequencies
        # Shape: [1, mapping_size]
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
        # t shape: [batch, 1]
        # self.B shape: [1, mapping_size]
        # x_proj shape: [batch, mapping_size]
        x_proj = 2 * np.pi * t @ self.B
        # Concatenate sin and cos features
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ScoreNetwork(nn.Module):
    """
    Neural network that maps time t to the expected score S_*(t).

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

        # Fourier time embedding
        self.time_embed = FourierTimeEmbedding(
            mapping_size=time_embed_size,
            scale=time_embed_scale
        )

        layers = []
        # Input: Fourier-embedded time (2 * time_embed_size dimensions)
        layers.append(nn.Linear(2 * time_embed_size, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_p))

        # Output: scalar score estimate
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values of shape [batch] or [batch, 1]

        Returns:
            Score estimates of shape [batch]
        """
        # Ensure t has shape [batch, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [batch, 1]

        # Apply Fourier time embedding
        t_embed = self.time_embed(t)  # [batch, 2*time_embed_size]

        return self.net(t_embed).squeeze(-1)  # [batch]

    def _reset_parameters(self) -> None:
        """Reset model parameters to random initialization."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class DirectELDREstimator(ELDREstimator):
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
        l: float = 0.25,
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
            hidden_dim: Hidden layer dimension for score network
            num_layers: Number of layers in score network
            dropout_p: Dropout probability
            time_embed_size: Size of Fourier time embedding (output will be 2*time_embed_size)
            time_embed_scale: Scale parameter for random Fourier frequencies
            k: Parameter for gamma(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t)))
            l: Parameter for importance distribution f(t) = (1 - exp(-l*t)) * (1 - exp(-l*(1-t)))
            eps: Boundary epsilon for importance sampling (t in [eps, 1-eps])
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

        # Set device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device

        # Store interpolant parameters
        # k controls gamma(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t)))
        # l controls importance sampling distribution f(t)
        self.k = k
        self.l = l
        self.eps = eps  # boundary epsilon for t in [eps, 1-eps]

        # Precompute max of f(t) for rejection sampling
        # f(t) = (1 - exp(-l*t)) * (1 - exp(-l*(1-t)))
        # max at t=0.5: f_max = (1 - exp(-l/2))^2
        self._f_max = (1 - np.exp(-self.l / 2)) ** 2

        # Create score network
        self.score_network = ScoreNetwork(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_p=dropout_p,
            time_embed_size=time_embed_size,
            time_embed_scale=time_embed_scale,
        ).to(self.device)

        # Store hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.integration_steps = integration_steps
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.verbose = verbose

        # Sample statistics for h(t) (computed in _compute_h_coefficients)
        self._A = None  # Var(x0)
        self._B = None  # Var(x1)
        self._C = None  # -2 * mean(x0)·mean(x)
        self._D = None  # -2 * mean(x1)·mean(x)
        self._E = None  # Var(x)

    def _f_raw(self, t_raw: np.ndarray) -> np.ndarray:
        """
        Compute the base importance distribution f(t) = (1 - exp(-l*t)) * (1 - exp(-l*(1-t))).
        Used internally for rejection sampling on [0, 1].
        """
        return (1 - np.exp(-self.l * t_raw)) * (1 - np.exp(-self.l * (1 - t_raw)))

    def _f(self, t: np.ndarray) -> np.ndarray:
        """
        Compute the scaled importance distribution for t in [eps, 1-eps].
        Maps t to [0, 1] via t_raw = (t - eps) / (1 - 2*eps), then applies f.
        """
        t_raw = (t - self.eps) / (1 - 2 * self.eps)
        return self._f_raw(t_raw)

    def _f_torch(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the scaled importance distribution for t in [eps, 1-eps].
        Maps t to [0, 1] via t_raw = (t - eps) / (1 - 2*eps), then applies f.
        Uses torch tensors for computing importance weights in training.
        """
        t_raw = (t - self.eps) / (1 - 2 * self.eps)
        return (1 - torch.exp(-self.l * t_raw)) * (1 - torch.exp(-self.l * (1 - t_raw)))

    def _sample_t_importance(self, batch_size: int) -> torch.Tensor:
        """
        Sample t from importance distribution using rejection sampling.

        Strategy: Sample t_raw from f_raw on [0, 1], then transform to [eps, 1-eps]:
            t = eps + (1 - 2*eps) * t_raw

        This guarantees:
        - No sampling outside [eps, 1-eps]
        - Infrequent sampling near eps and 1-eps (since f_raw(0) = f_raw(1) = 0)
        - Most sampling in the middle
        """
        t_raw_samples = []
        while len(t_raw_samples) < batch_size:
            # Sample t_raw from uniform proposal on [0, 1]
            n_needed = batch_size - len(t_raw_samples)
            # Oversample to reduce iterations
            t_raw_proposal = np.random.uniform(0, 1, n_needed * 3)

            # Accept with probability f_raw(t_raw) / f_max
            f_vals = self._f_raw(t_raw_proposal)
            u = np.random.uniform(0, 1, len(t_raw_proposal))
            accept_mask = u < (f_vals / self._f_max)
            t_raw_samples.extend(t_raw_proposal[accept_mask].tolist())

        t_raw_samples = np.array(t_raw_samples[:batch_size])

        # Transform from [0, 1] to [eps, 1-eps]
        t_samples = self.eps + (1 - 2 * self.eps) * t_raw_samples

        return torch.tensor(t_samples, dtype=torch.float32, device=self.device)

    def _compute_h_coefficients(
        self,
        samples_base: torch.Tensor,  # x
        samples_p0: torch.Tensor,    # x0
        samples_p1: torch.Tensor,    # x1
    ) -> None:
        """
        Compute coefficients A, B, C, D, E for h(t).

        h(t) = A*α²(t) + B*β²(t) + C*α(t) + D*β(t) + E

        where α(t) = 1-t, β(t) = t, and:
        - A = Var(x0), summed across dimensions
        - B = Var(x1), summed across dimensions
        - C = -2 * mean(x0)·mean(x), dot product
        - D = -2 * mean(x1)·mean(x), dot product
        - E = Var(x), summed across dimensions
        """
        # Mean vectors
        mean_x = samples_base.mean(dim=0)  # [dim]
        mean_x0 = samples_p0.mean(dim=0)   # [dim]
        mean_x1 = samples_p1.mean(dim=0)   # [dim]

        # Variances (sum across dimensions for scalar)
        self._A = samples_p0.var(dim=0).sum()  # Var(x0)
        self._B = samples_p1.var(dim=0).sum()  # Var(x1)
        self._E = samples_base.var(dim=0).sum()  # Var(x)

        # Cross terms (dot products of means)
        self._C = -2 * (mean_x0 * mean_x).sum()  # -2 * mean(x0)·mean(x)
        self._D = -2 * (mean_x1 * mean_x).sum()  # -2 * mean(x1)·mean(x)

    def mu(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """mu(t) = (1-t)*x0 + t*x1"""
        t_exp = t.unsqueeze(-1)  # [batch, 1]
        return (1 - t_exp) * x0 + t_exp * x1

    def dmudt(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """mu'(t) = x1 - x0 (constant in t)"""
        return x1 - x0

    def g(self, t: torch.Tensor) -> torch.Tensor:
        """gamma(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t)))"""
        return (1 - torch.exp(-self.k * t)) * (1 - torch.exp(-self.k * (1 - t)))

    def dgdt(self, t: torch.Tensor) -> torch.Tensor:
        """
        g'(t) = k*exp(-k*t)*(1-exp(-k*(1-t))) - k*exp(-k*(1-t))*(1-exp(-k*t))
        """
        exp_kt = torch.exp(-self.k * t)
        exp_k1t = torch.exp(-self.k * (1 - t))
        return self.k * exp_kt * (1 - exp_k1t) - self.k * exp_k1t * (1 - exp_kt)

    def h(self, t: torch.Tensor) -> torch.Tensor:
        """h(t) = A*(1-t)² + B*t² + C*(1-t) + D*t + E"""
        alpha = 1 - t  # α(t)
        beta = t       # β(t)
        return (self._A * alpha**2 + self._B * beta**2 +
                self._C * alpha + self._D * beta + self._E)

    def dhdt(self, t: torch.Tensor) -> torch.Tensor:
        """h'(t) = -2*A*(1-t) + 2*B*t - C + D"""
        alpha = 1 - t
        beta = t
        return -2 * self._A * alpha + 2 * self._B * beta - self._C + self._D

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """gamma(t) = g(t) * h(t)"""
        # return self.g(t) * self.h(t)
        return self.g(t)

    def dgammadt(self, t: torch.Tensor) -> torch.Tensor:
        """gamma'(t) = g'(t)*h(t) + g(t)*h'(t) (product rule)"""
        # return self.dgdt(t) * self.h(t) + self.g(t) * self.dhdt(t)
        return self.dgdt(t)

    def score(
        self,
        t: torch.Tensor,      # [batch]
        x: torch.Tensor,      # [batch, dim]
        x0: torch.Tensor,     # [batch, dim]
        x1: torch.Tensor,     # [batch, dim]
    ) -> torch.Tensor:
        """
        Compute conditioned score s(x, t; x0, x1).

        Formula:
            s = -dim*g'/g + ||r||^2*g'/g^3 + r^T*mu'/g^2

        where r = x - mu(t)

        Returns:
            Scalar score values of shape [batch]
        """
        dim = x0.shape[-1]
        t_exp = t.unsqueeze(-1)  # [batch, 1]

        gamma_t = self.gamma(t_exp)          # [batch, 1]
        gamma_prime_t = self.dgammadt(t_exp)  # [batch, 1]
        mu_t = self.mu(t, x0, x1)            # [batch, dim]
        mu_prime = self.dmudt(x0, x1)        # [batch, dim]

        # r = x - mu(t) (not divided by g)
        r = x - mu_t  # [batch, dim]
        r_norm_sq = (r ** 2).sum(dim=-1, keepdim=True)  # [batch, 1]

        # r^T * mu'(t)
        r_dot_mu_prime = (r * mu_prime).sum(dim=-1, keepdim=True)  # [batch, 1]

        # s = -dim*g'/g + ||r||^2*g'/g^3 + r^T*mu'/g^2
        term1 = -dim * (gamma_prime_t / gamma_t)  # [batch, 1]
        term2 = r_norm_sq * (gamma_prime_t / (gamma_t ** 3))  # [batch, 1]
        term3 = r_dot_mu_prime / (gamma_t ** 2)  # [batch, 1]
        score = term1 + term2 + term3

        return score.squeeze(-1)  # [batch]

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

        # Coefficients
        alpha = 1 - t
        beta = t
        alpha_prime = -1.0
        beta_prime = 1.0

        # gamma(t) = g(t) and gamma'(t) = g'(t)
        exp_kt = np.exp(-self.k * t)
        exp_k1t = np.exp(-self.k * (1 - t))
        gamma = (1 - exp_kt) * (1 - exp_k1t)
        gamma_prime = self.k * exp_kt * (1 - exp_k1t) - self.k * exp_k1t * (1 - exp_kt)

        # μ_t and μ'_t
        mu_t = alpha * mu0 + beta * mu1  # [dim]
        mu_prime_t = alpha_prime * mu0 + beta_prime * mu1  # = μ₁ - μ₀

        # Σ_t and Σ'_t
        I_d = torch.eye(dim, dtype=Sigma0.dtype, device=Sigma0.device)
        Sigma_t = alpha**2 * Sigma0 + beta**2 * Sigma1 + gamma**2 * I_d
        Sigma_prime_t = 2*alpha*alpha_prime * Sigma0 + 2*beta*beta_prime * Sigma1 + 2*gamma*gamma_prime * I_d

        # Σ_t⁻¹
        Sigma_t_inv = torch.linalg.inv(Sigma_t)  # [dim, dim]

        # r = x - μ_t
        r = x - mu_t  # [batch, dim]

        # Term 1: -1/2 tr(Σ_t⁻¹ Σ'_t)
        term1 = -0.5 * torch.trace(Sigma_t_inv @ Sigma_prime_t)  # scalar

        # Term 2: (x-μ_t)ᵀ Σ_t⁻¹ μ'_t = r @ Σ_t⁻¹ @ μ'_t
        term2 = r @ Sigma_t_inv @ mu_prime_t  # [batch]

        # Term 3: 1/2 (x-μ_t)ᵀ Σ_t⁻¹ Σ'_t Σ_t⁻¹ (x-μ_t)
        M = Sigma_t_inv @ Sigma_prime_t @ Sigma_t_inv  # [dim, dim]
        term3 = 0.5 * torch.einsum('bi,ij,bj->b', r, M, r)  # [batch]

        return term1 + term2 + term3  # [batch]

    def _fit_score_network(
        self,
        samples_base: torch.Tensor,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        # Optional true distribution parameters for sanity check
        mu0: Optional[torch.Tensor] = None,
        Sigma0: Optional[torch.Tensor] = None,
        mu1: Optional[torch.Tensor] = None,
        Sigma1: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Train the score network using weighted MSE loss.

        Loss = E_{t~f/Z, x0~p0, x1~p1}[g * ||NN(t) - E_{x~p_*}[s(x,t;x0,x1)]||^2]

        where:
        - t is sampled from importance distribution f(t)/Z via rejection sampling
        - x0, x1 are sampled from p0, p1 respectively
        - The target is the AVERAGE score over all x from base distribution p_*
        - weight = g(t) (Z is constant, cancels in optimization)

        This differs from the per-sample approach where the network learns individual
        scores s(x,t;x0,x1). Here, for each sampled (t, x0, x1) tuple, we first
        compute the mean score over all x ~ p_*, then train the network to predict
        this averaged value.
        """
        self.score_network._reset_parameters()
        self.score_network.train()

        # Move data to device
        samples_base = samples_base.to(self.device)
        samples_p0 = samples_p0.to(self.device)
        samples_p1 = samples_p1.to(self.device)

        # Compute h(t) coefficients from sample statistics
        self._compute_h_coefficients(samples_base, samples_p0, samples_p1)

        n_base = samples_base.shape[0]
        n_p0 = samples_p0.shape[0]
        n_p1 = samples_p1.shape[0]

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.score_network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Precompute t values for validation metrics (used for logging avg_nn)
        t_eval = torch.linspace(self.eps, 1 - self.eps, self.integration_steps, device=self.device)

        # Compute sanity check targets if true distribution parameters provided
        sanity_check_targets = None
        lambda_weights = None
        if mu0 is not None and Sigma0 is not None and mu1 is not None and Sigma1 is not None:
            # Move true params to device
            mu0 = mu0.to(self.device)
            Sigma0 = Sigma0.to(self.device)
            mu1 = mu1.to(self.device)
            Sigma1 = Sigma1.to(self.device)

            with torch.no_grad():
                sanity_check_targets = []
                for i, t_val in enumerate(t_eval):
                    scores = self._compute_marginal_score(
                        t_val.item(), samples_base, mu0, Sigma0, mu1, Sigma1
                    )
                    sanity_check_targets.append(scores.mean())
                sanity_check_targets = torch.stack(sanity_check_targets)  # [n_t]
                print(f'Sanity Check: ELDR {-sanity_check_targets.mean()}')

                # Precompute lambda(t) = 1 / (f(t) * g(t)) for weighted error
                t_eval_exp = t_eval.unsqueeze(-1)  # [n_t, 1]
                g_eval = self.g(t_eval_exp).squeeze(-1)  # [n_t]
                f_eval = self._f_torch(t_eval)  # [n_t]
                # lambda_weights = 1.0 / (f_eval * g_eval)  # [n_t]
                lambda_weights = 1.0

        best_loss = float('inf')
        patience_counter = 0
        global_iter = 0  # Global iteration counter across all epochs

        # Number of iterations per epoch (based on smaller dataset)
        num_iters = max(1, min(n_p0, n_p1) // self.batch_size)

        for epoch in range(self.num_epochs):
            # Shuffle data once per epoch
            perm_p0 = torch.randperm(n_p0, device=self.device)
            perm_p1 = torch.randperm(n_p1, device=self.device)
            shuffled_p0 = samples_p0[perm_p0]
            shuffled_p1 = samples_p1[perm_p1]

            for iter_idx in range(num_iters):
                global_iter += 1

                # Index consecutive samples from shuffled data
                start = iter_idx * self.batch_size
                end = start + self.batch_size
                x0 = shuffled_p0[start:end]  # [batch, dim]
                x1 = shuffled_p1[start:end]  # [batch, dim]

                # Sample t from importance sampling distribution
                t = self._sample_t_importance(self.batch_size)

                # Compute AVERAGED score across ALL x from base distribution
                # For each t value, we compute the mean score over all x ~ p_*
                # while using the current x0, x1 pair for each training step
                # (This balances computational efficiency with stable targets)

                # For each (t, x0, x1), compute average score over ALL x from base distribution
                avg_scores = []
                for i in range(self.batch_size):
                    # Expand t, x0, x1 to match all base samples
                    t_i = t[i].expand(n_base)  # [n_base]
                    x0_i = x0[i].unsqueeze(0).expand(n_base, -1)  # [n_base, dim]
                    x1_i = x1[i].unsqueeze(0).expand(n_base, -1)  # [n_base, dim]

                    # Compute score for all base samples
                    scores_i = self.score(t_i, samples_base, x0_i, x1_i)  # [n_base]

                    # Average over all base samples
                    avg_scores.append(scores_i.mean())

                avg_scores = torch.stack(avg_scores)  # [batch_size]

                # Forward pass
                predictions = self.score_network(t)  # [batch]

                # MSE loss per sample (now comparing to averaged scores)
                mse_per_sample = (predictions - avg_scores.detach()) ** 2

                # Compute importance sampling weight: 1 / (f(t) * g(t))
                t_exp = t.unsqueeze(-1)  # [batch, 1]
                g_t = self.g(t_exp).squeeze(-1)  # [batch]
                f_t = self._f_torch(t)  # [batch]
                weights = 1.0 / (f_t)

                # Weighted loss
                loss = (mse_per_sample * weights).mean()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.score_network.parameters(), max_norm=10.0)
                optimizer.step()

                loss_val = loss.item()

                # Check if this is best loss
                is_best = loss_val < best_loss

                # Update best loss and patience
                if is_best:
                    best_loss = loss_val
                    patience_counter = 0
                elif loss_val >= best_loss - self.convergence_threshold:
                    patience_counter += 1

                # Determine if logging should happen: best loss OR every 100 iterations
                should_log = is_best or (global_iter % 100 == 0)

                if should_log:
                    # === LOGGING ROUTINE ===
                    with torch.no_grad():
                        # Compute NN(t) for all t
                        nn_predictions = self.score_network(t_eval)  # [n_t]

                        # Sanity check errors (only if targets available)
                        weighted_error = None
                        unweighted_error = None
                        if sanity_check_targets is not None:
                            sq_errors = (nn_predictions - sanity_check_targets) ** 2
                            weighted_error = (lambda_weights * sq_errors).mean().item()
                            unweighted_error = sq_errors.mean().item()

                        # Mean NN(t)
                        avg_nn = nn_predictions.mean().item()

                    if self.verbose:
                        log_msg = f"[Iter {global_iter}] loss={loss_val:.6f}"
                        if weighted_error is not None:
                            log_msg += f", weighted_err={weighted_error:.6f}, unweighted_err={unweighted_error:.6f}"
                        log_msg += f", avg_nn={avg_nn:.6f}"
                        if is_best:
                            log_msg += " *best*"
                        print(log_msg)

                # Check early stopping
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Converged at iteration {global_iter}")
                    return

    def _integrate_score(self) -> float:
        """
        Integrate the learned score network from t=eps to t=1-eps.

        Uses Simpson's rule for numerical integration.

        Returns:
            Integral value (scalar)
        """
        self.score_network.eval()

        # Integration points (Simpson's rule requires odd number of points)
        n_points = self.integration_steps
        if n_points % 2 == 0:
            n_points += 1

        # Integrate over [eps, 1-eps] to match the importance sampling range
        t_vals = torch.linspace(self.eps, 1 - self.eps, n_points, device=self.device)

        with torch.no_grad():
            scores = self.score_network(t_vals).cpu().numpy()

        t_np = t_vals.cpu().numpy()

        # Simpson's rule integration
        h = (t_np[-1] - t_np[0]) / (n_points - 1)
        integral = scores[0] + scores[-1]
        for i in range(1, n_points - 1):
            if i % 2 == 0:
                integral += 2 * scores[i]
            else:
                integral += 4 * scores[i]
        integral *= h / 3

        return float(integral)

    def estimate_eldr(
        self,
        samples_base: np.ndarray,  # [n_base, dim], samples from p_*
        samples_p0: np.ndarray,    # [n_p0, dim]
        samples_p1: np.ndarray,    # [n_p1, dim]
        # Optional true distribution parameters for sanity check
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
        # Convert to tensors
        samples_base_t = torch.from_numpy(samples_base).float()
        samples_p0_t = torch.from_numpy(samples_p0).float()
        samples_p1_t = torch.from_numpy(samples_p1).float()

        # Train the score network
        self._fit_score_network(
            samples_base_t, samples_p0_t, samples_p1_t,
            mu0=mu0, Sigma0=Sigma0, mu1=mu1, Sigma1=Sigma1
        )

        # Integrate to get the estimate
        # The integral from 0->1 gives E[log p_1(x) - log p_0(x)] = -ELDR
        integral = self._integrate_score()

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
    # Create estimator with recommended hyperparameters
    # gamma(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t)))
    # At t=0.5: gamma(0.5) = (1 - exp(-k/2))^2 ≈ 0.98 for k=8
    estimator = DirectELDREstimator(
        input_dim=DIM,
        # Network architecture (keep defaults)
        hidden_dim=64,
        num_layers=3,
        time_embed_size=64,
        # Interpolant parameters
        k=8.0,        # gamma(0.5) ≈ 0.98
        l=2.0,        # importance sampling
        eps=0.01,      # boundary epsilon
        # Training parameters
        learning_rate=1e-6,
        weight_decay=1e-4,
        num_epochs=2000,     # Reduced from 2000
        batch_size=NSAMPLES//16,     # Reduced to allow 4 iters/epoch
        # Convergence
        patience=500,       # In total iterations
        verbose=True,
        integration_steps=NSAMPLES*2
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
