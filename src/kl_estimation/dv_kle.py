"""
Donsker-Varadhan KL Divergence Estimator

Ported from deep-preemptive-exploration/src/fragments/donsker_varadhan/dvkl.py
Adapted to conform to the KLEstimator API defined in base.py
"""

import torch
import torch.nn as nn
from typing import Optional

from src.kl_estimation.base import KLEstimator


class DVCriticNetwork(nn.Module):
    """
    Critic network for Donsker-Varadhan KL estimation.

    Outputs unbounded real values (not logits) for variational optimization.
    Similar architecture to DefaultBinaryClassifier but without final sigmoid.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout_p: float = 0.1,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_p))

        # Final layer outputs unbounded real values
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)

        Returns:
            Critic values of shape (batch_size,)
        """
        return self.net(x).squeeze(-1)

    def _reset_parameters(self) -> None:
        """Reset model parameters to random initialization."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class DV_KLE(KLEstimator):
    """
    Donsker-Varadhan KL Divergence Estimator conforming to KLEstimator API.

    Uses the variational representation:
        KL(p0 || p1) = sup_f E_p0[f(x)] - log E_p1[exp(f(x))]

    The critic network f is trained to maximize this lower bound.

    Usage:
        estimator = DV_KLE(input_dim=10)
        kl_estimate = estimator.estimate_kl(samples_p0, samples_p1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout_p: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 1000,
        batch_size: Optional[int] = None,
        noise_std: float = 0.01,
        convergence_threshold: float = 1e-3,
        patience: int = 50,
        verbose: bool = False,
        device: Optional[str] = None,
    ):
        """
        Args:
            input_dim: Dimensionality of input samples
            hidden_dim: Hidden layer dimension for critic network
            num_layers: Number of layers in critic network
            dropout_p: Dropout probability
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            num_epochs: Maximum training epochs
            batch_size: Batch size for training (None = use all data)
            noise_std: Standard deviation of noise added during training
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

        # Create critic network
        self.critic = DVCriticNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_p=dropout_p,
        ).to(self.device)

        # Store hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.verbose = verbose

        # Cache for partition function
        self._cached_log_partition: Optional[torch.Tensor] = None

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise for regularization during training."""
        if self.critic.training and self.noise_std > 0:
            return x + torch.randn_like(x) * self.noise_std
        return x

    def _compute_dv_loss(
        self,
        numerator_critics: torch.Tensor,
        denominator_critics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Donsker-Varadhan variational loss.

        L = -E_p0[f(x)] + log E_p1[exp(f(x))]

        Uses log-sum-exp trick for numerical stability.
        """
        # First term: -E_p0[f]
        first_term = -numerator_critics.mean()

        # Second term: log E_p1[exp(f)] using log-sum-exp trick
        # log(mean(exp(f))) = logsumexp(f) - log(N)
        n_samples = denominator_critics.shape[0]
        second_term = torch.logsumexp(denominator_critics, dim=0) - torch.log(
            torch.tensor(n_samples, dtype=denominator_critics.dtype, device=denominator_critics.device)
        )

        return first_term + second_term

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """
        Fit the critic network using Donsker-Varadhan variational optimization.

        Args:
            samples_p0: Samples from distribution p0, shape (n_samples, input_dim)
            samples_p1: Samples from distribution p1, shape (n_samples, input_dim)
        """
        # Reset and move to device
        self.critic._reset_parameters()
        self.critic.train()
        samples_p0 = samples_p0.to(self.device)
        samples_p1 = samples_p1.to(self.device)

        n_p0 = samples_p0.shape[0]
        n_p1 = samples_p1.shape[0]

        batch_size = self.batch_size if self.batch_size is not None else min(n_p0, n_p1)

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Sample batches
            p0_idx = torch.randperm(n_p0, device=self.device)[:batch_size]
            p1_idx = torch.randperm(n_p1, device=self.device)[:batch_size]

            p0_batch = samples_p0[p0_idx]
            p1_batch = samples_p1[p1_idx]

            # Add noise
            p0_batch = self._add_noise(p0_batch)
            p1_batch = self._add_noise(p1_batch)

            # Forward pass
            p0_critics = self.critic(p0_batch)
            p1_critics = self.critic(p1_batch)

            # Compute DV loss
            loss = self._compute_dv_loss(p0_critics, p1_critics)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            optimizer.step()

            # Check convergence
            loss_val = loss.item()
            if loss_val < best_loss - self.convergence_threshold:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Converged at epoch {epoch + 1}")
                break

            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss_val:.6f}")

        # Cache partition function after training
        self.critic.eval()
        with torch.no_grad():
            critic_values = self.critic(samples_p1)
            n_samples = critic_values.shape[0]
            log_partition = torch.logsumexp(critic_values, dim=0) - torch.log(
                torch.tensor(n_samples, dtype=critic_values.dtype, device=critic_values.device)
            )
            self._cached_log_partition = log_partition.detach()

    def estimate_kl(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> float:
        """
        Estimate KL divergence KL(p0 || p1) using Donsker-Varadhan representation.

        Args:
            samples_p0: Samples from distribution p0, shape (n_samples, input_dim)
            samples_p1: Samples from distribution p1, shape (n_samples, input_dim)

        Returns:
            Scalar estimate of KL(p0 || p1)
        """
        # Fit the critic
        self.fit(samples_p0, samples_p1)

        # Estimate KL using fitted critic
        # KL(p0 || p1) = E_p0[f(x)] - log E_p1[exp(f(x))]
        self.critic.eval()
        samples_p0 = samples_p0.to(self.device)

        with torch.no_grad():
            critic_values = self.critic(samples_p0)
            first_term = critic_values.mean()

            # Use cached partition function
            if self._cached_log_partition is not None:
                kl_estimate = first_term - self._cached_log_partition
            else:
                # Fallback (shouldn't happen after fit())
                kl_estimate = first_term

        return kl_estimate.item()
