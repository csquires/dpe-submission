"""
Direct ELDR Estimation using Stochastic Interpolants (Version 4)

Ports the spatial velocity denoiser approach to ELDR estimation using
separate networks for velocity (b) and denoiser (eta).

Training follows the spatial_velo_denoiser2.py recipe:
- Sample z ~ N(0, I) fresh noise
- Compute noisy interpolant x_t = I_t + gamma_t * z
- Train networks on x_t (not raw x from base distribution)
- Velocity target: (x1 - x0) + gamma'_t * z
- Denoiser target: z (the actual sampled noise)

From direct3.py:
- Fourier time embedding for better time representation
- ELDR estimation by integrating time score over samples from p_*
"""

from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        x_proj = 2 * np.pi * t @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with Fourier time embedding for estimating vector fields.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = None,
        time_embed_size: int = 64,
        time_embed_scale: float = 10.0,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.time_embed = FourierTimeEmbedding(
            mapping_size=time_embed_size,
            scale=time_embed_scale
        )

        # Input: 2 * time_embed_size (from Fourier) + input_dim (spatial)
        self.net = nn.Sequential(
            nn.Linear(2 * time_embed_size + input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
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
        t_embed = self.time_embed(t)
        tx = torch.cat([t_embed, x], dim=-1)
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
    ELDR estimator using spatial velocity-denoiser approach.

    Training follows spatial_velo_denoiser2.py exactly:
    - Sample z ~ N(0, I) fresh noise
    - Compute noisy interpolant x_t = I_t + gamma_t * z
    - Train velocity network b on x_t with target (x1 - x0) + gamma'_t * z
    - Train denoiser network eta on x_t with target z

    ELDR estimation:
    - Compute time score: dt_log_rho = -div(b) + b.eta/gamma
    - Integrate over t for each sample x from p_*
    - Return negative mean integral as ELDR estimate
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_epochs: int = 600,
        batch_size: int = 512,
        lr: float = 2e-3,
        k: float = 24.0,
        eps_train: float = 9e-4,  # Training eps for t sampling bounds and grad clipping
        eps_eval: float = 0.01,   # Integration eps for bounds [eps_eval, 1-eps_eval]
        loss_weight_exp: float = 0,  # Loss weighting exponent: 0 = disabled, >0 = 1/(gamma+eps)^exp
        integration_steps: int = 3000,
        integration_type: Literal['1', '2', '3'] = '2',
        verbose: bool = False,
        log_every: int = 100,
        device: Optional[str] = None,
        antithetic: bool = True,
        time_embed_size: int = 64,
        time_embed_scale: float = 10.0,
    ):
        """
        Args:
            input_dim: Dimensionality of input samples
            hidden_dim: Hidden layer dimension for networks
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            k: Parameter for gamma(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t)))
            eps_train: Training epsilon for t sampling bounds [eps_train, 1-eps_train] and grad clipping
            eps_eval: Integration epsilon for bounds [eps_eval, 1-eps_eval]
            loss_weight_exp: Loss weighting exponent: 0 = disabled, >0 = 1/(gamma+eps)^exp
            integration_steps: Number of points for numerical integration
            integration_type: '1' for mean, '2' for trapz, '3' for Simpson
            verbose: Print training progress
            log_every: Print every N epochs when verbose
            device: Device to use ('cuda', 'cpu', or None for auto)
            antithetic: Use antithetic sampling for variance reduction
            time_embed_size: Size of Fourier time embedding (output = 2 * time_embed_size)
            time_embed_scale: Scale for random Fourier frequencies
        """
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.k = k
        self.eps_train = eps_train
        self.eps_eval = eps_eval
        self.loss_weight_exp = loss_weight_exp
        self.integration_steps = integration_steps
        self.integration_type = integration_type
        self.verbose = verbose
        self.log_every = log_every
        self.antithetic = antithetic
        self.time_embed_size = time_embed_size
        self.time_embed_scale = time_embed_scale

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.net_b = None
        self.net_eta = None

    def init_model(self) -> None:
        """Initialize two completely separate networks for b and eta."""
        self.net_b = MLP(
            self.input_dim,
            self.hidden_dim,
            output_dim=self.input_dim,
            time_embed_size=self.time_embed_size,
            time_embed_scale=self.time_embed_scale,
        ).to(self.device)
        self.net_eta = MLP(
            self.input_dim,
            self.hidden_dim,
            output_dim=self.input_dim,
            time_embed_size=self.time_embed_size,
            time_embed_scale=self.time_embed_scale,
        ).to(self.device)

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

        The network predicts scaled eta (eta / v(t)), so we need to de-scale:
        eta = eta_pred * v(t)

        Args:
            t: Time tensor [batch, 1]
            x: Data tensor [batch, dim]
            gamma_t: Value of gamma at time t (scalar or [batch, 1])

        Returns:
            time_score: [batch, 1] estimate of the time score
        """
        x = x.clone().requires_grad_(True)

        # Get v(t) for de-scaling (uses clamped t internally)
        t_flat = t.squeeze(-1) if t.dim() > 1 else t
        v_t = self._interpolate_v(t_flat).unsqueeze(-1)  # [batch, 1]

        # Run separate forward passes
        b_pred = self.net_b(t, x)
        eta_scaled_pred = self.net_eta(t, x)

        # De-scale: eta = eta_scaled * v(t)
        eta_pred = eta_scaled_pred * v_t

        # Exact divergence (trace of Jacobian)
        div_b = compute_divergence(b_pred, x)

        # Scalar terms
        b_dot_eta = (b_pred * eta_pred).sum(dim=-1, keepdim=True)

        return -div_b.view(-1, 1) + b_dot_eta / gamma_t

    def _compute_time_score_single(self, t_scalar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute time score for a single time value across all samples.

        Time score: dt_log_rho = -div(b) + b.eta/gamma

        Args:
            t_scalar: Single time value (scalar tensor)
            x: Sample points [n_samples, dim]
        Returns:
            Time scores [n_samples]
        """
        n_samples = x.shape[0]
        t_batch = t_scalar.expand(n_samples, 1)

        gamma_t = self.gamma(t_scalar)

        # Compute b and eta
        b_pred = self.net_b(t_batch, x)
        eta_pred = self.net_eta(t_batch, x)

        # Compute divergence using jacrev + vmap over samples
        def b_single(x_single):
            return self.net_b(t_scalar.view(1, 1), x_single.unsqueeze(0)).squeeze(0)

        def jac_trace(x_single):
            jac = torch.func.jacrev(b_single)(x_single)  # [dim, dim]
            return torch.trace(jac)

        div_b = torch.vmap(jac_trace)(x)  # [n_samples]

        b_dot_eta = (b_pred * eta_pred).sum(dim=-1)

        return -div_b + b_dot_eta / gamma_t  # [n_samples]

    def _precompute_std_grid(
        self,
        samples_base: torch.Tensor,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        n_grid: int = 100,
    ) -> None:
        """
        Precompute std of denoiser target at grid of t values.

        Target = (x - I_t) / gamma_t where x ~ p_*, x0 ~ p0, x1 ~ p1
        We compute std of this target for each t in the grid.
        """
        self._v_grid_t = torch.linspace(0, 1, n_grid, device=self.device)
        self._std_grid_vals = []

        n_base = samples_base.shape[0]
        n_p0 = samples_p0.shape[0]
        n_p1 = samples_p1.shape[0]
        n_samples = min(1000, n_base, n_p0, n_p1)  # Use subset for efficiency

        with torch.no_grad():
            for t_val in self._v_grid_t:
                # Sample x from p_*, x0 from p0, x1 from p1
                base_idx = torch.randint(0, n_base, (n_samples,), device=self.device)
                p0_idx = torch.randint(0, n_p0, (n_samples,), device=self.device)
                p1_idx = torch.randint(0, n_p1, (n_samples,), device=self.device)

                x = samples_base[base_idx]
                x0 = samples_p0[p0_idx]
                x1 = samples_p1[p1_idx]

                # Compute I_t and gamma_t
                I_t = (1 - t_val) * x0 + t_val * x1
                gamma_t = self.gamma(t_val)

                # Target = (x - I_t) / gamma_t
                # Use clamped gamma for numerical stability
                gamma_clamped = max(gamma_t.item(), 1e-6)
                target = (x - I_t) / gamma_clamped

                # Compute std across samples (per dimension, then mean)
                std_t = target.std(dim=0).mean()
                self._std_grid_vals.append(std_t)

        self._std_grid_vals = torch.stack(self._std_grid_vals)

    def _interpolate_v(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolate v(t) (std scaling) using clamped t.

        Args:
            t: Time values [batch] or scalar

        Returns:
            Interpolated std values matching input shape
        """
        # Always clamp t for v(t) to ensure training/eval consistency
        t_clamped = torch.clamp(t, self.eps_train, 1 - self.eps_train)

        grid_range = 1
        idx_float = t_clamped / grid_range * (len(self._v_grid_t) - 1)
        idx_low = torch.floor(idx_float).long().clamp(0, len(self._v_grid_t) - 1)
        idx_high = torch.clamp(idx_low + 1, max=len(self._v_grid_t) - 1)
        frac = idx_float - idx_low.float()

        std_low = self._std_grid_vals[idx_low]
        std_high = self._std_grid_vals[idx_high]
        return std_low + frac * (std_high - std_low)

    def fit(
        self,
        samples_base: torch.Tensor,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
    ) -> None:
        """
        Train the velocity and denoiser networks.

        Training uses samples from p_* for denoiser (like direct3):
        - Sample x from p_* (samples_base)
        - Sample x0 from p0, x1 from p1
        - Compute I_t = (1-t)*x0 + t*x1
        - Denoiser target: (x - I_t) / gamma_t

        Args:
            samples_base: Samples from base distribution p_* [n_base, dim]
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

        # Precompute std grid for v(t) scaling
        self._precompute_std_grid(samples_base, samples_p0, samples_p1)

        if self.verbose:
            print(f"[DirectELDREstimator4] Starting Sequential Training")
            print(f"[DirectELDREstimator4] gamma range: [{self.gamma(torch.tensor(self.eps_train)).item():.4f}, {self.gamma(torch.tensor(0.5)).item():.4f}]")

        # ==========================================
        # PHASE 1: Train Velocity Network (b)
        # ==========================================
        if self.verbose:
            print(f"\n[Phase 1/2] Training Velocity Network (b) for {self.n_epochs} epochs...")

        self.net_b.train()
        self.net_eta.eval()
        optimizer_b = optim.Adam(self.net_b.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        for epoch in range(self.n_epochs):
            # Sample batches
            p0_idx = torch.randint(0, n_p0, (self.batch_size,), device=self.device)
            p1_idx = torch.randint(0, n_p1, (self.batch_size,), device=self.device)
            x0 = samples_p0[p0_idx]
            x1 = samples_p1[p1_idx]

            # Sample time uniformly from [0, 1] (full range, like direct3)
            t = torch.rand(self.batch_size, device=self.device)
            t_batch = t.unsqueeze(-1)  # [B, 1]

            # Sample noise (independent per sample)
            z = torch.randn_like(x0)

            # Compute gamma and gamma' for each t (unclamped for interpolant)
            gamma_t = self.gamma(t).unsqueeze(-1)  # [B, 1]
            gamma_prime_t = self.dgamma_dt(t).unsqueeze(-1)  # [B, 1]

            # Interpolant mean
            I_t = (1 - t_batch) * x0 + t_batch * x1
            dtIt = x1 - x0  # derivative of I_t w.r.t. t

            if self.antithetic:
                # Antithetic sampling: x_t+ and x_t-
                x_t_plus = I_t + gamma_t * z
                x_t_minus = I_t - gamma_t * z

                b_plus = self.net_b(t_batch, x_t_plus)
                b_minus = self.net_b(t_batch, x_t_minus)

                # Velocity loss with antithetic sampling
                b_norm_sq_plus = (b_plus ** 2).sum(dim=-1)
                b_norm_sq_minus = (b_minus ** 2).sum(dim=-1)
                target_dot_b_plus = ((dtIt + gamma_prime_t * z) * b_plus).sum(dim=-1)
                target_dot_b_minus = ((dtIt - gamma_prime_t * z) * b_minus).sum(dim=-1)
                per_sample_loss_b = (0.25 * b_norm_sq_plus - 0.5 * target_dot_b_plus
                        + 0.25 * b_norm_sq_minus - 0.5 * target_dot_b_minus)
            else:
                # Standard (non-antithetic) training
                x_t = I_t + gamma_t * z
                b_pred = self.net_b(t_batch, x_t)

                # Velocity loss: 0.5*||b||² - ((x1 - x0)+gamma_t'z)·b
                b_norm_sq = (b_pred ** 2).sum(dim=-1)
                target_dot_b = ((dtIt + gamma_prime_t * z) * b_pred).sum(dim=-1)
                per_sample_loss_b = (0.5 * b_norm_sq - target_dot_b)

            # Apply loss weighting if enabled
            if self.loss_weight_exp != 0:
                t_clamped = torch.clamp(t, self.eps_train, 1 - self.eps_train)
                gamma_clamped = self.gamma(t_clamped)
                weights = 1.0 / (gamma_clamped + self.eps_train) ** self.loss_weight_exp
                loss_b = (per_sample_loss_b * weights).mean()
            else:
                loss_b = per_sample_loss_b.mean()

            optimizer_b.zero_grad()
            loss_b.backward()

            # Gamma-based gradient clipping
            t_clamped = torch.clamp(t, self.eps_train, 1 - self.eps_train)
            gamma_clamped = self.gamma(t_clamped)
            gamma_min = gamma_clamped.min().item()
            clip_norm = 10.0 * (gamma_min + self.eps_train) ** (-2)
            torch.nn.utils.clip_grad_norm_(self.net_b.parameters(), max_norm=clip_norm)

            optimizer_b.step()

            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"  [Epoch {epoch+1}] loss_b={loss_b.item():.4f}")

        # ==========================================
        # PHASE 2: Train Denoiser Network (eta)
        # Uses samples from p_* with v(t) scaling (like direct3)
        # ==========================================
        if self.verbose:
            print(f"\n[Phase 2/2] Training Denoiser Network (eta) for {self.n_epochs} epochs...")
            print(f"  v(t) range: [{self._std_grid_vals.min().item():.4f}, {self._std_grid_vals.max().item():.4f}]")

        self.net_b.eval()
        self.net_eta.train()
        optimizer_eta = optim.Adam(self.net_eta.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        for epoch in range(self.n_epochs):
            # Sample from all three distributions
            base_idx = torch.randint(0, n_base, (self.batch_size,), device=self.device)
            p0_idx = torch.randint(0, n_p0, (self.batch_size,), device=self.device)
            p1_idx = torch.randint(0, n_p1, (self.batch_size,), device=self.device)
            x = samples_base[base_idx]  # Sample from p_*
            x0 = samples_p0[p0_idx]
            x1 = samples_p1[p1_idx]

            # Sample time uniformly from [0, 1] (full range, like direct3)
            t = torch.rand(self.batch_size, device=self.device)
            t_batch = t.unsqueeze(-1)  # [B, 1]

            # Compute gamma for each t (unclamped for interpolant)
            gamma_t = self.gamma(t).unsqueeze(-1)  # [B, 1]

            # Interpolant mean I_t = (1-t)*x0 + t*x1
            I_t = (1 - t_batch) * x0 + t_batch * x1

            # Target = (x - I_t) / gamma_t
            # This is the denoiser target when using samples from p_*
            target_eta = (x - I_t) / (gamma_t + 1e-8)

            # Scale target by v(t) for stability
            v_t = self._interpolate_v(t).unsqueeze(-1)  # [B, 1]
            scaled_target = target_eta / (v_t + 1e-8)

            # Network predicts scaled eta
            eta_pred = self.net_eta(t_batch, x)

            # Denoiser loss: 0.5*||eta||² - scaled_target·eta
            eta_norm_sq = (eta_pred ** 2).sum(dim=-1)
            target_dot_eta = (scaled_target * eta_pred).sum(dim=-1)
            per_sample_loss_eta = (0.5 * eta_norm_sq - target_dot_eta)

            # Apply loss weighting if enabled (uses clamped gamma)
            if self.loss_weight_exp != 0:
                t_clamped = torch.clamp(t, self.eps_train, 1 - self.eps_train)
                gamma_clamped = self.gamma(t_clamped)
                weights = 1.0 / (gamma_clamped + self.eps_train) ** self.loss_weight_exp
                loss_eta = (per_sample_loss_eta * weights).mean()
            else:
                loss_eta = per_sample_loss_eta.mean()

            optimizer_eta.zero_grad()
            loss_eta.backward()

            # Gamma-based gradient clipping (uses clamped gamma)
            t_clamped = torch.clamp(t, self.eps_train, 1 - self.eps_train)
            gamma_clamped = self.gamma(t_clamped)
            gamma_min = gamma_clamped.min().item()
            clip_norm = 10.0 * (gamma_min + self.eps_train) ** (-2)
            torch.nn.utils.clip_grad_norm_(self.net_eta.parameters(), max_norm=clip_norm)

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

        # Integration grid (uses eps_eval for bounds)
        n_points = self.integration_steps
        if n_points % 2 == 0:
            n_points += 1
        t_vals = torch.linspace(self.eps_eval, 1 - self.eps_eval, n_points, device=self.device)

        # Compute time scores at each t for all samples using vmap
        # Process in chunks to avoid OOM with large n_points * n_samples
        compute_vmapped = torch.vmap(
            self._compute_time_score_single,
            in_dims=(0, None),  # batch over t, broadcast samples
            out_dims=0
        )
        chunk_size = max(1, 100000 // n_samples)  # Aim for ~100k total evaluations per chunk
        time_score_chunks = []
        for i in range(0, n_points, chunk_size):
            t_chunk = t_vals[i:i + chunk_size]
            chunk_scores = compute_vmapped(t_chunk, samples).detach()
            time_score_chunks.append(chunk_scores)
        time_scores = torch.cat(time_score_chunks, dim=0)  # [n_points, n_samples]

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
            return torch.from_numpy(integral)
        elif self.integration_type == '1':
            return time_scores.mean(dim=0).cpu()
        elif self.integration_type == '2':
            return torch.trapz(time_scores, t_vals, dim=0).cpu()

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

        # Train the model (uses samples_pstar, samples_p0, and samples_p1)
        self.fit(samples_pstar, samples_p0, samples_p1)

        # Compute time_score integral for each x in samples_pstar
        samples_pstar = samples_pstar.float().to(self.device)
        integrals = self._integrate_time_score(samples_pstar)

        # ELDR = -E[integral of time_score]
        eldr = -integrals.mean().item()

        return eldr


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.distributions import MultivariateNormal, kl_divergence
    from experiments.utils.prescribed_kls import create_two_gaussians_kl

    DIM = 2
    NSAMPLES_TRAIN = 10000
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
    print("=" * 80)

    # === CONFIGS TO TRY ===
    # eps_train: training epsilon for t sampling and grad clipping (default 9e-4)
    # eps_eval: integration bounds (default 0.01)
    CONFIGS = [
        {'name': 'default', 'k': 24, 'eps_train': 9e-4, 'eps_eval': 0.01, 'antithetic': True, 'integration_type': '2', 'seed': 0},
        {'name': 'k=20', 'k': 20, 'eps_train': 9e-4, 'eps_eval': 0.01, 'antithetic': True, 'integration_type': '2', 'seed': 0},
        {'name': 'k=28', 'k': 28, 'eps_train': 9e-4, 'eps_eval': 0.01, 'antithetic': True, 'integration_type': '2', 'seed': 0},
        {'name': 'no_antithetic', 'k': 24, 'eps_train': 9e-4, 'eps_eval': 0.01, 'antithetic': False, 'integration_type': '2', 'seed': 0},
        {'name': 'simpson', 'k': 24, 'eps_train': 9e-4, 'eps_eval': 0.01, 'antithetic': True, 'integration_type': '3', 'seed': 0},
    ]

    all_results = {}

    for cfg in CONFIGS:
        print(f"Running config: {cfg['name']}")

        # Set seed for reproducibility
        if 'seed' in cfg:
            torch.manual_seed(cfg['seed'])
            np.random.seed(cfg['seed'])

        estimator = DirectELDREstimator4(
            input_dim=DIM,
            hidden_dim=256,
            n_epochs=600,
            k=cfg['k'],
            eps_train=cfg['eps_train'],
            eps_eval=cfg['eps_eval'],
            antithetic=cfg['antithetic'],
            integration_type=cfg['integration_type'],
            integration_steps=3000,
            verbose=False,
            device=DEVICE,
        )

        eldr_estimate = estimator.estimate_eldr(samples_base, samples_p0, samples_p1)
        error = abs(eldr_estimate - true_eldr)
        rel_error = error / true_eldr * 100

        all_results[cfg['name']] = {
            'estimate': eldr_estimate,
            'error': error,
            'rel_error': rel_error,
            'config': cfg,
        }

        print(f"  Estimate: {eldr_estimate:.4f}, Error: {error:.4f} ({rel_error:.2f}%)")

    # === PLOT RESULTS ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    names = list(all_results.keys())
    estimates = [all_results[n]['estimate'] for n in names]
    errors = [all_results[n]['error'] for n in names]
    rel_errors = [all_results[n]['rel_error'] for n in names]

    # Plot 1: ELDR Estimates
    ax = axes[0, 0]
    bars = ax.bar(names, estimates, color='steelblue', alpha=0.7)
    ax.axhline(y=true_eldr, color='red', linestyle='--', linewidth=2, label=f'True ELDR = {true_eldr:.2f}')
    ax.set_ylabel('ELDR Estimate')
    ax.set_title('ELDR Estimates by Config')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

    # Plot 2: Absolute Errors
    ax = axes[0, 1]
    bars = ax.bar(names, errors, color='coral', alpha=0.7)
    ax.set_ylabel('Absolute Error')
    ax.set_title('Absolute Errors by Config')
    ax.tick_params(axis='x', rotation=45)

    # Plot 3: Relative Errors
    ax = axes[1, 0]
    bars = ax.bar(names, rel_errors, color='seagreen', alpha=0.7)
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Relative Errors by Config')
    ax.tick_params(axis='x', rotation=45)

    # Plot 4: Gamma function for reference
    ax = axes[1, 1]
    t_vals = np.linspace(0, 1, 200)
    for cfg in CONFIGS[:3]:  # Plot gamma for different k values
        k = cfg['k']
        gamma_vals = (1 - np.exp(-k * t_vals)) * (1 - np.exp(-k * (1 - t_vals)))
        ax.plot(t_vals, gamma_vals, label=f"k={k}")
    ax.set_xlabel('t')
    ax.set_ylabel('gamma(t)')
    ax.set_title('Gamma Function for Different k')
    ax.legend()

    plt.tight_layout()
    plt.savefig('direct4_hpo.png', dpi=150)
    plt.show()

    # === PRINT SUMMARY ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<20} {'Estimate':>10} {'Error':>10} {'Rel Err %':>10}")
    print("-" * 50)
    for name in names:
        r = all_results[name]
        print(f"{name:<20} {r['estimate']:>10.4f} {r['error']:>10.4f} {r['rel_error']:>10.2f}")
    print("-" * 50)
    print(f"True ELDR: {true_eldr:.4f}")
    print(f"Mean estimate: {np.mean(estimates):.4f} (std={np.std(estimates):.4f})")
    print(f"Mean error: {np.mean(errors):.4f} (std={np.std(errors):.4f})")
    print(f"Best error: {min(errors):.4f}, Worst error: {max(errors):.4f}")
    print("Saved plot to direct4_hpo.png")
