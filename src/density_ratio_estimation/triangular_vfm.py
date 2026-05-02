"""TriangularVFM: continuous-time barycentric VFM density ratio estimator.

Mirrors SpatialVeloDenoiser2 (sequential-only b/eta training, vmap-jacrev divergence
at inference, create_graph=False). The interpolant uses a C^1 triangular barycentric
path via three anchor distributions p_0, p_1, p_*.

Note on the path geometry: under the C^1 barycentric weights, all weight derivatives
vanish at tau=0.5, so Delta_{0.5} = 0. This is benign — measure-zero under
tau ~ Uniform([eps, 1-eps]), geometrically expected (mu_{0.5} = x_*, locally
stationary), and empirically navigated by the sibling TriangularCTSM V2 which uses
the identical BarycentricCtsm1D path. 
"""
from typing import Optional, Literal
import warnings

import numpy as np
import torch
import torch.optim as optim

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation.spatial_velo_denoiser2 import MLP, compute_divergence
from src.waypoints.path_1d import VfmPath1D
from src.waypoints.triangular_continuous import BarycentricVfm1D


class TriangularVFM(DensityRatioEstimator):
    """
    Triangular continuous-time velocity field matching for density ratio estimation.

    Uses a continuous barycentric path through p0 -> p* -> p1.
    Trains velocity field (b) and denoiser (eta) networks sequentially.
    Inference: time-score integration from tau=eps to 1-eps yields log(p0/p1).

    Contract: fit(samples_p0, samples_p1, samples_pstar) with three tensors [N, D].
    predict_ldr(xs) returns log density ratios as 1D CPU tensor [N].
    """

    def __init__(
        self,
        input_dim: int,
        path: Optional[VfmPath1D] = None,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1.3e-3,
        eps: float = 1e-3,
        device: Optional[str] = None,
        integration_steps: int = 3000,
        integration_type: Literal['1', '2', '3'] = '2',
        antithetic: bool = True,
        verbose: bool = False,
        log_every: int = 100,
    ) -> None:
        """
        Initialize TriangularVFM.

        Args:
            input_dim: Input dimension D.
            path: VfmPath1D instance. If None, instantiate BarycentricVfm1D(k=20.0, vertex=0.5, eps=eps).
            hidden_dim: Hidden layer width for MLP networks.
            n_hidden_layers: Number of hidden layers for MLP networks.
            n_epochs: Number of training epochs.
            batch_size: Batch size for stochastic gradient descent.
            lr: Adam learning rate.
            eps: Margin for tau sampling and inference bounds. tau in [eps, 1-eps].
                 Must be >= 1e-3 for boundary regularity (raises ValueError if not).
            device: Device string ("cuda", "cpu", etc.). If None, auto-detect.
            integration_steps: Number of time steps for integration.
            integration_type: Integration method: '1' (mean), '2' (trapz), '3' (Simpson).
            antithetic: If True, use antithetic variance reduction in b-phase training.
            verbose: If True, print loss per epoch.
            log_every: Log frequency (epochs between prints).
        """
        # BLOCKING: boundary-regularity validation (must be FIRST)
        if eps < 1e-3:
            raise ValueError(
                f"eps must be >= 1e-3 for boundary regularity of b*eta/gamma "
                f"; got eps={eps}"
            )

        super().__init__(input_dim)

        # store all constructor args
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.integration_steps = integration_steps
        self.integration_type = integration_type
        self.antithetic = antithetic
        self.verbose = verbose
        self.log_every = log_every

        # resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # resolve path (default to BarycentricVfm1D if not provided)
        if path is None:
            self.path = BarycentricVfm1D(k=20.0, vertex=0.5, eps=eps)
        else:
            self.path = path

        # initialize network placeholders
        self.net_b = None
        self.net_eta = None

    def init_model(self) -> None:
        """
        Instantiate and move net_b and net_eta to device.

        Creates MLPs with forward signature forward(t: [B, 1], x: [B, D]) -> [B, D].
        """
        self.net_b = MLP(self.input_dim, self.hidden_dim, output_dim=self.input_dim, n_hidden_layers=self.n_hidden_layers).to(self.device)
        self.net_eta = MLP(self.input_dim, self.hidden_dim, output_dim=self.input_dim, n_hidden_layers=self.n_hidden_layers).to(self.device)

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """
        Train b and eta networks sequentially on three distributions.

        Args:
            samples_p0: Samples from p0, shape [N0, D].
            samples_p1: Samples from p1, shape [N1, D].
            samples_pstar: Samples from p* (anchor distribution), shape [Nstar, D].

        Procedure:
            phase 1: train net_b (net_eta frozen) with velocity matching loss
            phase 2: train net_eta (net_b frozen) with denoising loss
        """
        # extract sample counts and validate n_star
        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        if n_star < 1:
            raise ValueError(f"samples_pstar must have at least 1 row; got n_star={n_star}")

        if n_star < self.batch_size // 4:
            warnings.warn(
                f"n_star={n_star} is small relative to batch_size={self.batch_size}; "
                f"pstar bootstrap will sample with high replication"
            )

        # move samples to device and cast to float
        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)
        samples_pstar = samples_pstar.float().to(self.device)

        # initialize model
        self.init_model()

        # logging
        if self.verbose:
            print("[TriangularVFM] Starting Sequential Training (3 distributions)")

        # train phases
        self._train_b_phase(samples_p0, samples_p1, samples_pstar)
        self._train_eta_phase(samples_p0, samples_p1, samples_pstar)

        # post-training cleanup
        self.net_b.eval()
        self.net_eta.eval()

    def _train_b_phase(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """
        Train velocity field network net_b with net_eta frozen.

        Updates net_b parameters via Adam to minimize velocity matching loss.
        Supports antithetic variance reduction if self.antithetic=True.

        Args:
            samples_p0: [N0, D] samples from p0, on device.
            samples_p1: [N1, D] samples from p1, on device.
            samples_pstar: [Nstar, D] samples from p*, on device.
        """
        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        self.net_b.train()
        self.net_eta.eval()
        optimizer_b = optim.Adam(self.net_b.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        for epoch in range(self.n_epochs):
            # bootstrap sampling (with replacement)
            idx0 = torch.randint(0, n0, (self.batch_size,), device=self.device)
            idx1 = torch.randint(0, n1, (self.batch_size,), device=self.device)
            idx_star = torch.randint(0, n_star, (self.batch_size,), device=self.device)
            x0 = samples_p0[idx0]  # [B, D]
            x1 = samples_p1[idx1]  # [B, D]
            xstar = samples_pstar[idx_star]  # [B, D]

            # time sampling (clamped to [eps, 1-eps])
            tau = torch.rand(self.batch_size, 1, device=self.device) * (1 - 2*self.eps) + self.eps  # [B, 1]

            # noise and path derivatives
            z = torch.randn_like(x0)  # [B, D]
            Delta = self.path.drift_deriv(x0, x1, xstar, tau).detach()  # [B, D]
            gamma_t = self.path.gamma(tau).detach()  # [B, 1] or scalar broadcast
            gamma_prime = self.path.dgamma_dtau(tau).detach()  # [B, 1]
            mu = self.path.drift(x0, x1, xstar, tau).detach()  # [B, D]

            # branch on antithetic flag
            if self.antithetic:
                # antithetic variance reduction
                x_t_plus = mu + gamma_t * z  # [B, D]
                x_t_minus = mu - gamma_t * z  # [B, D]

                b_plus = self.net_b(tau, x_t_plus)  # [B, D]
                b_minus = self.net_b(tau, x_t_minus)  # [B, D]

                b_norm_sq_plus = (b_plus ** 2).sum(dim=-1)  # [B]
                b_norm_sq_minus = (b_minus ** 2).sum(dim=-1)  # [B]

                target_plus = Delta + gamma_prime * z  # [B, D]
                target_minus = Delta - gamma_prime * z  # [B, D]

                t_dot_b_plus = (target_plus * b_plus).sum(dim=-1)  # [B]
                t_dot_b_minus = (target_minus * b_minus).sum(dim=-1)  # [B]

                loss_b = (0.25 * b_norm_sq_plus - 0.5 * t_dot_b_plus
                        + 0.25 * b_norm_sq_minus - 0.5 * t_dot_b_minus).mean()
            else:
                # standard training (no antithetic)
                x_t = mu + gamma_t * z  # [B, D]
                b_pred = self.net_b(tau, x_t)  # [B, D]

                target = Delta + gamma_prime * z  # [B, D]
                b_norm_sq = (b_pred ** 2).sum(dim=-1)  # [B]
                t_dot_b = (target * b_pred).sum(dim=-1)  # [B]

                loss_b = (0.5 * b_norm_sq - t_dot_b).mean()

            # gradient step
            optimizer_b.zero_grad()
            loss_b.backward()
            optimizer_b.step()

            # logging
            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"  [Epoch {epoch+1}] loss_b={loss_b.item():.4f}")

    def _train_eta_phase(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """
        Train denoiser network net_eta with net_b frozen.

        Updates net_eta parameters via Adam to minimize denoising loss.
        KEY: tau is clamped to [eps, 1-eps] (deviation from stock VFM which uses [0,1]).

        Args:
            samples_p0: [N0, D] samples from p0, on device.
            samples_p1: [N1, D] samples from p1, on device.
            samples_pstar: [Nstar, D] samples from p*, on device.
        """
        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        self.net_b.eval()
        self.net_eta.train()
        optimizer_eta = optim.Adam(self.net_eta.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        for epoch in range(self.n_epochs):
            # bootstrap sampling (identical to b-phase)
            idx0 = torch.randint(0, n0, (self.batch_size,), device=self.device)
            idx1 = torch.randint(0, n1, (self.batch_size,), device=self.device)
            idx_star = torch.randint(0, n_star, (self.batch_size,), device=self.device)
            x0 = samples_p0[idx0]  # [B, D]
            x1 = samples_p1[idx1]  # [B, D]
            xstar = samples_pstar[idx_star]  # [B, D]

            # time sampling (CLAMPED to [eps, 1-eps] — KEY DEVIATION FROM STOCK VFM)
            tau = torch.rand(self.batch_size, 1, device=self.device) * (1 - 2*self.eps) + self.eps  # [B, 1]

            # noise and path sampling
            z = torch.randn_like(x0)  # [B, D]
            gamma_t = self.path.gamma(tau).detach()  # [B, 1]
            x_t = self.path.sample(x0, x1, xstar, tau, z).detach()  # [B, D]

            # forward pass and loss
            eta_pred = self.net_eta(tau, x_t)  # [B, D]

            eta_norm_sq = (eta_pred ** 2).sum(dim=-1)  # [B]
            z_dot_eta = (z * eta_pred).sum(dim=-1)  # [B]

            loss_eta = (0.5 * eta_norm_sq - z_dot_eta).mean()

            # gradient step
            optimizer_eta.zero_grad()
            loss_eta.backward()
            optimizer_eta.step()

            # logging
            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"  [Epoch {epoch+1}] loss_eta={loss_eta.item():.4f}")

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Estimate log p_0(x) / p_1(x) via time-score integration.

        Integrates time-score (derivative of log rho w.r.t. tau) over [eps, 1-eps]
        using chunked vmap for memory efficiency.

        Args:
            xs: Test samples, shape [N, D], on CPU or device (will be moved to self.device).

        Returns:
            Log density ratios, shape [N], on CPU, float32.

        Raises:
            RuntimeError: If model is not trained (net_b or net_eta is None).
        """
        if self.net_b is None or self.net_eta is None:
            raise RuntimeError("TriangularVFM model is not trained. Call fit() before predict_ldr().")

        self.net_b.eval()
        self.net_eta.eval()
        samples = xs.float().to(self.device)  # [n_samples, D]
        n_samples = samples.shape[0]

        # time grid (ensure odd for Simpson's rule)
        n_points = self.integration_steps
        if n_points % 2 == 0:
            n_points += 1
        t_vals = torch.linspace(self.eps, 1.0 - self.eps, steps=n_points, device=self.device)

        # EXPLICIT CHUNKING (mirrors spatial_velo_denoiser2.py lines 348–352)
        chunk_size = max(1, 100000 // n_samples)
        compute_vmapped = torch.vmap(
            self._compute_time_score_single,
            in_dims=(0, None),
            out_dims=0,  # explicit (matches spatial_velo_denoiser2.py:346)
        )

        time_score_chunks = []
        for i in range(0, n_points, chunk_size):
            t_chunk = t_vals[i:i + chunk_size]
            chunk_scores = compute_vmapped(t_chunk, samples).detach()  # [chunk_len, n_samples]
            time_score_chunks.append(chunk_scores)

        time_scores = torch.cat(time_score_chunks, dim=0)  # [n_points, n_samples]

        # integration
        if self.integration_type == '2':
            # trapezoidal rule (default)
            out = -torch.trapz(time_scores, t_vals, dim=0).cpu()
        elif self.integration_type == '3':
            # Simpson's rule (mirror spatial_velo_denoiser2.py:356–369)
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
            # mean (uniform quadrature)
            out = -time_scores.mean(dim=0).cpu()

        return out  # [n_samples], CPU, float32

    def _compute_time_score_single(self, t_scalar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute time-score (derivative of log ρ w.r.t. tau) for a single time point.

        Mirrors SpatialVeloDenoiser2._compute_time_score_single (lines 168–199).
        Uses vmap(jacrev) to compute divergence efficiently per-sample.

        Args:
            t_scalar: Single time value (scalar tensor or [1]).
            x: Sample points [n_samples, D].

        Returns:
            Time scores [n_samples].

        Procedure:
            1. expand t_scalar to [n_samples, 1] for batch forward pass.
            2. evaluate b and eta networks.
            3. use vmap(jac_trace) to compute divergence of b w.r.t. x.
            4. compute b·η dot product.
            5. return -div_b + b·η / gamma_t.
        """
        n_samples = x.shape[0]
        t_batch = t_scalar.expand(n_samples, 1)  # [n_samples, 1]

        # get path-dependent quantity
        gamma_t = self.path.gamma(t_scalar)  # scalar or [1]; ensure scalar for division
        if gamma_t.dim() > 0:
            gamma_t = gamma_t.squeeze()

        # network forward passes
        b_pred = self.net_b(t_batch, x)  # [n_samples, D]
        eta_pred = self.net_eta(t_batch, x)  # [n_samples, D]

        # divergence computation via vmap
        def b_single(x_single):
            """Evaluate b at a single point with current tau."""
            return self.net_b(t_scalar.view(1, 1), x_single.unsqueeze(0)).squeeze(0)

        def jac_trace(x_single):
            """Compute trace of Jacobian of b w.r.t. x_single."""
            jac = torch.func.jacrev(b_single)(x_single)  # [D, D]
            return torch.trace(jac)

        div_b = torch.vmap(jac_trace)(x)  # [n_samples]

        # compute b·η dot product
        b_dot_eta = (b_pred * eta_pred).sum(dim=-1)  # [n_samples]

        # time-score formula
        time_score = -div_b + b_dot_eta / gamma_t  # [n_samples]
        return time_score


if __name__ == '__main__':
    from torch.distributions import MultivariateNormal
    from experiments.utils.prescribed_kls import create_two_gaussians_kl
    from src.waypoints.piecewise_sb import PiecewiseSBVfm1D

    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 10
    KL_DIVERGENCE = 5.0

    # Create Gaussian pair
    gaussian_pair = create_two_gaussians_kl(dim=DIM, k=KL_DIVERGENCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair["mu0"], gaussian_pair["Sigma0"]
    mu1, Sigma1 = gaussian_pair["mu1"], gaussian_pair["Sigma1"]

    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

    # Anchor distribution p* = midpoint Gaussian
    mu_star = (mu0 + mu1) / 2.0
    Sigma_star = (Sigma0 + Sigma1) / 2.0
    pstar = MultivariateNormal(mu_star, covariance_matrix=Sigma_star)

    # Sample
    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))
    samples_pstar = pstar.sample((NSAMPLES_TRAIN,))
    samples_test = p0.sample((NSAMPLES_TEST,))

    # Instantiate and train
    estimator = TriangularVFM(input_dim=DIM, verbose=True)
    estimator.fit(samples_p0, samples_p1, samples_pstar)

    # Predict and evaluate
    est_ldrs = estimator.predict_ldr(samples_test)
    true_ldrs = p0.log_prob(samples_test) - p1.log_prob(samples_test)
    mae = torch.mean(torch.abs(est_ldrs - true_ldrs))

    print(f"MAE: {mae:.6f}")

    # V1 VFM toy block: vertex sweep + gamma_min sweep
    # Each training run takes ~30–90 seconds on CPU.
    # Full sweep (3 vertices + 3 gamma_min = 6 runs) may take 3–9 minutes.

    print("\n" + "="*60)
    print("V1 VFM Vertex Sweep (gamma_min=5e-2)")
    print("="*60)

    for vertex in [0.2, 0.5, 0.8]:
        path = PiecewiseSBVfm1D(sigma=1.0, vertex=vertex, gamma_min=5e-2, eps=1e-3)
        estimator = TriangularVFM(input_dim=DIM, path=path, verbose=False)
        estimator.fit(samples_p0, samples_p1, samples_pstar)
        est_ldrs = estimator.predict_ldr(samples_test)
        mae = torch.mean(torch.abs(est_ldrs - true_ldrs))
        print(f"[V1 VFM, vertex={vertex}, gamma_min=5e-2] MAE: {mae:.6f}")
        # smoke-test bound matches CTSM toy; tighten once the empirical
        # MAE distribution on a larger test set is known.
        assert torch.isfinite(mae) and mae < 10.0, (
            f"V1 VFM vertex={vertex} failed: mae={mae} >= 10.0"
        )

    print("\n" + "="*60)
    print("V1 VFM Gamma_min Sweep (vertex=0.5)")
    print("="*60)

    best_mae = float('inf')
    best_gamma_min = None

    for gamma_min in [1e-2, 5e-2, 1e-1]:
        path = PiecewiseSBVfm1D(sigma=1.0, vertex=0.5, gamma_min=gamma_min, eps=1e-3)
        estimator = TriangularVFM(input_dim=DIM, path=path, verbose=False)
        estimator.fit(samples_p0, samples_p1, samples_pstar)
        est_ldrs = estimator.predict_ldr(samples_test)
        mae = torch.mean(torch.abs(est_ldrs - true_ldrs))
        print(f"[V1 VFM, gamma_min={gamma_min}, vertex=0.5] MAE: {mae:.6f}")

        if mae < best_mae:
            best_mae = mae
            best_gamma_min = gamma_min

    print("\n" + "="*60)
    print("[V1 VFM gamma_min sweep summary]")
    print(f"  Best MAE: {best_mae:.6f} at gamma_min={best_gamma_min}")
    print(f"  Recommended gamma_min: {best_gamma_min}")
    print("="*60)
