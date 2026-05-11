"""TriangularVFM2D: V3-VFM 2D-time stacked-interpolant density ratio estimator.

Migrated per B13 spec. Trains two velocity heads (b_1, b_2) and one denoiser (eta)
sequentially on a 2D-time stacked interpolant path via inline losses. Inference
integrates the time-score along a Curve2D from tau=eps to 1-eps.

Note: Inline losses used (not delegated to train_two_phase or train_score_flow).
This is pragmatic (transparent, no closure overhead) until Scope A losses are
extended with model_call kwarg.
"""
from typing import Optional, Literal, Callable
import warnings
import itertools

import torch
import torch.optim as optim

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._ema import EMA, maybe_clip_grad
from src.models.flow.div_estimators import exact_div, hutch_div
from src.waypoints.path_2d import VfmPath2D
from src.waypoints.triangular_continuous_2d import Stacked2DVfm
from src.waypoints.curve_2d import Curve2D
from src.models.time_score_matching.velocity_network_2d import MLP2D


class TriangularVFM2D(DensityRatioEstimator):
    """V3-VFM 2D-time triangular VFM density ratio estimator.

    Trains two velocity heads (b_1, b_2) and one denoiser (eta) sequentially
    on three distributions p_0, p_1, p_*. Inference integrates the time-score
    along self.curve from tau=eps to 1-eps.

    Contract: fit(samples_p0, samples_p1, samples_pstar) with three [N, D]
    tensors; predict_ldr(xs) returns log(p_0/p_1) as [n_samples] CPU tensor.
    """

    def __init__(
        self,
        input_dim: int,
        path: Optional[VfmPath2D] = None,
        curve: Optional[Curve2D] = None,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1.3e-3,
        eps: float = 1e-3,
        device: Optional[str] = None,
        integration_steps: int = 200,
        antithetic: bool = True,
        div_method: Literal['hutchinson', 'exact'] = 'hutchinson',
        div_noise: Literal['rademacher', 'gaussian'] = 'rademacher',
        n_hutch_samples: int = 1,
        verbose: bool = False,
        log_every: int = 100,
        ema_decay: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        activation: str = "gelu",
    ) -> None:
        """Initialize TriangularVFM2D.

        Args:
            input_dim: Spatial dimension D.
            path: VfmPath2D instance. If None, default Stacked2DVfm(eps=eps).
            curve: Curve2D instance. If None, default Curve2D(path_height=1.0).
            hidden_dim: MLP2D hidden width.
            n_hidden_layers: Number of hidden layers for MLP2D networks.
            n_epochs: Training epochs per phase.
            batch_size: Minibatch size.
            lr: Adam learning rate.
            eps: Boundary margin for tau / t_1 sampling. Must be >= 1e-3.
            device: Device string. Auto-resolves if None.
            integration_steps: Number of tau quadrature points (uniform grid).
            antithetic: Toggle antithetic variance reduction in b-phase.
            verbose: Toggle epoch-level logging.
            log_every: Epochs between log prints.
        """
        # blocking validation: must be FIRST
        if eps < 1e-3:
            raise ValueError(
                f"eps must be >= 1e-3 for boundary regularity of b*eta/gamma; "
                f"got eps={eps}"
            )

        super().__init__(input_dim)

        # store hyperparameters
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.integration_steps = integration_steps
        self.antithetic = antithetic
        if div_method not in ('hutchinson', 'exact'):
            raise ValueError(f"div_method must be 'hutchinson' or 'exact'; got {div_method!r}")
        if div_noise not in ('rademacher', 'gaussian'):
            raise ValueError(f"div_noise must be 'rademacher' or 'gaussian'; got {div_noise!r}")
        if n_hutch_samples < 1:
            raise ValueError(f"n_hutch_samples must be >= 1; got {n_hutch_samples}")
        self.div_method = div_method
        self.div_noise = div_noise
        self.n_hutch_samples = n_hutch_samples
        self.verbose = verbose
        self.log_every = log_every
        self.ema_decay = ema_decay
        self.grad_clip_norm = grad_clip_norm
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )
        self.activation = activation

        # resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # resolve path / curve defaults
        self.path = path if path is not None else Stacked2DVfm(
            k=20.0, gamma_schedule="linear-stiff", t2_max=0.3, eps=eps
        )
        self.curve = curve if curve is not None else Curve2D(path_height=1.0)

        # coverage assertion: curve t_2 range must lie within trained t_2 range
        peak = float(self.curve.peak_t2())
        t2_max = float(self.path.t2_max)
        assert peak <= t2_max + 1e-9, (
            f"curve peak_t2 {peak} exceeds path.t2_max {t2_max}"
        )

        # network placeholders
        self.net_b1 = None
        self.net_b2 = None
        self.net_eta = None
        self.ema_b1: Optional[EMA] = None
        self.ema_b2: Optional[EMA] = None
        self.ema_eta: Optional[EMA] = None

    def init_model(self) -> None:
        """Instantiate three independent MLP2D networks on self.device."""
        self.net_b1 = MLP2D(self.input_dim, self.hidden_dim, output_dim=self.input_dim, n_hidden_layers=self.n_hidden_layers, activation=self.activation).to(self.device)
        self.net_b2 = MLP2D(self.input_dim, self.hidden_dim, output_dim=self.input_dim, n_hidden_layers=self.n_hidden_layers, activation=self.activation).to(self.device)
        self.net_eta = MLP2D(self.input_dim, self.hidden_dim, output_dim=self.input_dim, n_hidden_layers=self.n_hidden_layers, activation=self.activation).to(self.device)
        self.ema_b1 = EMA(self.net_b1, self.ema_decay) if self.ema_decay is not None else None
        self.ema_b2 = EMA(self.net_b2, self.ema_decay) if self.ema_decay is not None else None
        self.ema_eta = EMA(self.net_eta, self.ema_decay) if self.ema_decay is not None else None

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """Train b_1, b_2, eta networks sequentially on three distributions.

        Args:
            samples_p0: [N0, D] samples from p_0.
            samples_p1: [N1, D] samples from p_1.
            samples_pstar: [Nstar, D] samples from p_*.

        Procedure:
            phase 1: joint b_1, b_2 optimizer (eta frozen) — velocity matching loss.
            phase 2: eta optimizer (b_1, b_2 frozen) — denoising loss.
        """
        n_star = samples_pstar.shape[0]

        if n_star < 1:
            raise ValueError(f"samples_pstar must have at least 1 row; got n_star={n_star}")

        if n_star < self.batch_size // 4:
            warnings.warn(
                f"n_star={n_star} is small relative to batch_size={self.batch_size}; "
                f"pstar bootstrap will sample with high replication"
            )

        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)
        samples_pstar = samples_pstar.float().to(self.device)

        self.init_model()

        if self.verbose:
            print("[TriangularVFM2D] Starting Sequential Training (3 distributions)")

        self._train_b_phase(samples_p0, samples_p1, samples_pstar)
        self._train_eta_phase(samples_p0, samples_p1, samples_pstar)

        # post-training cleanup
        self.net_b1.eval()
        self.net_b2.eval()
        self.net_eta.eval()

    def _train_b_phase(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """Train b_1 and b_2 jointly with eta frozen.

        Velocity matching via inline losses (half-norm minus dot). Per-direction
        losses computed separately; sum is back-propagated. Antithetic variance
        reduction applied if self.antithetic=True.
        """
        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        self.net_b1.train()
        self.net_b2.train()
        self.net_eta.eval()
        optimizer_b = optim.Adam(
            itertools.chain(self.net_b1.parameters(), self.net_b2.parameters()),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        t2_max = float(self.path.t2_max)

        for epoch in range(self.n_epochs):
            # bootstrap minibatches [B, D]
            idx0 = torch.randint(0, n0, (self.batch_size,), device=self.device)
            idx1 = torch.randint(0, n1, (self.batch_size,), device=self.device)
            idx_star = torch.randint(0, n_star, (self.batch_size,), device=self.device)
            x0 = samples_p0[idx0]  # [B, D]
            x1 = samples_p1[idx1]  # [B, D]
            xstar = samples_pstar[idx_star]  # [B, D]

            # sample time on 2D domain
            t1 = torch.rand(self.batch_size, 1, device=self.device) * (1 - 2*self.eps) + self.eps  # [B, 1]
            t2 = torch.rand(self.batch_size, 1, device=self.device) * (t2_max - self.eps) + self.eps  # [B, 1]

            # sample noise
            z = torch.randn_like(x0)  # [B, D]

            # compute path quantities (detached — no gradients through path)
            mu = self.path.mu(x0, x1, xstar, t1, t2).detach()  # [B, D]
            dmu_dt1 = self.path.dmu_dt1(x0, x1, xstar, t1, t2).detach()  # [B, D]
            dmu_dt2 = self.path.dmu_dt2(x0, x1, xstar, t1, t2).detach()  # [B, D]
            gamma_t = self.path.gamma(t1, t2).detach()  # [B, 1]
            dgamma_dt1 = self.path.dgamma_dt1(t1, t2).detach()  # [B, 1]
            dgamma_dt2 = self.path.dgamma_dt2(t1, t2).detach()  # [B, 1]

            if self.antithetic:
                # antithetic variance reduction: evaluate at ±z
                x_t_plus = mu + gamma_t * z  # [B, D]
                x_t_minus = mu - gamma_t * z  # [B, D]

                b1_plus = self.net_b1(t1, t2, x_t_plus)  # [B, D]
                b2_plus = self.net_b2(t1, t2, x_t_plus)  # [B, D]
                b1_minus = self.net_b1(t1, t2, x_t_minus)  # [B, D]
                b2_minus = self.net_b2(t1, t2, x_t_minus)  # [B, D]

                target_1_plus = dmu_dt1 + dgamma_dt1 * z  # [B, D]
                target_2_plus = dmu_dt2 + dgamma_dt2 * z  # [B, D]
                target_1_minus = dmu_dt1 - dgamma_dt1 * z  # [B, D]
                target_2_minus = dmu_dt2 - dgamma_dt2 * z  # [B, D]

                # half-norm-minus-dot per direction, averaged over ±z pair
                loss_b1 = (
                    0.25 * (b1_plus ** 2).sum(dim=-1)
                    - 0.5 * (target_1_plus * b1_plus).sum(dim=-1)
                    + 0.25 * (b1_minus ** 2).sum(dim=-1)
                    - 0.5 * (target_1_minus * b1_minus).sum(dim=-1)
                ).mean()
                loss_b2 = (
                    0.25 * (b2_plus ** 2).sum(dim=-1)
                    - 0.5 * (target_2_plus * b2_plus).sum(dim=-1)
                    + 0.25 * (b2_minus ** 2).sum(dim=-1)
                    - 0.5 * (target_2_minus * b2_minus).sum(dim=-1)
                ).mean()
            else:
                # single forward pass
                x_t = mu + gamma_t * z  # [B, D]
                b1_pred = self.net_b1(t1, t2, x_t)  # [B, D]
                b2_pred = self.net_b2(t1, t2, x_t)  # [B, D]

                target_1 = dmu_dt1 + dgamma_dt1 * z  # [B, D]
                target_2 = dmu_dt2 + dgamma_dt2 * z  # [B, D]

                loss_b1 = (
                    0.5 * (b1_pred ** 2).sum(dim=-1)
                    - (target_1 * b1_pred).sum(dim=-1)
                ).mean()
                loss_b2 = (
                    0.5 * (b2_pred ** 2).sum(dim=-1)
                    - (target_2 * b2_pred).sum(dim=-1)
                ).mean()

            # backward + step
            loss = loss_b1 + loss_b2
            optimizer_b.zero_grad()
            loss.backward()
            maybe_clip_grad(
                list(self.net_b1.parameters()) + list(self.net_b2.parameters()),
                self.grad_clip_norm,
            )
            optimizer_b.step()
            if self.ema_b1 is not None:
                self.ema_b1.update(self.net_b1)
            if self.ema_b2 is not None:
                self.ema_b2.update(self.net_b2)

            # logging
            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"  [Epoch {epoch+1}] loss_b1={loss_b1.item():.4f} loss_b2={loss_b2.item():.4f}")

    def _train_eta_phase(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """Train eta with b_1 and b_2 frozen.

        Denoising loss (half-norm minus dot with noise).
        """
        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        self.net_b1.eval()
        self.net_b2.eval()
        self.net_eta.train()
        optimizer_eta = optim.Adam(
            self.net_eta.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )

        t2_max = float(self.path.t2_max)

        for epoch in range(self.n_epochs):
            # bootstrap minibatches [B, D]
            idx0 = torch.randint(0, n0, (self.batch_size,), device=self.device)
            idx1 = torch.randint(0, n1, (self.batch_size,), device=self.device)
            idx_star = torch.randint(0, n_star, (self.batch_size,), device=self.device)
            x0 = samples_p0[idx0]  # [B, D]
            x1 = samples_p1[idx1]  # [B, D]
            xstar = samples_pstar[idx_star]  # [B, D]

            # sample time on 2D domain
            t1 = torch.rand(self.batch_size, 1, device=self.device) * (1 - 2*self.eps) + self.eps  # [B, 1]
            t2 = torch.rand(self.batch_size, 1, device=self.device) * (t2_max - self.eps) + self.eps  # [B, 1]

            # sample noise and forward through path
            z = torch.randn_like(x0)  # [B, D]
            x_t = self.path.sample(x0, x1, xstar, t1, t2, z).detach()  # [B, D]

            # forward through eta
            eta_pred = self.net_eta(t1, t2, x_t)  # [B, D]

            # denoising loss: half-norm minus dot
            loss_eta = (
                0.5 * (eta_pred ** 2).sum(dim=-1) - (z * eta_pred).sum(dim=-1)
            ).mean()

            # backward + step
            optimizer_eta.zero_grad()
            loss_eta.backward()
            maybe_clip_grad(self.net_eta.parameters(), self.grad_clip_norm)
            optimizer_eta.step()
            if self.ema_eta is not None:
                self.ema_eta.update(self.net_eta)

            # logging
            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"  [Epoch {epoch+1}] loss_eta={loss_eta.item():.4f}")

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """Estimate log p_0(x) / p_1(x) via time-score line integral on self.curve.

        Computes TWO divergences (one per velocity head) per integration step.
        The 2x cost is intentional: 2D-time decomposition splits time-score into
        two directional components, each requiring its own div-of-velocity. Do
        NOT collapse the two jacrev calls — they have different signatures.

        Args:
            xs: [N, D] test points (CPU or device); moved to self.device.

        Returns:
            [N] log density ratios, CPU float32.

        Raises:
            RuntimeError: if any of self.net_b1, self.net_b2, self.net_eta is None.
        """
        if self.net_b1 is None or self.net_b2 is None or self.net_eta is None:
            raise RuntimeError(
                "TriangularVFM2D model is not trained. Call fit() before predict_ldr()."
            )

        self.net_b1.eval()
        self.net_b2.eval()
        self.net_eta.eval()
        samples = xs.float().to(self.device)  # [n_samples, D]
        n_samples = samples.shape[0]

        # create uniform tau grid and pack curve outputs
        n_points = self.integration_steps
        tau_vals = torch.linspace(
            self.eps, 1.0 - self.eps, steps=n_points, device=self.device
        )  # [n_points]

        curve = self.curve
        tau_list = tau_vals.tolist()  # n_points python floats
        t_data = torch.tensor(
            [
                [curve.t1(tau), curve.t2(tau), curve.dt1(tau), curve.dt2(tau)]
                for tau in tau_list
            ],
            device=self.device,
            dtype=samples.dtype,
        )  # [n_points, 4]

        # apply EMA if active
        if self.ema_b1 is not None:
            self.ema_b1.apply_to(self.net_b1)
        if self.ema_b2 is not None:
            self.ema_b2.apply_to(self.net_b2)
        if self.ema_eta is not None:
            self.ema_eta.apply_to(self.net_eta)

        try:
            # chunked inference: vmap over leading dim of t_data to avoid OOM
            chunk_size = max(1, 100000 // n_samples)
            compute_vmapped = torch.vmap(
                self._compute_time_score_single,
                in_dims=(0, None),
                out_dims=0,
                randomness="different",  # required for Hutchinson noise inside
            )

            time_score_chunks = []
            for i in range(0, n_points, chunk_size):
                t_chunk = t_data[i : i + chunk_size]  # [chunk_len, 4]
                chunk_scores = compute_vmapped(t_chunk, samples).detach()  # [chunk_len, n_samples]
                time_score_chunks.append(chunk_scores)

            time_scores = torch.cat(time_score_chunks, dim=0)  # [n_points, n_samples]

            # integrate via trapezoidal rule
            return -torch.trapezoid(time_scores, tau_vals, dim=0).cpu()  # [n_samples]
        finally:
            # restore original weights if EMA was applied
            if self.ema_b1 is not None:
                self.ema_b1.restore(self.net_b1)
            if self.ema_b2 is not None:
                self.ema_b2.restore(self.net_b2)
            if self.ema_eta is not None:
                self.ema_eta.restore(self.net_eta)

    def _compute_time_score_single(
        self, t_tau: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute time-score d log rho / d tau at a single tau via 2D-time decomposition.

        Unpack curve outputs (t_1, t_2, dt_1/dtau, dt_2/dtau), evaluate b_1, b_2, eta,
        compute divergences separately, and combine via chain rule.

        Args:
            t_tau: [4] packed (t_1, t_2, dt_1/dtau, dt_2/dtau) as 0-d slices
                   produced by outer vmap over [n_points, 4].
            x: [n_samples, D] test points (broadcast — not vmapped).

        Returns:
            [n_samples] time scores at the current tau.
        """
        # unpack 0-d tensors from curve
        t1_s = t_tau[0]  # 0-d
        t2_s = t_tau[1]  # 0-d
        dt1_s = t_tau[2]  # 0-d
        dt2_s = t_tau[3]  # 0-d

        n_samples = x.shape[0]

        # expand 0-d tensors to [n_samples, 1] for network interface
        t1_batch = t1_s.view(1, 1).expand(n_samples, 1)  # [n_samples, 1]
        t2_batch = t2_s.view(1, 1).expand(n_samples, 1)  # [n_samples, 1]

        # gamma at scalar (t_1, t_2)
        gamma_t = self.path.gamma(t1_s.view(1, 1), t2_s.view(1, 1)).squeeze()  # 0-d

        # network forwards (full batch)
        b1_pred = self.net_b1(t1_batch, t2_batch, x)  # [n_samples, D]
        b2_pred = self.net_b2(t1_batch, t2_batch, x)  # [n_samples, D]
        eta_pred = self.net_eta(t1_batch, t2_batch, x)  # [n_samples, D]

        # divergence via vmap(jacrev) per head — TWO separate calls, do NOT collapse
        t1_one = t1_s.view(1, 1)
        t2_one = t2_s.view(1, 1)

        def b1_single(x_single):
            return self.net_b1(t1_one, t2_one, x_single.unsqueeze(0)).squeeze(0)

        def b2_single(x_single):
            return self.net_b2(t1_one, t2_one, x_single.unsqueeze(0)).squeeze(0)

        # compute divergences
        if self.div_method == "exact":
            div_b1 = exact_div(b1_single, x)  # [n_samples]
            div_b2 = exact_div(b2_single, x)  # [n_samples]
        else:
            # hutchinson with optional multiple samples
            div_b1 = hutch_div(b1_single, x, noise=self.div_noise)  # [n_samples]
            div_b2 = hutch_div(b2_single, x, noise=self.div_noise)  # [n_samples]
            for _ in range(self.n_hutch_samples - 1):
                div_b1 = div_b1 + hutch_div(b1_single, x, noise=self.div_noise)
                div_b2 = div_b2 + hutch_div(b2_single, x, noise=self.div_noise)
            div_b1 = div_b1 / self.n_hutch_samples
            div_b2 = div_b2 / self.n_hutch_samples

        # dot products
        b1_dot_eta = (b1_pred * eta_pred).sum(dim=-1)  # [n_samples]
        b2_dot_eta = (b2_pred * eta_pred).sum(dim=-1)  # [n_samples]

        # directional time-score components
        s_1 = -div_b1 + b1_dot_eta / gamma_t  # [n_samples]
        s_2 = -div_b2 + b2_dot_eta / gamma_t  # [n_samples]

        # combine via curve derivatives (chain rule)
        time_score = s_1 * dt1_s + s_2 * dt2_s  # [n_samples]
        return time_score
