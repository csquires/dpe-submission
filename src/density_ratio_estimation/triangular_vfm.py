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
from typing import Optional, Literal, Tuple, Callable
import warnings

import numpy as np
import torch
import torch.optim as optim

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._ema import EMA, maybe_clip_grad
from src.density_ratio_estimation._trainer import train_two_phase
from src.density_ratio_estimation._losses import velo_matching_loss, denoiser_loss
from src.models.common.mlp import MLP
from src.models.flow.div_estimators import exact_div, hutch_div
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
        div_method: Literal['hutchinson', 'exact'] = 'hutchinson',
        div_noise: Literal['rademacher', 'gaussian'] = 'rademacher',
        n_hutch_samples: int = 1,
        verbose: bool = False,
        log_every: int = 100,
        ema_decay: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        activation: str = "gelu",
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        cosine_min_factor: float = 1.0,
        layernorm: str = "off",
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
            activation: MLP activation function {"elu", "gelu", "silu"};
            default "gelu" preserves byte-identical behavior.
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
        self.adam_betas = tuple(adam_betas)
        self.weight_decay = float(weight_decay)
        if not (0.0 <= cosine_min_factor <= 1.0):
            raise ValueError(f"cosine_min_factor must be in [0, 1], got {cosine_min_factor}")
        self.cosine_min_factor = float(cosine_min_factor)
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off', 'pre', 'post'}}; got {layernorm!r}")
        self.layernorm = layernorm

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
        self.ema_b: Optional[EMA] = None
        self.ema_eta: Optional[EMA] = None

    def init_model(self) -> None:
        """
        Instantiate and move net_b and net_eta to device.

        Creates MLPs with forward signature forward(t: [B, 1], x: [B, D]) -> [B, D].
        """
        self.net_b = MLP(self.input_dim, self.hidden_dim, output_dim=self.input_dim, n_hidden_layers=self.n_hidden_layers, activation=self.activation, layernorm=self.layernorm).to(self.device)
        self.net_eta = MLP(self.input_dim, self.hidden_dim, output_dim=self.input_dim, n_hidden_layers=self.n_hidden_layers, activation=self.activation, layernorm=self.layernorm).to(self.device)
        self.ema_b = EMA(self.net_b, self.ema_decay) if self.ema_decay is not None else None
        self.ema_eta = EMA(self.net_eta, self.ema_decay) if self.ema_decay is not None else None

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
            1. Validate samples and move to device.
            2. Initialize net_b and net_eta.
            3. Set up optimizers, schedulers, and EMA helpers.
            4. Call train_two_phase with velo_matching_loss (b-phase) and denoiser_loss (eta-phase).
            5. Set both networks to eval mode.
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
        samples_p0 = samples_p0.float().to(self.device)  # [n0, D]
        samples_p1 = samples_p1.float().to(self.device)  # [n1, D]
        samples_pstar = samples_pstar.float().to(self.device)  # [n_star, D]

        # initialize model
        self.init_model()

        # set up optimizers
        optim_b = optim.Adam(
            self.net_b.parameters(),
            lr=self.lr,
            betas=self.adam_betas,
            eps=1e-8,
            weight_decay=self.weight_decay,
        )

        scheduler_b = (
            None
            if self.cosine_min_factor == 1.0
            else optim.lr_scheduler.CosineAnnealingLR(
                optim_b, T_max=self.n_epochs, eta_min=self.lr * self.cosine_min_factor
            )
        )

        optim_eta = optim.Adam(
            self.net_eta.parameters(),
            lr=self.lr,
            betas=self.adam_betas,
            eps=1e-8,
            weight_decay=self.weight_decay,
        )

        scheduler_eta = (
            None
            if self.cosine_min_factor == 1.0
            else optim.lr_scheduler.CosineAnnealingLR(
                optim_eta, T_max=self.n_epochs, eta_min=self.lr * self.cosine_min_factor
            )
        )

        # set up EMA helpers
        ema_b = EMA(self.net_b, self.ema_decay) if self.ema_decay is not None else None
        ema_eta = EMA(self.net_eta, self.ema_decay) if self.ema_decay is not None else None

        # create time sampler
        def time_sampler(batch_size: int, eps: float, device) -> tuple[torch.Tensor, torch.Tensor]:
            """sample tau ~ U([eps, 1-eps]) with importance weight 1."""
            sampler = getattr(self.path, "sample_tau", None)
            if callable(sampler):
                tau = sampler(batch_size, eps, device)  # [B, 1]
            else:
                tau = torch.rand(batch_size, 1, device=device) * (1 - 2*eps) + eps  # [B, 1]
            iw = torch.ones(batch_size, 1, device=device)  # [B, 1]
            return tau, iw

        # logging
        if self.verbose:
            print("[TriangularVFM] Starting Sequential Training (3 distributions)")

        # call train_two_phase
        train_two_phase(
            model_b=self.net_b,
            model_eta=self.net_eta,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_b=velo_matching_loss,
            loss_eta=denoiser_loss,
            optim_b=optim_b,
            optim_eta=optim_eta,
            n_steps_b=self.n_epochs,
            n_steps_eta=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler_b=scheduler_b,
            scheduler_eta=scheduler_eta,
            ema_b=ema_b,
            ema_eta=ema_eta,
            grad_clip_norm_b=self.grad_clip_norm,
            grad_clip_norm_eta=self.grad_clip_norm,
            eps=self.eps,
            loss_kwargs_b={"path": self.path, "antithetic": self.antithetic},
            loss_kwargs_eta={"path": self.path},
        )

        # final state (train_two_phase already sets both to eval, but explicit for clarity)
        self.net_b.eval()
        self.net_eta.eval()



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

        # if EMA is active, swap in shadow weights
        if self.ema_b is not None:
            self.ema_b.apply_to(self.net_b)
        if self.ema_eta is not None:
            self.ema_eta.apply_to(self.net_eta)

        try:
            # EXPLICIT CHUNKING (mirrors spatial_velo_denoiser2.py lines 348–352)
            chunk_size = max(1, 100000 // n_samples)
            compute_vmapped = torch.vmap(
                self._compute_time_score_single,
                in_dims=(0, None),
                out_dims=0,  # explicit (matches spatial_velo_denoiser2.py:346)
                randomness='different',  # required for Hutchinson noise inside
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
        finally:
            if self.ema_b is not None:
                self.ema_b.restore(self.net_b)
            if self.ema_eta is not None:
                self.ema_eta.restore(self.net_eta)

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

        # divergence computation
        def b_single(x_single):
            """Evaluate b at a single point with current tau."""
            return self.net_b(t_scalar.view(1, 1), x_single.unsqueeze(0)).squeeze(0)

        if self.div_method == 'exact':
            div_b = exact_div(b_single, x)  # [n_samples]
        else:
            div_b = hutch_div(b_single, x, noise=self.div_noise)
            for _ in range(self.n_hutch_samples - 1):
                div_b = div_b + hutch_div(b_single, x, noise=self.div_noise)
            div_b = div_b / self.n_hutch_samples

        # compute b·η dot product
        b_dot_eta = (b_pred * eta_pred).sum(dim=-1)  # [n_samples]

        # time-score formula
        time_score = -div_b + b_dot_eta / gamma_t  # [n_samples]
        return time_score
