"""TriangularVFM: VFM DRE on a barycentric path p0 -> p* -> p1."""
from typing import Optional, Literal, Tuple, Callable
import warnings

import numpy as np
import torch
import torch.optim as optim

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._ema import EMA, maybe_clip_grad
from src.density_ratio_estimation._trainer import train_two_phase
from src.density_ratio_estimation._losses import velo_loss, denoiser_loss
from src.models.common.mlp import MLP
from src.models.flow.div_estimators import exact_div, hutch_div
from src.waypoints.path_1d import VfmPath1D
from src.waypoints.triangular_continuous import BarycentricVfm1D


class TriangularVFM(DensityRatioEstimator):
    """VFM with barycentric path p0 -> p* -> p1.

    trains b and eta sequentially via `train_two_phase`; integrates the time-score
    over tau in [eps, 1-eps] at inference.
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
        """path defaults to BarycentricVfm1D(k=20.0, vertex=0.5, eps=eps);
        eps must be >= 1e-3 for boundary regularity of b eta / gamma."""
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
        """instantiate net_b, net_eta, and optional EMA wrappers."""
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
        """init b/eta nets and Adam+cosine optimizers, then delegate to `train_two_phase`
        with `velo_loss` (b-phase) and `denoiser_loss` (eta-phase)."""
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

        def time_sampler(B: int, eps: float, device) -> tuple[torch.Tensor, torch.Tensor]:
            sampler = getattr(self.path, "sample_tau", None)
            if callable(sampler):
                tau = sampler(B, eps, device)
            else:
                tau = torch.rand(B, 1, device=device) * (1 - 2 * eps) + eps
            return tau, torch.ones(B, 1, device=device)

        if self.verbose:
            print("[TriangularVFM] starting train_two_phase")

        train_two_phase(
            model_b=self.net_b,
            model_eta=self.net_eta,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_b=velo_loss,
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

        self.net_b.eval()
        self.net_eta.eval()



    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """integrate the time-score over tau in [eps, 1-eps] (chunked vmap); return -integral on CPU."""
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
            chunk_size = max(1, 100000 // n_samples)
            time_score_fn = torch.vmap(
                self._compute_time_score_single,
                in_dims=(0, None),
                out_dims=0,
                randomness="different",
            )
            chunks = []
            for i in range(0, n_points, chunk_size):
                chunks.append(time_score_fn(t_vals[i:i + chunk_size], samples).detach())
            time_scores = torch.cat(chunks, dim=0)

            if self.integration_type == "2":
                return -torch.trapz(time_scores, t_vals, dim=0).cpu()
            if self.integration_type == "3":
                t_np = t_vals.cpu().numpy()
                h = (t_np[-1] - t_np[0]) / (n_points - 1)
                integrand = time_scores.cpu().numpy()
                integral = integrand[0] + integrand[-1]
                for i in range(1, n_points - 1):
                    integral += (2 if i % 2 == 0 else 4) * integrand[i]
                integral *= h / 3
                return -torch.from_numpy(integral)
            return -time_scores.mean(dim=0).cpu()
        finally:
            if self.ema_b is not None:
                self.ema_b.restore(self.net_b)
            if self.ema_eta is not None:
                self.ema_eta.restore(self.net_eta)

    def _compute_time_score_single(self, t_scalar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """time-score at a single tau: -div(b) + <b, eta> / gamma; vmap-ready."""
        n = x.shape[0]
        t_batch = t_scalar.expand(n, 1)

        gamma_t = self.path.gamma(t_scalar)
        if gamma_t.dim() > 0:
            gamma_t = gamma_t.squeeze()

        b_pred = self.net_b(t_batch, x)
        eta_pred = self.net_eta(t_batch, x)

        def b_single(x_single):
            return self.net_b(t_scalar.view(1, 1), x_single.unsqueeze(0)).squeeze(0)

        if self.div_method == "exact":
            div_b = exact_div(b_single, x)
        else:
            div_b = hutch_div(b_single, x, noise=self.div_noise)
            for _ in range(self.n_hutch_samples - 1):
                div_b = div_b + hutch_div(b_single, x, noise=self.div_noise)
            div_b = div_b / self.n_hutch_samples
        b_dot_eta = (b_pred * eta_pred).sum(dim=-1)
        return -div_b + b_dot_eta / gamma_t
