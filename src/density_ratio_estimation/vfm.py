"""velocity field model (VFM): denoiser-based stochastic-interpolant DRE.

trains b (velocity) and eta (denoiser) sequentially; integrates the time-score
-div(b) + b.eta/gamma over tau in [eps, 1-eps] to predict log(p0/p1).
"""
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import integrate

from src.density_ratio_estimation.base import DRE
from src.density_ratio_estimation._ema import EMA, maybe_clip_grad
from src.density_ratio_estimation._trainer import train_two_phase
# VFM exposes inline gamma / dgamma_dt rather than a Path object, so it bypasses
# velo_loss / denoiser_loss from `_losses` (which expect a path). fit() defines
# inline closures with the same loss math.
from src.models.flow.div_estimators import exact_div, hutch_div, compute_divergence
from src.models.common.mlp import MLP


class VFM(DRE):
    """VFM under `train_two_phase`: b then eta, with vmap-based time-score integration."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 2e-3,
        k: float = 0.5,
        n_t: int = 50,
        eps: float = 0.01,
        device: Optional[str] = None,
        integration_steps: int = 10000,
        integration_type: Literal['1', '2', '3'] = '1',
        verbose: bool = False,
        log_every: int = 100,
        antithetic: bool = False,
        div_method: Literal['hutchinson', 'exact'] = 'hutchinson',
        div_noise: Literal['rademacher', 'gaussian'] = 'rademacher',
        n_hutch_samples: int = 1,
        ema_decay: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        activation: str = "silu",
    ):
        super().__init__(input_dim)
        self.integration_type = integration_type
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.k = k
        self.n_t = n_t
        self.eps = eps
        self.integration_steps = integration_steps
        self.verbose = verbose
        self.log_every = log_every
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
        self.ema_decay = ema_decay
        self.grad_clip_norm = grad_clip_norm
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )
        self.activation = activation

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.net_b = None
        self.net_eta = None
        self.ema_b: Optional[EMA] = None
        self.ema_eta: Optional[EMA] = None

    def init_model(self) -> None:
        """instantiate net_b (velocity) and net_eta (denoiser) MLPs, plus optional EMA wrappers."""
        self.net_b = MLP(self.input_dim, self.hidden_dim,
                         output_dim=self.input_dim,
                         n_hidden_layers=self.n_hidden_layers,
                         activation=self.activation).to(self.device)
        self.net_eta = MLP(self.input_dim, self.hidden_dim,
                           output_dim=self.input_dim,
                           n_hidden_layers=self.n_hidden_layers,
                           activation=self.activation).to(self.device)
        self.ema_b = EMA(self.net_b, self.ema_decay) if self.ema_decay is not None else None
        self.ema_eta = EMA(self.net_eta, self.ema_decay) if self.ema_decay is not None else None

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """gamma(t) = (1 - exp(-k t)) (1 - exp(-k (1-t)))."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        return (1 - torch.exp(-self.k * t)) * (1 - torch.exp(-self.k * (1 - t)))

    def dgamma_dt(self, t: torch.Tensor) -> torch.Tensor:
        """gamma'(t)."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        e0 = torch.exp(-self.k * t)
        e1 = torch.exp(-self.k * (1 - t))
        return self.k * e0 * (1 - e1) - self.k * e1 * (1 - e0)

    def _compute_time_score_single(self, t_scalar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """time-score at a single tau: -div(b) + <b, eta> / gamma; vmap-ready."""
        n = x.shape[0]
        t_batch = t_scalar.expand(n, 1)
        gamma_t = self.gamma(t_scalar)

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

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """train b then eta sequentially via `train_two_phase`."""
        self.init_model()

        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)

        if self.verbose:
            print(f"[VFM] starting train_two_phase.")
            print(f"[VFM] gamma range: [{self.gamma(torch.tensor(self.eps)).item():.4f}, {self.gamma(torch.tensor(0.5)).item():.4f}]")

        optim_b = optim.Adam(self.net_b.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        optim_eta = optim.Adam(self.net_eta.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        def time_sampler(B, eps, device):
            return torch.rand(B, 1, device=device) * (1 - 2 * eps) + eps, torch.ones(B, 1, device=device)

        antithetic = self.antithetic

        def loss_b(model, batch, tau, iw):
            x0, x1 = batch["x0"], batch["x1"]
            z = torch.randn_like(x0)
            g = self.gamma(tau)
            dg = self.dgamma_dt(tau)
            mu = (1 - tau) * x0 + tau * x1
            delta = x1 - x0
            if antithetic:
                b_p = model(tau, mu + g * z)
                b_m = model(tau, mu - g * z)
                v_p = delta + dg * z
                v_m = delta - dg * z
                lp = 0.5 * (b_p ** 2).sum(-1) - (v_p * b_p).sum(-1)
                lm = 0.5 * (b_m ** 2).sum(-1) - (v_m * b_m).sum(-1)
                return 0.5 * (lp + lm).mean()
            b = model(tau, mu + g * z)
            v_star = delta + dg * z
            return (0.5 * (b ** 2).sum(-1) - (v_star * b).sum(-1)).mean()

        loss_b.required_keys = frozenset({"x0", "x1"})
        loss_b.requires_tau_grad = False

        def loss_eta(model, batch, tau, iw):
            x0, x1 = batch["x0"], batch["x1"]
            z = torch.randn_like(x0)
            x_t = (1 - tau) * x0 + tau * x1 + self.gamma(tau) * z
            eta = model(tau, x_t)
            return (0.5 * (eta ** 2).sum(-1) - (z * eta).sum(-1)).mean()

        loss_eta.required_keys = frozenset({"x0", "x1"})
        loss_eta.requires_tau_grad = False

        train_two_phase(
            model_b=self.net_b,
            model_eta=self.net_eta,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_b=loss_b,
            loss_eta=loss_eta,
            optim_b=optim_b,
            optim_eta=optim_eta,
            n_steps_b=self.n_epochs,
            n_steps_eta=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            ema_b=self.ema_b,
            ema_eta=self.ema_eta,
            grad_clip_norm_b=self.grad_clip_norm,
            grad_clip_norm_eta=self.grad_clip_norm,
            eps=self.eps,
        )

        if self.verbose:
            print("[VFM] training complete")

        self.net_b.eval()
        self.net_eta.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """integrate the time-score over tau in [eps, 1-eps]; return -integral on CPU.

        chunked vmap over the tau grid; integration_type selects mean / trapz / Simpson.
        """
        if self.net_b is None or self.net_eta is None:
            raise RuntimeError("VFM model is not trained. Call fit() before predict_ldr().")

        self.net_b.eval()
        self.net_eta.eval()
        samples = xs.float().to(self.device)
        n_samples = samples.shape[0]

        n_points = self.integration_steps
        if n_points % 2 == 0:
            n_points += 1
        t_vals = torch.linspace(self.eps, 1 - self.eps, n_points, device=self.device)

        if self.ema_b is not None:
            self.ema_b.apply_to(self.net_b)
        if self.ema_eta is not None:
            self.ema_eta.apply_to(self.net_eta)

        try:
            time_score_fn = torch.vmap(
                self._compute_time_score_single,
                in_dims=(0, None),
                out_dims=0,
                randomness="different",
            )
            chunk_size = max(1, 100000 // n_samples)
            chunks = []
            for i in range(0, n_points, chunk_size):
                chunks.append(time_score_fn(t_vals[i:i + chunk_size], samples).detach())
            time_scores = torch.cat(chunks, dim=0)

            if self.integration_type == "3":
                t_np = t_vals.cpu().numpy()
                h = (t_np[-1] - t_np[0]) / (n_points - 1)
                integrand = time_scores.cpu().numpy()
                integral = integrand[0] + integrand[-1]
                for i in range(1, n_points - 1):
                    integral += (2 if i % 2 == 0 else 4) * integrand[i]
                integral *= h / 3
                return -torch.from_numpy(integral)
            if self.integration_type == "1":
                return -time_scores.mean(dim=0).cpu()
            return -torch.trapz(time_scores, t_vals, dim=0).cpu()
        finally:
            if self.ema_b is not None:
                self.ema_b.restore(self.net_b)
            if self.ema_eta is not None:
                self.ema_eta.restore(self.net_eta)


SpatialVeloDenoiser = VFM
