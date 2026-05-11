"""
velocity field model (VFM): denoiser-based stochastic interpolant density ratio estimator.

sequential-only variant with vmap-based divergence computation.
uses create_graph=False for efficiency during inference.
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
# NOTE: VFM (2-source) does NOT use velo_matching_loss / denoiser_loss from
# `_losses` because those losses expect a 3-source `VfmPath1D` instance, while
# VFM has inline `gamma()` / `dgamma_dt()` methods (no Path object). We define
# inline closures inside `fit()` instead. The 3-source TriangularVFM and its
# 2D counterpart use the path-based losses.
from src.models.flow.div_estimators import exact_div, hutch_div, compute_divergence
from src.models.common.mlp import MLP


class VFM(DRE):
    """
    velocity field model (VFM): denoiser-based stochastic interpolant density ratio estimator.

    renamed from SpatialVeloDenoiser. combines velocity (b) and denoiser (eta) networks
    trained sequentially via train_two_phase. inference uses vmap-accelerated divergence
    computation and time-score integration.

    key properties:
      - sequential b-then-eta training (no simultaneous mode).
      - efficient divergence estimation via hutchinson or exact methods.
      - vmap-based time-score computation in predict_ldr.
      - backward compatible: SpatialVeloDenoiser = VFM alias at module bottom.
    """
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
        """build net_b and net_eta as MLP networks.

        net_b: velocity field [input_dim] -> [input_dim]
        net_eta: denoiser field [input_dim] -> [input_dim]

        also initializes EMA wrappers if ema_decay is not None.
        """
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
        """compute gamma(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t))).

        used to scale noise in the stochastic interpolant x_t = I_t + gamma(t)*z.

        args:
            t: time values (scalar or tensor).

        returns:
            gamma(t) (same shape as t).
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        return (1 - torch.exp(-self.k * t)) * (1 - torch.exp(-self.k * (1 - t)))

    def dgamma_dt(self, t: torch.Tensor) -> torch.Tensor:
        """compute gamma'(t), the time derivative of gamma(t).

        args:
            t: time values (scalar or tensor).

        returns:
            gamma'(t) (same shape as t).
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        exp_kt = torch.exp(-self.k * t)
        exp_k1t = torch.exp(-self.k * (1 - t))
        return self.k * exp_kt * (1 - exp_k1t) - self.k * exp_k1t * (1 - exp_kt)

    def _compute_time_score_single(self, t_scalar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """compute time score for a single time value across all samples.

        uses vmap over jacrev to compute divergence efficiently per-sample,
        avoiding the loop in compute_divergence when called from predict_ldr.

        formula: d_t log rho(t,x) = -div(b) + b·eta/gamma

        args:
            t_scalar: single time value (scalar tensor).
            x: sample points [n_samples, dim].

        returns:
            time scores [n_samples].
        """
        n_samples = x.shape[0]
        t_batch = t_scalar.expand(n_samples, 1)

        gamma_t = self.gamma(t_scalar)

        b_pred = self.net_b(t_batch, x)
        eta_pred = self.net_eta(t_batch, x)

        def b_single(x_single):
            return self.net_b(t_scalar.view(1, 1), x_single.unsqueeze(0)).squeeze(0)

        if self.div_method == 'exact':
            div_b = exact_div(b_single, x)  # [n_samples]
        else:
            div_b = hutch_div(b_single, x, noise=self.div_noise)
            for _ in range(self.n_hutch_samples - 1):
                div_b = div_b + hutch_div(b_single, x, noise=self.div_noise)
            div_b = div_b / self.n_hutch_samples
        b_dot_eta = (b_pred * eta_pred).sum(dim=-1)  # [n_samples]

        return -div_b + b_dot_eta / gamma_t  # [n_samples]

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """train b and eta networks sequentially via train_two_phase.

        procedure:
          1. initialize both networks via init_model.
          2. call train_two_phase with:
             - phase 1: train net_b (eta frozen) using velo_matching_loss.
             - phase 2: train net_eta (b frozen) using denoiser_loss.

        args:
            samples_p0: samples from source distribution [N0, D].
            samples_p1: samples from target distribution [N1, D].
        """
        self.init_model()

        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)

        if self.verbose:
            print(f"[VFM] starting sequential training (train_two_phase).")
            print(f"[VFM] gamma range: [{self.gamma(torch.tensor(self.eps)).item():.4f}, {self.gamma(torch.tensor(0.5)).item():.4f}]")

        # build optimizers for both phases
        optim_b = optim.Adam(self.net_b.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        optim_eta = optim.Adam(self.net_eta.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        # time sampler: uniform in [eps, 1-eps] with unit importance weights
        def time_sampler(batch_size, eps, device):
            tau = torch.rand(batch_size, 1, device=device) * (1 - 2*eps) + eps
            iw = torch.ones(batch_size, 1, device=device)
            return tau, iw

        # closures over self.gamma / self.dgamma_dt; matches trainer's
        # loss_fn(model, batch, tau, iw) -> scalar contract.
        antithetic = self.antithetic

        def loss_b_2src(model, batch, tau, iw):
            x0, x1 = batch["x0"], batch["x1"]                          # [B, D] each
            z = torch.randn_like(x0)                                   # [B, D]
            gamma_t = self.gamma(tau)                                  # [B, 1]
            gamma_prime_t = self.dgamma_dt(tau)                        # [B, 1]
            i_t = (1 - tau) * x0 + tau * x1                            # [B, D] linear interpolant
            dt_it = x1 - x0                                            # [B, D] dI/dt

            if antithetic:
                x_t_plus = i_t + gamma_t * z
                x_t_minus = i_t - gamma_t * z
                b_plus = model(tau, x_t_plus)
                b_minus = model(tau, x_t_minus)
                # antithetic VFM b-loss: 0.5||b||^2 - <dI/dt + gamma'*z, b>
                target_plus = dt_it + gamma_prime_t * z
                target_minus = dt_it - gamma_prime_t * z
                loss_plus = 0.5 * (b_plus ** 2).sum(-1) - (target_plus * b_plus).sum(-1)
                loss_minus = 0.5 * (b_minus ** 2).sum(-1) - (target_minus * b_minus).sum(-1)
                return 0.5 * (loss_plus + loss_minus).mean()
            else:
                x_t = i_t + gamma_t * z
                b_pred = model(tau, x_t)
                target = dt_it + gamma_prime_t * z
                return (0.5 * (b_pred ** 2).sum(-1) - (target * b_pred).sum(-1)).mean()

        loss_b_2src.required_keys = frozenset({"x0", "x1"})
        loss_b_2src.requires_tau_grad = False

        def loss_eta_2src(model, batch, tau, iw):
            x0, x1 = batch["x0"], batch["x1"]
            z = torch.randn_like(x0)
            gamma_t = self.gamma(tau)
            i_t = (1 - tau) * x0 + tau * x1
            x_t = i_t + gamma_t * z
            eta_pred = model(tau, x_t)
            # denoiser loss: 0.5||eta||^2 - <z, eta>  (target = z)
            return (0.5 * (eta_pred ** 2).sum(-1) - (z * eta_pred).sum(-1)).mean()

        loss_eta_2src.required_keys = frozenset({"x0", "x1"})
        loss_eta_2src.requires_tau_grad = False

        # call train_two_phase with the inline closure losses
        train_two_phase(
            model_b=self.net_b,
            model_eta=self.net_eta,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_b=loss_b_2src,
            loss_eta=loss_eta_2src,
            optim_b=optim_b,
            optim_eta=optim_eta,
            n_steps_b=self.n_epochs,
            n_steps_eta=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler_b=None,
            scheduler_eta=None,
            ema_b=self.ema_b,
            ema_eta=self.ema_eta,
            grad_clip_norm_b=self.grad_clip_norm,
            grad_clip_norm_eta=self.grad_clip_norm,
            eps=self.eps,
            loss_kwargs_b={},
            loss_kwargs_eta={},
        )

        if self.verbose:
            print(f"[VFM] training complete")

        self.net_b.eval()
        self.net_eta.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """estimate log(p0/p1) by integrating time score over [eps, 1-eps].

        uses vmap to efficiently compute time scores across all time points.
        processes in chunks to avoid OOM with large n_points * n_samples.

        procedure:
            1. evaluate time_score at n_points in [eps, 1-eps] via vmap over t.
            2. integrate via trapezoidal (type='2'), simpson (type='3'), or mean (type='1').
            3. return negative integral (ldr = -int_0^1 d_t log rho dt).

        args:
            xs: sample points [n_samples, dim].

        returns:
            log density ratio estimates [n_samples].
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

        # if EMA is active, swap in shadow weights
        if self.ema_b is not None:
            self.ema_b.apply_to(self.net_b)
        if self.ema_eta is not None:
            self.ema_eta.apply_to(self.net_eta)

        try:
            compute_vmapped = torch.vmap(
                self._compute_time_score_single,
                in_dims=(0, None),  # batch over t, broadcast samples
                out_dims=0,
                randomness='different',  # required for Hutchinson noise inside
            )
            chunk_size = max(1, 100000 // n_samples)
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
                out = -torch.from_numpy(integral)
            elif self.integration_type == '1':
                out = -time_scores.mean(dim=0).cpu()
            elif self.integration_type == '2':
                out = -torch.trapz(time_scores, t_vals, dim=0).cpu()

            return out
        finally:
            if self.ema_b is not None:
                self.ema_b.restore(self.net_b)
            if self.ema_eta is not None:
                self.ema_eta.restore(self.net_eta)


# backward-compat alias
SpatialVeloDenoiser = VFM
