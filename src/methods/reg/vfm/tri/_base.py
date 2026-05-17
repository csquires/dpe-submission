"""shared two-phase fit + time-score integration for the triangular VFM variants.

constructors of V1 / V2 build their own path and resolve TimeCfg before calling
super().__init__. base owns net_b/net_eta training and inference.
"""
from typing import Optional, Literal
import warnings

import torch

from ....common.base import ELDR
from ...common._cfgs import (
    OptimCfg, SchedCfg, EmaCfg, TimeCfg,
    make_optim, make_sched, make_ema, make_time_sampler,
)
from ...common._trainer import train_two_phase
from ...common._losses import make_velo_loss, make_denoiser_loss
from ...common._integration import build_integrator
from src.models.common.mlp import MLP
from src.models.flow.div_estimators import build_div_fn


class _TriangularVFMBase(ELDR):
    """shared two-phase fit + time-score integration for triangular VFM."""

    def __init__(
        self,
        input_dim: int,
        path,
        hidden_dim: int,
        n_hidden_layers: int,
        n_epochs: int,
        batch_size: int,
        *,
        optim: OptimCfg,
        sched: SchedCfg,
        ema: EmaCfg,
        time: TimeCfg,
        device: Optional[str],
        antithetic: bool,
        div_method: Literal["hutchinson", "exact"],
        div_noise: Literal["rademacher", "gaussian"],
        n_hutch_samples: int,
        integration_steps: int,
        integration_type: Literal["1", "2", "3"],
        activation: str,
        layernorm: str,
        reweight: bool,
    ) -> None:
        if time.eps < 1e-3:
            raise ValueError(
                f"timecfg.eps must be >= 1e-3 for boundary regularity of b*eta/gamma; "
                f"got eps={time.eps}"
            )

        super().__init__(input_dim)
        self.path = path
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optim = optim
        self.sched = sched
        self.ema = ema
        self.time = time
        self.integration_steps = integration_steps
        self.integration_type = integration_type
        self.antithetic = antithetic

        if div_method not in ("hutchinson", "exact"):
            raise ValueError(f"div_method must be 'hutchinson' or 'exact'; got {div_method!r}")
        if div_noise not in ("rademacher", "gaussian"):
            raise ValueError(f"div_noise must be 'rademacher' or 'gaussian'; got {div_noise!r}")
        if n_hutch_samples < 1:
            raise ValueError(f"n_hutch_samples must be >= 1; got {n_hutch_samples}")
        self.div_method = div_method
        self.div_noise = div_noise
        self.n_hutch_samples = n_hutch_samples

        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )
        self.activation = activation
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off', 'pre', 'post'}}; got {layernorm!r}")
        self.layernorm = layernorm
        self.reweight = reweight

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._div_fn = build_div_fn(div_method, noise=div_noise, n_samples=n_hutch_samples)
        self._integrator = build_integrator(integration_type)

        self.net_b = None
        self.net_eta = None
        self.ema_b: Optional[object] = None
        self.ema_eta: Optional[object] = None

    def init_model(self) -> None:
        """build net_b and net_eta as MLPs; EMA is constructed in fit."""
        self.net_b = MLP(
            self.input_dim,
            self.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            layernorm=self.layernorm,
        ).to(self.device)
        self.net_eta = MLP(
            self.input_dim,
            self.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            layernorm=self.layernorm,
        ).to(self.device)

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """two-phase training; loss closures built once via factories."""
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

        optim_b = make_optim(self.net_b.parameters(), self.optim)
        optim_eta = make_optim(self.net_eta.parameters(), self.optim)
        sched_b = make_sched(optim_b, self.n_epochs, self.optim.lr, self.sched)
        sched_eta = make_sched(optim_eta, self.n_epochs, self.optim.lr, self.sched)
        ema_b = make_ema(self.net_b, self.ema)
        ema_eta = make_ema(self.net_eta, self.ema)
        self.ema_b = ema_b
        self.ema_eta = ema_eta

        time_sampler = make_time_sampler(self.time)

        loss_b = make_velo_loss(path=self.path, antithetic=self.antithetic, reweight=self.reweight)
        loss_eta = make_denoiser_loss(path=self.path, reweight=self.reweight)

        train_two_phase(
            model_b=self.net_b,
            model_eta=self.net_eta,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_b=loss_b,
            loss_eta=loss_eta,
            optim_b=optim_b,
            optim_eta=optim_eta,
            n_steps_b=self.n_epochs,
            n_steps_eta=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler_b=sched_b,
            scheduler_eta=sched_eta,
            ema_b=ema_b,
            ema_eta=ema_eta,
            grad_clip_norm_b=self.optim.grad_clip_norm,
            grad_clip_norm_eta=self.optim.grad_clip_norm,
            eps=self.time.eps,
        )

        self.net_b.eval()
        self.net_eta.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """integrate the time-score over tau in [eps, 1-eps] (chunked vmap); return -integral on cpu."""
        if self.net_b is None or self.net_eta is None:
            raise RuntimeError("triangular vfm not trained; call fit() before predict_ldr().")
        self.net_b.eval()
        self.net_eta.eval()
        samples = xs.float().to(self.device)
        n_samples = samples.shape[0]

        n_points = self.integration_steps
        if n_points % 2 == 0:
            n_points += 1
        t_vals = torch.linspace(self.time.eps, 1.0 - self.time.eps, steps=n_points, device=self.device)

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
            return self._integrator(time_scores, t_vals)
        finally:
            if self.ema_b is not None:
                self.ema_b.restore(self.net_b)
            if self.ema_eta is not None:
                self.ema_eta.restore(self.net_eta)

    def _compute_time_score_single(self, t_scalar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """time-score at a single t: -div(b) + <b, eta> / gamma; vmap-ready."""
        n = x.shape[0]
        t_batch = t_scalar.expand(n, 1)

        gamma_t = self.path.gamma(t_scalar)
        if gamma_t.dim() > 0:
            gamma_t = gamma_t.squeeze()

        b_pred = self.net_b(x, t_batch)
        eta_pred = self.net_eta(x, t_batch)

        def b_single(x_single):
            return self.net_b(x_single.unsqueeze(0), t_scalar.view(1, 1)).squeeze(0)

        div_b = self._div_fn(b_single, x)
        b_dot_eta = (b_pred * eta_pred).sum(dim=-1)
        return -div_b + b_dot_eta / gamma_t
