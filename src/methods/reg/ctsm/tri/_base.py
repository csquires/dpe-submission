"""shared fit / predict_ldr for the two triangular-CTSM variants.

constructors of V1 / V2 build their own path and pass it here along with a
resolved TimeCfg. base owns training and inference; subclasses own construction.
"""
from typing import Optional

import torch

from ....common.base import ELDR
from ...common._cfgs import (
    OptimCfg, SchedCfg, EmaCfg, TimeCfg,
    make_optim, make_sched, make_ema, make_time_sampler,
)
from ...common._ema import EMA
from ...common._trainer import train_loop
from ...common._losses import make_sb_loss
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D


class _TriangularCTSMBase(ELDR):
    """shared fit / predict_ldr for triangular-CTSM variants; see module docstring."""

    def __init__(
        self,
        input_dim: int,
        path,
        hidden_dim: int,
        n_hidden_layers: int,
        n_steps: int,
        batch_size: int,
        *,
        optim: OptimCfg,
        sched: SchedCfg,
        ema: EmaCfg,
        time: TimeCfg,
        device: Optional[str],
        sigma: float,
        activation: str,
        integration_steps: int,
        reweight: bool,
    ) -> None:
        super().__init__(input_dim)
        self.path = path
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.sigma = sigma
        self.integration_steps = integration_steps
        self.reweight = reweight

        self.optim = optim
        self.sched = sched
        self.ema = ema
        self.time = time

        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )
        self.activation = activation

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.ema_obj: Optional[EMA] = None

    def init_model(self) -> None:
        """build TimeScoreNetwork1D; optimizer/EMA are created in fit."""
        self.model = TimeScoreNetwork1D(
            self.input_dim,
            self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
        ).to(self.device)

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """build network, optim/sched/ema/time-sampler from cfgs, delegate to train_loop."""
        self.init_model()
        optim_obj = make_optim(self.model.parameters(), self.optim)
        sched_obj = make_sched(optim_obj, self.n_steps, self.optim.lr, self.sched)
        self.ema_obj = make_ema(self.model, self.ema)
        time_sampler = make_time_sampler(self.time)
        loss_fn = make_sb_loss(path=self.path, reweight=self.reweight)

        train_loop(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_fn=loss_fn,
            optim=optim_obj,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler=sched_obj,
            ema=self.ema_obj,
            grad_clip_norm=self.optim.grad_clip_norm,
            eps=self.time.eps,
        )

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """trapezoid-integrate -model over tau in [eps, 1-eps]; uses EMA shadow if set."""
        if self.model is None:
            raise RuntimeError("model not fitted; call fit() first.")
        self.model.eval()
        xs = xs.to(self.device)
        n = xs.shape[0]

        if self.ema_obj is not None:
            self.ema_obj.apply_to(self.model)
        try:
            ts = torch.linspace(
                self.time.eps, 1.0 - self.time.eps, self.integration_steps, device=self.device,
            )
            with torch.no_grad():
                vals = torch.stack([
                    -self.model(xs, torch.full((n, 1), float(t.item()), device=self.device)).squeeze(-1)
                    for t in ts
                ])
            dt = (1.0 - 2.0 * self.time.eps) / (self.integration_steps - 1)
            return torch.trapezoid(vals, dx=dt, dim=0).cpu()
        finally:
            if self.ema_obj is not None:
                self.ema_obj.restore(self.model)
