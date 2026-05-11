"""TriangularTSM: time-score matching DRE on a bell-shaped path."""
from typing import Optional, Tuple
from math import ceil

import torch

from ...common.base import ELDR
from ..common._trainer import train_loop
from ..common._losses import tri_tsm_loss
from ..common._cfgs import (
    OptimCfg,
    SchedCfg,
    EmaCfg,
    TimeCfg,
    make_optim,
    make_sched,
    make_ema,
    make_time_sampler,
)
from src.models.time_score_matching.time_score_net_2d import TimeScoreNetwork2D


class TriangularTSM(ELDR):
    """time-score matching DRE under `tri_tsm_loss` on a piecewise-quadratic bell path.

    bell: t' = peak_max (1 - ((tau - vertex)/scale)^2) on (0, vertex) and (vertex, 1).
    integrates -score over a tau grid at inference.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_epochs: int = 1000,
        batch_size: int = 512,
        *,
        optim: OptimCfg,
        sched: SchedCfg = SchedCfg(),
        ema: EmaCfg = EmaCfg(),
        time: TimeCfg = TimeCfg(),
        device: Optional[str] = None,
        # TriTSM-specific (explicit kwargs)
        reweight: bool = False,
        vertex: float = 0.5,
        peak_max: float = 1.0,
        activation: str = "silu",
    ) -> None:
        """bell path: t' = peak_max (1 - ((tau - vertex)/scale)^2) on (0, vertex) and (vertex, 1).

        args:
            input_dim: feature dimension.
            hidden_dim: width of hidden layers. default 256.
            n_hidden_layers: depth. default 3.
            n_epochs: training epochs. default 1000.
            batch_size: batch size. default 512.
            optim: optimizer config (lr, weight_decay, grad_clip_norm, etc).
            sched: scheduler config (name, cosine_min_factor, etc). default SchedCfg().
            ema: exponential moving average config. default EmaCfg().
            time: time sampling config (eps, flavor). default TimeCfg().
            device: torch device string. auto-detect if None.
            reweight: whether to reweight loss. default False.
            vertex: peak location in (0, 1). default 0.5.
            peak_max: peak height in (0, 1]. default 1.0.
            activation: nonlinearity in {'elu', 'gelu', 'silu'}. default 'silu'.
        """
        super().__init__(input_dim)

        # validate params
        if not 0.0 < vertex < 1.0:
            raise ValueError(f"vertex must be in (0, 1), got {vertex}")
        if not 0.0 < peak_max <= 1.0:
            raise ValueError(f"peak_max must be in (0, 1], got {peak_max}")
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(f"activation must be in {{'elu', 'gelu', 'silu'}}, got {activation!r}")

        # store hyperparameters
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reweight = reweight
        self.vertex = vertex
        self.peak_max = peak_max
        self.activation = activation

        # store cfg objects
        self.optim = optim
        self.sched = sched
        self.ema = ema
        self.time = time

        # device handling
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # model (lazy-initialized in _init_model)
        self.model = None

    def _init_model(self) -> None:
        """instantiate TimeScoreNetwork2D. optimizer created in fit via factory."""
        self.model = TimeScoreNetwork2D(
            self.input_dim,
            self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
        ).to(self.device)

    def _path_t_tprime(self, tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """piecewise-quadratic bell at tau: t' is 0 at endpoints and peak_max at vertex."""
        t = torch.clamp(tau, min=self.time.eps, max=1.0)
        v, m = self.vertex, self.peak_max
        left = m * (2.0 * (tau / v) - (tau / v) ** 2)
        right = m * (1.0 - ((tau - v) / (1.0 - v)) ** 2)
        t_prime = torch.clamp(torch.where(tau <= v, left, right), min=0.0, max=1.0)
        return t, t_prime

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """init model, build optim/scheduler/ema/time_sampler from cfg, then delegate to train_loop."""
        self._init_model()
        self.model.train()

        # TriTSM-specific n_steps formula: based on smallest sample set
        min_size = min(samples_p0.shape[0], samples_p1.shape[0], samples_pstar.shape[0])
        n_steps = self.n_epochs * ceil(min_size / self.batch_size)

        # build optimizer, scheduler, ema, time_sampler from cfg
        optim_obj = make_optim(self.model.parameters(), self.optim)
        sched_obj = make_sched(optim_obj, n_steps, self.optim.lr, self.sched)
        ema_obj = make_ema(self.model, self.ema)
        time_sampler = make_time_sampler(self.time)

        train_loop(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_fn=tri_tsm_loss,
            optim=optim_obj,
            n_steps=n_steps,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler=sched_obj,
            ema=ema_obj,
            grad_clip_norm=self.optim.grad_clip_norm,
            eps=self.time.eps,
            loss_kwargs={
                "reweight": self.reweight,
                "eps": self.time.eps,
                "vertex": self.vertex,
                "peak_max": self.peak_max,
            },
        )

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """trapezoid-integrate -model(xs, t, t') over a 100-point tau grid in [eps, 1]."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() before predict_ldr().")

        self.model.eval()
        samples = xs.float().to(self.device)
        n = samples.shape[0]
        if n == 0:
            return torch.zeros(0, dtype=samples.dtype, device=self.device)

        with torch.no_grad():
            tau_grid = torch.linspace(self.time.eps, 1.0, 100, device=self.device)
            scores = []
            for tau_scalar in tau_grid:
                t, t_prime = self._path_t_tprime(tau_scalar.view(1, 1))
                score = self.model(samples, t.expand(n, 1), t_prime.expand(n, 1))
                scores.append(-score.squeeze(-1))
            return torch.trapezoid(torch.stack(scores, dim=0).t(), tau_grid, dim=1)
