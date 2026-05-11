"""TriangularCTSM: continuous-time score matching DRE on a barycentric path."""
from typing import Optional

import torch

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._cfgs import (
    OptimCfg, SchedCfg, EmaCfg, TimeCfg,
    make_optim, make_sched, make_ema, make_time_sampler
)
from src.density_ratio_estimation._ema import EMA
from src.density_ratio_estimation._trainer import train_loop
from src.density_ratio_estimation._losses import sb_loss
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D
from src.waypoints.path_1d import CtsmPath1D
from src.waypoints.triangular_continuous import BarycentricCtsm1D
from src.waypoints.piecewise_sb import PiecewiseSBCtsm1D


class TriangularCTSM(DensityRatioEstimator):
    """CTSM with path p0 -> p* -> p1 (supports barycentric, piecewise, or custom CtsmPath1D).

    trains s_phi(x, tau) under `sb_loss(path=self.path)`; integrates -score over
    tau in [eps, 1-eps] at inference. if path exposes sample_tau, it overrides
    time_dist at training time (guarded by path-conflict check).
    """

    def __init__(
        self,
        input_dim: int,
        path: Optional[CtsmPath1D] = None,
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
        # TriangularCTSM-specific (explicit)
        sigma: float = 1.0,
        activation: str = "elu",
        integration_steps: int = 200,
        reweight: bool = False,
    ) -> None:
        """cfg-based constructor for path-based continuous time-score matching DRE.

        Args:
            input_dim: feature dimension.
            path: CtsmPath1D subclass (defaults to BarycentricCtsm1D with eps from TimeCfg).
            hidden_dim: hidden layer width.
            n_hidden_layers: count of hidden layers in TimeScoreNetwork1D.
            n_epochs: training steps.
            batch_size: batch size.
            optim: optimizer config (required keyword-only).
            sched: scheduler config (defaults to no scheduling).
            ema: EMA config (defaults to no EMA).
            time: time-sampling config (defaults to uniform).
            device: torch device; auto-detects GPU if None.
            sigma: noise scale for sb_loss.
            activation: nonlinearity {"elu", "gelu", "silu"}.
            integration_steps: discretization points for trapezoid integration [eps, 1-eps].
            reweight: passed to sb_loss for path-aware reweighting.

        Raises:
            ValueError: if path.sample_tau is callable and time.dist != "uniform".
        """
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.sigma = sigma
        self.integration_steps = integration_steps
        self.reweight = reweight

        # cfg objects
        self.optim = optim
        self.sched = sched
        self.ema = ema
        self.time = time

        # activation validation
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )
        self.activation = activation

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # path resolution: default to BarycentricCtsm1D with eps from TimeCfg
        if path is None:
            self.path = BarycentricCtsm1D(sigma=1.0, vertex=0.5, eps=self.time.eps)
        else:
            self.path = path

        # path-conflict guard: sample_tau requires uniform time distribution
        if callable(getattr(self.path, "sample_tau", None)) and self.time.dist != "uniform":
            raise ValueError(
                f"TimeCfg.dist must be 'uniform' when path provides sample_tau "
                f"(got dist={self.time.dist!r}; path is {type(self.path).__name__})"
            )

        # model and EMA instance attributes (set by init_model/fit)
        self.model = None
        self.ema_obj: Optional[EMA] = None

    def init_model(self) -> None:
        """build TimeScoreNetwork1D only; optimizer/EMA now created by fit."""
        self.model = TimeScoreNetwork1D(
            self.input_dim,
            self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation
        ).to(self.device)

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor, samples_pstar: torch.Tensor) -> None:
        """build network, optimizer, scheduler, EMA, time sampler; delegate to train_loop.

        if path.sample_tau is callable, time sampler defers to it; else uses make_time_sampler.
        """
        self.init_model()
        optim_obj = make_optim(self.model.parameters(), self.optim)
        sched_obj = make_sched(optim_obj, self.n_epochs, self.optim.lr, self.sched)
        self.ema_obj = make_ema(self.model, self.ema)

        # defer to path.sample_tau if available, else use factory
        sampler = getattr(self.path, "sample_tau", None)
        if callable(sampler):
            def time_sampler(B: int, eps: float, device) -> tuple[torch.Tensor, torch.Tensor]:
                return sampler(B, eps, device), torch.ones(B, 1, device=device)
        else:
            time_sampler = make_time_sampler(self.time)

        train_loop(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_fn=sb_loss,
            optim=optim_obj,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler=sched_obj,
            ema=self.ema_obj,
            grad_clip_norm=self.optim.grad_clip_norm,
            eps=self.time.eps,
            loss_kwargs={"sigma": self.sigma, "path": self.path, "reweight": self.reweight},
        )

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """trapezoid-integrate -model(xs, tau) over tau in [eps, 1-eps]; uses EMA shadow if set.

        Reads self.time.eps (not self.eps), self.ema_obj, self.integration_steps.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        xs = xs.to(self.device)
        n = xs.shape[0]

        if self.ema_obj is not None:
            self.ema_obj.apply_to(self.model)
        try:
            ts = torch.linspace(self.time.eps, 1.0 - self.time.eps, self.integration_steps, device=self.device)
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
