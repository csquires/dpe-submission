"""TriangularCTSM V1/V2: continuous-time score matching DRE on a triangular path.

two user-facing classes:
  - ``TriangularCTSMV1``: barycentric path (``BarycentricCtsm1D``). default
    sampler ``UniformSampler``; any TimeSampler is valid since the path has no
    forbidden support.
  - ``TriangularCTSMV2``: piecewise Schroedinger-bridge path (``PiecewiseSBCtsm1D``)
    with a vertex-adjacent forbidden band. default sampler ``PathSampler`` so
    the band exclusion is honored; user can override with any TimeSampler at
    their own risk (sampling through the band yields biased gradients).

the two classes share a private ``_TriangularCTSMBase`` for fit / predict_ldr;
only the constructor differs (path construction and default TimeCfg).
"""
from typing import Optional

import torch

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._cfgs import (
    OptimCfg, SchedCfg, EmaCfg, TimeCfg,
    make_optim, make_sched, make_ema, make_time_sampler,
)
from src.density_ratio_estimation._ema import EMA
from src.density_ratio_estimation._time_samplers import PathSampler
from src.density_ratio_estimation._trainer import train_loop
from src.density_ratio_estimation._losses import make_sb_loss
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D
from src.waypoints.triangular_continuous import BarycentricCtsm1D
from src.waypoints.piecewise_sb import PiecewiseSBCtsm1D


class _TriangularCTSMBase(DensityRatioEstimator):
    """shared fit / predict_ldr for the two triangular-CTSM variants.

    constructors of V1 / V2 build their own path and pass it here along with
    a resolved TimeCfg. base owns training and inference; subclasses own
    construction.
    """

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
        sigma: float,
        activation: str,
        integration_steps: int,
        reweight: bool,
    ) -> None:
        super().__init__(input_dim)
        self.path = path
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
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
        sched_obj = make_sched(optim_obj, self.n_epochs, self.optim.lr, self.sched)
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
            n_steps=self.n_epochs,
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


class TriangularCTSMV1(_TriangularCTSMBase):
    """CTSM with a barycentric triangular path (p0 -> p* -> p1).

    the path has no forbidden support; any TimeSampler in TimeCfg is valid.
    default is uniform tau sampling.
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
        sigma: float = 1.0,
        vertex: float = 0.5,
        activation: str = "elu",
        integration_steps: int = 200,
        reweight: bool = False,
    ) -> None:
        """barycentric CTSM with cfg surface; path constructed internally."""
        path = BarycentricCtsm1D(sigma=sigma, vertex=vertex, eps=time.eps)
        super().__init__(
            input_dim=input_dim,
            path=path,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            n_epochs=n_epochs,
            batch_size=batch_size,
            optim=optim,
            sched=sched,
            ema=ema,
            time=time,
            device=device,
            sigma=sigma,
            activation=activation,
            integration_steps=integration_steps,
            reweight=reweight,
        )
        # expose path-specific knob for inspection (sigma already lives on base)
        self.vertex = vertex


class TriangularCTSMV2(_TriangularCTSMBase):
    """CTSM with a piecewise-Schroedinger-bridge triangular path.

    the path has a forbidden support band of width O(inner_eps) around tau=vertex
    where sb_target clamping produces biased gradients. default TimeCfg uses
    PathSampler so the band is automatically excluded at training time.

    explicit ``time=TimeCfg(sampler=...)`` is allowed but the user takes the bias
    if their sampler draws tau inside the band.
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
        time: Optional[TimeCfg] = None,
        device: Optional[str] = None,
        sigma: float = 1.0,
        vertex: float = 0.5,
        inner_eps: float = 0.02,
        activation: str = "elu",
        integration_steps: int = 200,
        reweight: bool = False,
    ) -> None:
        """piecewise-SB CTSM; default TimeCfg uses PathSampler for band safety.

        time=None (default) auto-builds TimeCfg(sampler=PathSampler(path=self.path)).
        pass time=TimeCfg(...) explicitly to override (e.g. for non-uniform
        sampling on the safe support); the user is responsible for band handling.
        """
        eps = time.eps if time is not None else 1e-3
        path = PiecewiseSBCtsm1D(sigma=sigma, vertex=vertex, eps=eps, inner_eps=inner_eps)
        if time is None:
            time = TimeCfg(sampler=PathSampler(path=path))
        super().__init__(
            input_dim=input_dim,
            path=path,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            n_epochs=n_epochs,
            batch_size=batch_size,
            optim=optim,
            sched=sched,
            ema=ema,
            time=time,
            device=device,
            sigma=sigma,
            activation=activation,
            integration_steps=integration_steps,
            reweight=reweight,
        )
        self.vertex = vertex
        self.inner_eps = inner_eps
