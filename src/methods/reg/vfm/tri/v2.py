"""TriangularVFMV2: VFM on a piecewise Schroedinger-bridge path (floored gamma).

when ``inner_eps > 0`` the path defines a forbidden support band around
tau=vertex. default TimeCfg uses PathSampler so the band is excluded.
user may override; in that case they take any band-related bias.
"""
from typing import Optional, Literal

from ...common._cfgs import OptimCfg, SchedCfg, EmaCfg, TimeCfg
from ...common._time_samplers import PathSampler
from ._base import _TriangularVFMBase
from src.waypoints.piecewise_sb import PiecewiseSBVfm1D


class TriangularVFMV2(_TriangularVFMBase):
    """VFM with a piecewise-Schroedinger-bridge path."""

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
        gamma_min: float = 5e-2,
        inner_eps: float = 0.0,
        antithetic: bool = True,
        div_method: Literal["hutchinson", "exact"] = "hutchinson",
        div_noise: Literal["rademacher", "gaussian"] = "rademacher",
        n_hutch_samples: int = 1,
        integration_steps: int = 3000,
        integration_type: Literal["1", "2", "3"] = "2",
        activation: str = "gelu",
        layernorm: str = "off",
        reweight: bool = False,
    ) -> None:
        """piecewise-SB VFM; default TimeCfg uses PathSampler for band safety."""
        eps = time.eps if time is not None else 1e-3
        path = PiecewiseSBVfm1D(
            sigma=sigma, vertex=vertex, gamma_min=gamma_min, eps=eps, inner_eps=inner_eps,
        )
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
            antithetic=antithetic,
            div_method=div_method,
            div_noise=div_noise,
            n_hutch_samples=n_hutch_samples,
            integration_steps=integration_steps,
            integration_type=integration_type,
            activation=activation,
            layernorm=layernorm,
            reweight=reweight,
        )
        self.sigma = sigma
        self.vertex = vertex
        self.gamma_min = gamma_min
        self.inner_eps = inner_eps
