"""TriangularVFMV1: velocity field matching DRE on a barycentric path (k-parameterized gamma).

no forbidden support; any TimeSampler in TimeCfg is valid. default uniform.
"""
from typing import Optional, Literal

from ...common._cfgs import OptimCfg, SchedCfg, EmaCfg, TimeCfg
from ._base import _TriangularVFMBase
from src.waypoints.triangular_continuous import BarycentricVfm1D


class TriangularVFMV1(_TriangularVFMBase):
    """VFM with a barycentric triangular path (BarycentricVfm1D, k-parameterized gamma)."""

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
        k: float = 20.0,
        vertex: float = 0.5,
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
        """barycentric VFM; path constructed internally from k, vertex, time.eps."""
        path = BarycentricVfm1D(k=k, vertex=vertex, eps=time.eps)
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
        self.k = k
        self.vertex = vertex
