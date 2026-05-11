"""TriangularCTSMV1: continuous-time score matching DRE on a barycentric triangular path.

the path has no forbidden support; any TimeSampler in TimeCfg is valid.
default is uniform tau sampling.
"""
from typing import Optional

from ...common._cfgs import OptimCfg, SchedCfg, EmaCfg, TimeCfg
from ._base import _TriangularCTSMBase
from src.waypoints.triangular_continuous import BarycentricCtsm1D


class TriangularCTSMV1(_TriangularCTSMBase):
    """CTSM with a barycentric triangular path (p0 -> p* -> p1)."""

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
        self.vertex = vertex
