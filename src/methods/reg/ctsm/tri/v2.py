"""TriangularCTSMV2: continuous-time score matching DRE on a piecewise-Schroedinger-bridge path.

the path has a forbidden support band of width O(inner_eps) around tau=vertex
where sb_target clamping produces biased gradients. default TimeCfg uses
PathSampler so the band is automatically excluded at training time.

explicit ``time=TimeCfg(sampler=...)`` is allowed but the user takes the bias
if their sampler draws tau inside the band.
"""
from typing import Optional

from ...common._cfgs import OptimCfg, SchedCfg, EmaCfg, TimeCfg
from ...common._time_samplers import PathSampler
from ._base import _TriangularCTSMBase
from src.waypoints.piecewise_sb import PiecewiseSBCtsm1D


class TriangularCTSMV2(_TriangularCTSMBase):
    """CTSM with a piecewise-Schroedinger-bridge triangular path."""

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
        """piecewise-SB CTSM; default TimeCfg uses PathSampler for band safety."""
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
