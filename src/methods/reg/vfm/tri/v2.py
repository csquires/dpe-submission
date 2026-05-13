"""TriangularVFMV2: VFM on a piecewise Schroedinger-bridge path (floored gamma).

when ``inner_eps > 0`` the path defines a forbidden support band around
tau=vertex. default time sampler uses piecewise_sb_sampler to avoid the band.
user may override both path and time sampler; in that case they take
responsibility for band-related bias.
"""
from typing import Optional, Literal

import torch

from ....common.base import ELDR
from ...common._cfgs import OptimCfg, SchedCfg, EmaCfg
from ...common._estimator_helpers import _validate_and_store_slots
from ...common._paradigm_funcs import vfm_time_score_1d
from ...common._predict_ldr import predict_ldr_via_curve
from ...common._losses import make_velo_loss, make_denoiser_loss
from ...common._trainer import train_two_phase
from ...common._time_samplers import (
    TimeSampler1D, make_uniform, make_piecewise_sb_sampler,
)
from ...common._curves import Curve, IdentityCurve1D
from ...common._integrators import Integrator, integrator_trapezoid
from src.models.flow.div_estimators import build_div_fn
from src.models.common.mlp import MLP
from src.waypoints.dataclass_paths import TriangularPath1D
from src.waypoints.path_builders import piecewise_sb_triangular_path_1d


def build_velocity_net(
    input_dim: int,
    hidden_dim: int,
    n_hidden_layers: int,
    activation: str,
    layernorm: str,
    device: torch.device,
) -> MLP:
    """build velocity field network (b)."""
    net = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        n_hidden_layers=n_hidden_layers,
        activation=activation,
        layernorm=layernorm,
    )
    return net.to(device)


def build_denoiser_net(
    input_dim: int,
    hidden_dim: int,
    n_hidden_layers: int,
    activation: str,
    layernorm: str,
    device: torch.device,
) -> MLP:
    """build denoiser network (eta)."""
    net = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        n_hidden_layers=n_hidden_layers,
        activation=activation,
        layernorm=layernorm,
    )
    return net.to(device)


class TriangularVFMV2(ELDR):
    """vfm dre with piecewise-schroedinger-bridge path.

    trains velocity field (b) and denoiser (eta) sequentially on three
    distributions p_0, p_1, p_*. inference integrates the time-score
    along curve from tau=eps to 1-eps. when inner_eps > 0, path defines
    a forbidden band and default time sampler avoids it.

    contract: fit(samples_p0, samples_p1, samples_pstar) with three [n, d]
    tensors; predict_ldr(xs) returns log(p_0/p_1) as [n_samples] cpu tensor.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        path: Optional[TriangularPath1D] = None,
        time: Optional[TimeSampler1D] = None,
        curve: Curve = IdentityCurve1D(),
        integrator: Integrator = integrator_trapezoid,
        sigma: float = 1.0,
        vertex: float = 0.5,
        gamma_min: float = 5e-2,
        inner_eps: float = 0.0,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_epochs: int = 1000,
        batch_size: int = 512,
        optim: OptimCfg,
        sched: SchedCfg = SchedCfg(),
        ema: EmaCfg = EmaCfg(),
        device: Optional[str] = None,
        antithetic: bool = True,
        div_method: Literal["hutchinson", "exact"] = "hutchinson",
        div_noise: Literal["rademacher", "gaussian"] = "rademacher",
        n_hutch_samples: int = 1,
        integration_steps: int = 3000,
        activation: str = "gelu",
        layernorm: str = "off",
        reweight: bool = False,
    ) -> None:
        """initialize triangular vfm v2 with four-slot surface.

        args:
            input_dim: dimensionality of data.
            path: optional triangularpath1d; defaults to
                  piecewise_sb_triangular_path_1d(..., eps=1e-3).
            time: optional timesampler1d; when inner_eps > 0, defaults to
                  make_piecewise_sb_sampler(...). when inner_eps == 0,
                  defaults to make_uniform(...).
            curve: optional curve; defaults to identitycurve1d().
            integrator: optional integrator; defaults to integrator_trapezoid.
            sigma, vertex, gamma_min, inner_eps: path geometry scalars.
            hidden_dim, n_hidden_layers, n_epochs, batch_size: network and
                  training shape.
            optim: required optimcfg (optimizer config with lr, etc.).
            sched: schedcfg for learning rate schedule (default schedcfg()).
            ema: emacfg for exponential moving average (default emacfg()).
            device: torch device (defaults to cuda if available, else cpu).
            antithetic: whether to use antithetic sampling in velocity loss.
            div_method, div_noise, n_hutch_samples: divergence estimation config.
            integration_steps: number of tau quadrature points (uniform grid).
            activation: mlp activation choice.
            layernorm: layernorm placement ("off", "pre", "post").
            reweight: whether to apply path-variance reweighting to losses.
        """
        # step 1: resolve defaults before validation
        if path is None:
            path = piecewise_sb_triangular_path_1d(
                sigma=sigma,
                vertex=vertex,
                gamma_min=gamma_min,
                inner_eps=inner_eps,
                eps=1e-3,
            )

        if time is None:
            if inner_eps > 0:
                time = make_piecewise_sb_sampler(
                    vertex=vertex,
                    inner_eps=inner_eps,
                    eps=path.eps,
                )
            else:
                time = make_uniform(eps=path.eps)

        # step 2: validate and store slots
        _validate_and_store_slots(
            self,
            input_dim=input_dim,
            path=path,
            time=time,
            curve=curve,
            integrator=integrator,
            expected_path_type=TriangularPath1D,
            expected_curve_dim=1,
            device=device,
        )

        # step 3: store v2-specific scalars
        self.sigma = sigma
        self.vertex = vertex
        self.gamma_min = gamma_min
        self.inner_eps = inner_eps

        # step 4: store network and training scalars
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optim = optim
        self.sched = sched
        self.ema = ema
        self.antithetic = antithetic
        self.div_method = div_method
        self.div_noise = div_noise
        self.n_hutch_samples = n_hutch_samples
        self.integration_steps = integration_steps
        self.activation = activation
        self.layernorm = layernorm
        self.reweight = reweight

        # step 5: build divergence estimator function
        self._div_fn = build_div_fn(
            method=div_method,
            noise=div_noise,
            n_samples=n_hutch_samples,
        )

        # step 6: network placeholders (frozen attribute names for checkpoint compat)
        self.net_b = None
        self.net_eta = None
        self.ema_b: Optional[object] = None
        self.ema_eta: Optional[object] = None

    def init_model(self) -> None:
        """initialize velocity field and denoiser networks on device."""
        self.net_b = build_velocity_net(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            layernorm=self.layernorm,
            device=self.device,
        )
        self.net_eta = build_denoiser_net(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            layernorm=self.layernorm,
            device=self.device,
        )

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """train velocity field and denoiser via two-phase loss.

        args:
            samples_p0: [n_samples, input_dim] batch from p_0.
            samples_p1: [n_samples, input_dim] batch from p_1.
            samples_pstar: [n_samples, input_dim] batch from p_*.
        """
        x0, x1, xstar = samples_p0, samples_p1, samples_pstar
        self.init_model()
        optimizer = self.optim.build(
            list(self.net_b.parameters()) + list(self.net_eta.parameters()),
        )
        scheduler = self.sched.build(optimizer) if self.sched else None
        ema_helper = self.ema.build(self) if self.ema else None

        loss_b = make_velo_loss(
            path=self.path,
            div_fn=self._div_fn,
            net_eta=self.net_eta,
        )
        loss_eta = make_denoiser_loss(
            path=self.path,
            net_b=self.net_b,
        )

        train_two_phase(
            estimator=self,
            x0=x0,
            x1=x1,
            xstar=xstar,
            loss_b=loss_b,
            loss_eta=loss_eta,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_helper=ema_helper,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=self.time,
            antithetic=self.antithetic,
            integration_steps=self.integration_steps,
            device=self.device,
        )

    def predict_ldr(self, samples: torch.Tensor) -> torch.Tensor:
        """infer log density ratio via path integral over velocity field.

        args:
            samples: [n_samples, input_dim] batch of test points.

        returns:
            [n_samples] cpu tensor of log(p_0 / p_1) values.
        """
        n_points = self.integration_steps

        return predict_ldr_via_curve(
            time_score_fn=lambda tau_samples: vfm_time_score_1d(
                net_b=self.net_b,
                net_eta=self.net_eta,
                path=self.path,
                x=samples,
                tau=tau_samples,
                div_fn=self._div_fn,
            ),
            path=self.path,
            curve=self.curve,
            integrator=self.integrator,
            n_points=n_points,
            samples=samples,
        )
