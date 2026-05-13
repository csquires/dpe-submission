"""VFM family: velocity flow matching DRE.

The two-source (DRE) variant VFM lives in this module; triangular variants
(V1 barycentric, V2 piecewise-SB, V3 2D) live under `.tri`.
"""
import warnings
from typing import Optional, Literal, Callable

import torch

from ..common._cfgs import (
    OptimCfg, SchedCfg, EmaCfg,
    make_optim, make_sched, make_ema,
)
from ..common._trainer import train_two_phase
from ..common._weighting import resolve_outer_lambda
from ..common._estimator_helpers import _validate_and_store_slots
from ..common._paradigm_funcs import vfm_velocity_target_direct_1d, vfm_time_score_1d
from ..common._time_samplers import make_uniform, TimeSampler1D
from ..common._curves import IdentityCurve1D, Curve
from ..common._integrators import integrator_trapezoid, Integrator
from ..common._predict_ldr import predict_ldr_via_curve
from src.waypoints.dataclass_paths import DirectPath1D
from src.waypoints.path_builders import direct_path_1d
from ...common.base import DRE
from src.models.common.mlp import MLP
from src.models.flow.div_estimators import build_div_fn


class VFM(DRE):
    """VFM stock: velocity flow matching on direct (two-source) path, two-phase training.

    trains b (velocity) and eta (denoiser) sequentially; integrates the time-score
    via predict_ldr_via_curve with IdentityCurve1D to predict log(p0/p1).

    uses DirectPath1D (no xstar), TimeSampler1D, IdentityCurve1D, and Integrator
    (Pillar E: four-slot surface). legacy scalar k and n_t parameters preserved
    for backward compatibility; k parameterizes the path if path=None.
    """
    def __init__(
        self,
        input_dim: int,
        *,
        path: Optional[DirectPath1D] = None,
        time: Optional[TimeSampler1D] = None,
        curve: Curve = None,
        integrator: Integrator = None,
        k: float = 0.5,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_epochs: int = 1000,
        batch_size: int = 512,
        optim: Optional[OptimCfg] = None,
        sched: SchedCfg = None,
        ema: EmaCfg = None,
        device: Optional[str] = None,
        antithetic: bool = False,
        div_method: Literal['hutchinson', 'exact'] = 'hutchinson',
        div_noise: Literal['rademacher', 'gaussian'] = 'rademacher',
        n_hutch_samples: int = 1,
        integration_steps: int = 10000,
        activation: str = "silu",
        layernorm: str = "off",
        reweight: bool = False,
        n_t: Optional[int] = None,
    ) -> None:
        super().__init__(input_dim)

        # resolve all four slots before validation
        if path is None:
            path = direct_path_1d(k=k, eps=1e-3)

        if time is None:
            time = make_uniform(eps=path.eps)

        if curve is None:
            curve = IdentityCurve1D()

        if integrator is None:
            integrator = integrator_trapezoid

        # validate and store four-slot surface
        _validate_and_store_slots(
            self,
            input_dim=input_dim,
            path=path,
            time=time,
            curve=curve,
            integrator=integrator,
            expected_path_type=DirectPath1D,
            expected_curve_dim=1,
            device=device,
        )

        # store network and training hyperparameters
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optim = optim if optim is not None else OptimCfg(lr=1e-3)
        self.sched = sched if sched is not None else SchedCfg()
        self.ema = ema if ema is not None else EmaCfg()
        self.k = k

        # deprecation warning for n_t
        if n_t is not None:
            warnings.warn(
                "`n_t` is deprecated and unused; it has no effect on VFM training "
                "or inference and will be removed in a future revision.",
                DeprecationWarning,
                stacklevel=2,
            )

        # divergence-related parameters
        if div_method not in ('hutchinson', 'exact'):
            raise ValueError(f"div_method must be 'hutchinson' or 'exact'; got {div_method!r}")
        if div_noise not in ('rademacher', 'gaussian'):
            raise ValueError(f"div_noise must be 'rademacher' or 'gaussian'; got {div_noise!r}")
        if n_hutch_samples < 1:
            raise ValueError(f"n_hutch_samples must be >= 1; got {n_hutch_samples}")

        self.div_method = div_method
        self.div_noise = div_noise
        self.n_hutch_samples = n_hutch_samples
        self._div_fn = build_div_fn(div_method, noise=div_noise, n_samples=n_hutch_samples)

        # miscellaneous parameters
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off', 'pre', 'post'}}; got {layernorm!r}")

        self.activation = activation
        self.layernorm = layernorm
        self.antithetic = antithetic
        self.integration_steps = integration_steps
        self.reweight = reweight

        # network placeholders (frozen attribute names for checkpoint compat)
        self.net_b = None
        self.net_eta = None
        self.ema_b = None
        self.ema_eta = None

    def init_model(self) -> None:
        """instantiate net_b (velocity) and net_eta (denoiser) MLPs on device."""
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


    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """train b then eta sequentially via train_two_phase.

        the loss_b closure binds antithetic, reweight, path geometry, and
        the time sampler so the inner loop sees only one specialized body.
        """
        self.init_model()
        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)

        optim_b = make_optim(self.net_b.parameters(), self.optim)
        optim_eta = make_optim(self.net_eta.parameters(), self.optim)
        sched_b = make_sched(optim_b, self.n_epochs, self.optim.lr, self.sched)
        sched_eta = make_sched(optim_eta, self.n_epochs, self.optim.lr, self.sched)
        ema_b = make_ema(self.net_b, self.ema)
        ema_eta = make_ema(self.net_eta, self.ema)
        self.ema_b, self.ema_eta = ema_b, ema_eta

        path = self.path
        reweight = self.reweight
        antithetic = self.antithetic

        if antithetic:
            def loss_b_antithetic(model, batch, tau, iw):
                x0, x1 = batch["x0"], batch["x1"]
                z = torch.randn_like(x0)
                # positive noise
                x_t_p, v_star_p = vfm_velocity_target_direct_1d(path, x0, x1, tau, z)
                b_p = model(tau.unsqueeze(-1), x_t_p)
                l_p = 0.5 * (b_p ** 2).sum(-1) - (v_star_p * b_p).sum(-1)
                # negative noise
                x_t_m, v_star_m = vfm_velocity_target_direct_1d(path, x0, x1, tau, -z)
                b_m = model(tau.unsqueeze(-1), x_t_m)
                l_m = 0.5 * (b_m ** 2).sum(-1) - (v_star_m * b_m).sum(-1)
                outer = resolve_outer_lambda(reweight, tau)
                return (0.5 * (l_p + l_m) * outer * iw.squeeze(-1)).mean()
            loss_b = loss_b_antithetic
        else:
            def loss_b_naive(model, batch, tau, iw):
                x0, x1 = batch["x0"], batch["x1"]
                z = torch.randn_like(x0)
                x_t, v_star = vfm_velocity_target_direct_1d(path, x0, x1, tau, z)
                outer = resolve_outer_lambda(reweight, tau)
                b = model(tau.unsqueeze(-1), x_t)
                return ((0.5 * (b ** 2).sum(-1) - (v_star * b).sum(-1)) * outer * iw.squeeze(-1)).mean()
            loss_b = loss_b_naive

        loss_b.required_keys = frozenset({"x0", "x1"})
        loss_b.requires_tau_grad = False

        def loss_eta(model, batch, tau, iw):
            x0, x1 = batch["x0"], batch["x1"]
            z = torch.randn_like(x0)
            x_t, _ = vfm_velocity_target_direct_1d(path, x0, x1, tau, z)
            eta = model(tau.unsqueeze(-1), x_t)
            outer = resolve_outer_lambda(reweight, tau)
            return ((0.5 * (eta ** 2).sum(-1) - (z * eta).sum(-1)) * outer * iw.squeeze(-1)).mean()

        loss_eta.required_keys = frozenset({"x0", "x1"})
        loss_eta.requires_tau_grad = False

        train_two_phase(
            model_b=self.net_b,
            model_eta=self.net_eta,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_b=loss_b,
            loss_eta=loss_eta,
            optim_b=optim_b,
            optim_eta=optim_eta,
            n_steps_b=self.n_epochs,
            n_steps_eta=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=self.time,
            scheduler_b=sched_b,
            scheduler_eta=sched_eta,
            ema_b=ema_b,
            ema_eta=ema_eta,
            grad_clip_norm_b=self.optim.grad_clip_norm,
            grad_clip_norm_eta=self.optim.grad_clip_norm,
            eps=self.path.eps,
        )

        self.net_b.eval()
        self.net_eta.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """integrate the time-score over tau in [eps, 1-eps] using predict_ldr_via_curve.

        applies EMA if available, uses unified curve + integrator surface.
        """
        if self.net_b is None or self.net_eta is None:
            raise RuntimeError("VFM model is not trained. Call fit() before predict_ldr().")

        self.net_b.eval()
        self.net_eta.eval()
        samples = xs.float().to(self.device)

        def time_score_fn(path, ts, xs):
            """closure that evaluates vfm_time_score_1d at curve points ts."""
            # ts: [n_points, 1] (curve points)
            # xs: [n_samples, input_dim]
            # returns: [n_points, n_samples] (one score per time point per sample)
            tau = ts[:, 0]  # extract tau from curve points
            return vfm_time_score_1d(
                self.net_b, self.net_eta, path, xs, tau, self._div_fn
            )

        if self.ema_b is not None:
            self.ema_b.apply_to(self.net_b)
        if self.ema_eta is not None:
            self.ema_eta.apply_to(self.net_eta)

        try:
            ldr = predict_ldr_via_curve(
                time_score_fn=time_score_fn,
                path=self.path,
                curve=self.curve,
                integrator=self.integrator,
                n_points=self.integration_steps,
                samples=samples,
            )
            return ldr
        finally:
            if self.ema_b is not None:
                self.ema_b.restore(self.net_b)
            if self.ema_eta is not None:
                self.ema_eta.restore(self.net_eta)


from .tri import TriangularVFMV1, TriangularVFMV2, TriangularVFM2D

__all__ = ["VFM", "TriangularVFMV1", "TriangularVFMV2", "TriangularVFM2D"]
