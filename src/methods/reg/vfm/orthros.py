"""VFMOrthros: single-network orthros-variant velocity flow matching.

single-phase training on OrthrosNet (unified 2-head architecture outputting
posterior means x0_hat and x1_hat). integrates the orthros time-score via
predict_ldr_via_curve with IdentityCurve1D to predict log(p0/p1).

uses DirectPath1D (no xstar), TimeSampler1D, IdentityCurve1D, and Integrator
(Pillar E: four-slot surface). inherits legacy k parameter from VFM for path
construction; n_t is not supported (not preserved).
"""
import warnings
from typing import Optional, Literal

import torch

from ..common._cfgs import (
    OptimCfg, SchedCfg, EmaCfg,
    make_optim, make_sched, make_ema,
)
from ..common._trainer import train_loop
from ..common._weighting import resolve_outer_lambda
from ..common._estimator_helpers import _validate_and_store_slots
from ..common._paradigm_funcs import (
    vfm_velocity_target_direct_1d,
    vfm_orthros_time_score_1d,
)
from ..common._time_samplers import make_uniform, TimeSampler1D
from ..common._curves import IdentityCurve1D, Curve
from ..common._integrators import integrator_trapezoid, Integrator
from ..common._predict_ldr import predict_ldr_via_curve
from src.waypoints.dataclass_paths import DirectPath1D
from src.waypoints.path_builders import direct_vfm
from ...common.base import DRE
from src.models.flow.orthros_net import OrthrosNet
from src.models.flow.div_estimators import build_div_fn


class VFMOrthros(DRE):
    """VFMOrthros: single-network orthros-variant velocity flow matching.

    single-phase training on OrthrosNet (unified 2-head architecture outputting
    posterior means x0_hat and x1_hat). integrates the orthros time-score via
    predict_ldr_via_curve with IdentityCurve1D to predict log(p0/p1).

    uses DirectPath1D (no xstar), TimeSampler1D, IdentityCurve1D, and Integrator
    (Pillar E: four-slot surface). inherits legacy k parameter from VFM for path
    construction; n_t is not supported (not preserved).

    note: gamma_min defaults to 0.1 (not 0.0 as in stock VFM). the denoiser is
    reconstructed from the interpolant constraint as (x - a*x0 - b*x1)/gamma, so
    head error gets a 1/gamma^2 amplification in the time-score; a positive
    noise floor keeps inference numerically stable. tune it via the search space.
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
        inner_eps: float = 0.0,
        gamma_min: float = 0.1,
        test_inner_eps: float = 0.0,
        test_gamma_min: float = 0.1,
        test_path: Optional["DirectPath1D"] = None,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_shared_layers: int = 2,
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
    ) -> None:
        super().__init__(input_dim)

        # resolve all four slots before validation
        if path is None:
            path = direct_vfm(k=k, inner_eps=inner_eps, gamma_min=gamma_min, eps=1e-3)

        if time is None:
            time = make_uniform(eps=path.eps)

        if test_path is None:
            test_path = direct_vfm(k=k, inner_eps=test_inner_eps, gamma_min=test_gamma_min, eps=1e-3)

        if curve is None:
            curve = IdentityCurve1D()

        if integrator is None:
            integrator = integrator_trapezoid

        # F2/F3 sampler/path inner_eps consistency
        samp_ie = getattr(time, "inner_eps", 0.0) if time is not None else 0.0
        if (samp_ie > 0) != (inner_eps > 0):
            warnings.warn(
                f"asymmetric inner_eps: sampler={samp_ie}, path={inner_eps}. "
                "Probably unintentional.", UserWarning, stacklevel=2,
            )
        elif samp_ie > 0 and inner_eps > 0:
            assert abs(samp_ie - inner_eps) < 1e-9, \
                f"sampler/path inner_eps mismatch: {samp_ie} vs {inner_eps}"

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

        # store test path and clamping scalars
        self.test_path = test_path
        self.inner_eps = inner_eps
        self.gamma_min = gamma_min
        self.test_inner_eps = test_inner_eps
        self.test_gamma_min = test_gamma_min

        # store network and training hyperparameters
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_shared_layers = n_shared_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optim = optim if optim is not None else OptimCfg(lr=1e-3)
        self.sched = sched if sched is not None else SchedCfg()
        self.ema = ema if ema is not None else EmaCfg()
        self.k = k

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

        # network placeholders (single network + ema runtime object)
        self.net = None
        self.ema_net = None

    def init_model(self) -> None:
        """instantiate self.net (OrthrosNet) on device."""
        self.net = OrthrosNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.n_hidden_layers,
            n_shared_layers=self.n_shared_layers,
            activation=self.activation,
            layernorm=self.layernorm,
        ).to(self.device)

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """train orthros net via single-phase train_loop.

        posterior-mean MSE loss. loss closure binds antithetic, reweight, path
        geometry, and time sampler so the inner loop sees one specialized body.
        """
        self.init_model()
        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)

        optim = make_optim(self.net.parameters(), self.optim)
        sched = make_sched(optim, self.n_epochs, self.optim.lr, self.sched)
        ema_net = make_ema(self.net, self.ema)
        self.ema_net = ema_net

        path = self.path
        reweight = self.reweight
        antithetic = self.antithetic

        if antithetic:
            def loss_orthros_antithetic(model, batch, tau, iw):
                x0, x1 = batch["x0"], batch["x1"]
                z = torch.randn_like(x0)

                # positive noise
                x_t_p, _ = vfm_velocity_target_direct_1d(path, x0, x1, tau, z)
                x0_hat_p, x1_hat_p = model(x_t_p, tau)
                mse_p = ((x0_hat_p - x0)**2 + (x1_hat_p - x1)**2).sum(-1)
                l_p = mse_p

                # negative noise
                x_t_m, _ = vfm_velocity_target_direct_1d(path, x0, x1, tau, -z)
                x0_hat_m, x1_hat_m = model(x_t_m, tau)
                mse_m = ((x0_hat_m - x0)**2 + (x1_hat_m - x1)**2).sum(-1)
                l_m = mse_m

                outer = resolve_outer_lambda(reweight, tau)
                return (0.5 * (l_p + l_m) * outer * iw.squeeze(-1)).mean()
            loss_orthros = loss_orthros_antithetic
        else:
            def loss_orthros_naive(model, batch, tau, iw):
                x0, x1 = batch["x0"], batch["x1"]
                z = torch.randn_like(x0)
                x_t, _ = vfm_velocity_target_direct_1d(path, x0, x1, tau, z)
                x0_hat, x1_hat = model(x_t, tau)
                mse = ((x0_hat - x0)**2 + (x1_hat - x1)**2).sum(-1)
                outer = resolve_outer_lambda(reweight, tau)
                return (mse * outer * iw.squeeze(-1)).mean()
            loss_orthros = loss_orthros_naive

        loss_orthros.required_keys = frozenset({"x0", "x1"})
        loss_orthros.requires_tau_grad = False

        train_loop(
            model=self.net,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_fn=loss_orthros,
            optim=optim,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=self.time,
            scheduler=sched,
            ema=ema_net,
            grad_clip_norm=self.optim.grad_clip_norm,
            eps=self.path.eps,
        )

        self.net.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """integrate orthros time-score over tau in [eps, 1-eps].

        applies EMA if available, uses unified curve + integrator surface.
        """
        if self.net is None:
            raise RuntimeError(
                "VFMOrthros model is not trained. Call fit() before predict_ldr()."
            )

        self.net.eval()
        samples = xs.float().to(self.device)

        def time_score_fn(path, ts, xs):
            """loop per curve point; vfm_orthros_time_score_1d expects tau:[B,1] matching xs:[B,D]."""
            n = xs.shape[0]
            out = []
            for i in range(ts.shape[0]):
                tau_i = ts[i:i+1].expand(n, 1)
                out.append(
                    vfm_orthros_time_score_1d(self.net, path, xs, tau_i, self._div_fn)
                )
            return torch.stack(out, dim=0)

        if self.ema_net is not None:
            self.ema_net.apply_to(self.net)

        try:
            ldr = predict_ldr_via_curve(
                time_score_fn=time_score_fn,
                path=self.test_path,
                curve=self.curve,
                integrator=self.integrator,
                n_points=self.integration_steps,
                samples=samples,
            )
            return ldr
        finally:
            if self.ema_net is not None:
                self.ema_net.restore(self.net)


__all__ = ["VFMOrthros"]
