"""TriangularVFMV1: velocity field matching DRE on a barycentric path (k-parameterized gamma).

no inheritance from base; uses free functions + explicit path/time/curve/integrator slots.
"""
from typing import Optional, Literal
import warnings

import torch

from ....common.base import ELDR
from ...common._cfgs import (
    OptimCfg, SchedCfg, EmaCfg,
    make_optim, make_sched, make_ema,
)
from ...common._trainer import train_two_phase
from ...common._losses import make_velo_loss, make_denoiser_loss
from src.waypoints.dataclass_paths import TriangularPath1D
from src.waypoints.path_builders import bary_vfm
from src.methods.reg.common._time_samplers import TimeSampler1D, make_uniform
from src.methods.reg.common._curves import Curve, IdentityCurve1D
from src.methods.reg.common._integrators import Integrator, integrator_trapezoid
from src.methods.reg.common._paradigm_funcs import vfm_time_score_1d
from src.methods.reg.common._predict_ldr import predict_ldr_via_curve
from src.methods.reg.common._precond import endpoint_moments, make_coeffs, make_lambda, wrap
from src.methods.reg.common._estimator_helpers import _validate_and_store_slots
from src.models.common.mlp import MLP
from src.models.flow.div_estimators import build_div_fn


class TriangularVFMV1(ELDR):
    """VFM with a barycentric triangular path (k-parameterized gamma)."""

    def __init__(
        self,
        input_dim: int,
        *,
        path: Optional[TriangularPath1D] = None,
        test_path: Optional[TriangularPath1D] = None,
        time: Optional[TimeSampler1D] = None,
        curve: Curve = IdentityCurve1D(),
        integrator: Integrator = integrator_trapezoid,
        k: float = 20.0,
        vertex: float = 0.5,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_epochs: int = 1000,
        batch_size: int = 512,
        optim: OptimCfg = None,
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
        inner_eps: float = 0.0,
        gamma_min: float = 0.0,
        test_inner_eps: float = 0.0,
        test_gamma_min: float = 0.0,
        precond: bool = False,
    ) -> None:
        """barycentric VFM on triangular path; defaults use k and vertex if path is None."""
        super().__init__(input_dim)

        # resolve defaults before validation
        if path is None:
            path = bary_vfm(k=k, vertex=vertex, inner_eps=inner_eps, gamma_min=gamma_min, eps=1e-3)
        if time is None:
            time = make_uniform(eps=path.eps)

        # build test path; caller may pass an explicit test_path (independent
        # test-path config), else fall back to the test_* clamp scalars.
        if test_path is None:
            test_path = bary_vfm(
                k=k, vertex=vertex,
                inner_eps=test_inner_eps, gamma_min=test_gamma_min, eps=1e-3,
            )

        # f2/f3 sampler/path inner_eps consistency check
        samp_ie = getattr(time, "inner_eps", 0.0) if time is not None else 0.0
        if (samp_ie > 0) != (inner_eps > 0):
            warnings.warn(
                f"asymmetric inner_eps: sampler={samp_ie}, path={inner_eps}. "
                "Probably unintentional.", UserWarning, stacklevel=2,
            )
        elif samp_ie > 0 and inner_eps > 0:
            assert abs(samp_ie - inner_eps) < 1e-9, \
                f"sampler/path inner_eps mismatch: {samp_ie} vs {inner_eps}"

        # validate and store path/time/curve/integrator slots
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
        # store test_path and clamping scalars
        self.test_path = test_path
        self.inner_eps = inner_eps
        self.gamma_min = gamma_min
        self.test_inner_eps = test_inner_eps
        self.test_gamma_min = test_gamma_min

        # store network hyperparameters for hpo introspection
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.k = k
        self.vertex = vertex

        # store training configuration
        if optim is None:
            raise ValueError("optim is required")
        self.optim = optim
        self.sched = sched
        self.ema = ema

        # validate and store divergence function hyperparameters
        if div_method not in ("hutchinson", "exact"):
            raise ValueError(f"div_method must be 'hutchinson' or 'exact'; got {div_method!r}")
        if div_noise not in ("rademacher", "gaussian"):
            raise ValueError(f"div_noise must be 'rademacher' or 'gaussian'; got {div_noise!r}")
        if n_hutch_samples < 1:
            raise ValueError(f"n_hutch_samples must be >= 1; got {n_hutch_samples}")
        self.div_method = div_method
        self.div_noise = div_noise
        self.n_hutch_samples = n_hutch_samples

        # validate and store activation + layernorm
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}")
        self.activation = activation

        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off', 'pre', 'post'}}; got {layernorm!r}")
        self.layernorm = layernorm

        # store antithetic, reweight, and integration parameters
        self.antithetic = antithetic
        self.reweight = reweight
        self.integration_steps = integration_steps

        # build divergence function
        self._div_fn = build_div_fn(self.div_method, noise=self.div_noise, n_samples=self.n_hutch_samples)

        # store preconditioning flag and placeholder for endpoint moments
        self.precond = precond
        self._moments = None

        # initialize network attributes (frozen names for checkpoint compat)
        self.net_b = None
        self.net_eta = None
        self.ema_b: Optional[object] = None
        self.ema_eta: Optional[object] = None

    def init_model(self) -> None:
        """build net_b and net_eta as MLPs; EMA is constructed in fit."""
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

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """two-phase training; loss closures built once via factories."""
        n_star = samples_pstar.shape[0]
        if n_star < 1:
            raise ValueError(f"samples_pstar must have at least 1 row; got n_star={n_star}")
        if n_star < self.batch_size // 4:
            warnings.warn(
                f"n_star={n_star} is small relative to batch_size={self.batch_size}; "
                f"pstar bootstrap will sample with high replication"
            )

        # move samples to device as float32
        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)
        samples_pstar = samples_pstar.float().to(self.device)

        # estimate endpoint moments if preconditioning is enabled
        if self.precond:
            self._moments = endpoint_moments({
                "x0": samples_p0,
                "x1": samples_p1,
                "xstar": samples_pstar,
            })

        # initialize networks and build optimizer/scheduler/EMA
        self.init_model()

        optim_b = make_optim(self.net_b.parameters(), self.optim)
        optim_eta = make_optim(self.net_eta.parameters(), self.optim)
        sched_b = make_sched(optim_b, self.n_epochs, self.optim.lr, self.sched)
        sched_eta = make_sched(optim_eta, self.n_epochs, self.optim.lr, self.sched)
        ema_b = make_ema(self.net_b, self.ema)
        ema_eta = make_ema(self.net_eta, self.ema)
        self.ema_b = ema_b
        self.ema_eta = ema_eta

        # wire loss factories with paradigm defaults and optional outer-weight
        if self.precond:
            if self.reweight:
                warnings.warn(
                    "precond=True will ignore reweight setting; "
                    "EDM lambda replaces path-variance weighting.",
                    UserWarning, stacklevel=2,
                )
            coeff_b = make_coeffs(self.path, self._moments, "velocity")
            coeff_eta = make_coeffs(self.path, self._moments, "noise")
            loss_b = make_velo_loss(
                path=self.path,
                antithetic=self.antithetic,
                reweight=self.reweight,
                outer_weight=make_lambda(coeff_b),
            )
            loss_eta = make_denoiser_loss(
                path=self.path,
                reweight=self.reweight,
                outer_weight=make_lambda(coeff_eta),
            )
        else:
            coeff_b = None
            coeff_eta = None
            loss_b = make_velo_loss(path=self.path, antithetic=self.antithetic, reweight=self.reweight)
            loss_eta = make_denoiser_loss(path=self.path, reweight=self.reweight)

        # wrap networks once at fit time if preconditioning; pass wrapped model
        # and raw module to train_two_phase (trainer applies EMA to module only).
        if self.precond:
            model_b_to_train = wrap(self.net_b, coeff_b)
            model_eta_to_train = wrap(self.net_eta, coeff_eta)
        else:
            model_b_to_train = self.net_b
            model_eta_to_train = self.net_eta

        # call shared training orchestrator. wrapped callables are model_b/model_eta;
        # raw nets are model_module_b/model_module_eta (params, device, EMA, train/eval).
        # train_two_phase gains model_module_b/eta per spec_trainer.md.
        train_two_phase(
            model_b=model_b_to_train,
            model_eta=model_eta_to_train,
            model_module_b=self.net_b,
            model_module_eta=self.net_eta,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
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

        # finalize networks to eval mode
        self.net_b.eval()
        self.net_eta.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """integrate the time-score over tau in [eps, 1-eps]; return -integral on cpu."""
        if self.net_b is None or self.net_eta is None:
            raise RuntimeError("triangular vfm v1 not trained; call fit() before predict_ldr().")

        # set networks to eval, move input to device
        self.net_b.eval()
        self.net_eta.eval()
        samples = xs.float().to(self.device)

        # rebuild preconditioned wrappers from stored moments and train path.
        # EMA was already applied to net_b and net_eta above; wrapper is rebuilt
        # after EMA so it sees the averaged weights.
        if self.precond:
            coeff_b = make_coeffs(self.path, self._moments, "velocity")
            coeff_eta = make_coeffs(self.path, self._moments, "noise")
            net_b_use = wrap(self.net_b, coeff_b)
            net_eta_use = wrap(self.net_eta, coeff_eta)
        else:
            net_b_use = self.net_b
            net_eta_use = self.net_eta

        # bind time-score function as closure over networks and divergence fn
        def time_score_fn(path, ts, x):
            """evaluate time-score at batch of time points.

            ts: [n_points, 1] (IdentityCurve1D maps tau -> [tau])
            x:  [n_samples, input_dim]

            returns: [n_points, n_samples] (one score per time-sample pair)
            """
            # ts: [chunk_len, 1]; loop per time-point since vfm_time_score_1d
            # expects tau:[B,1] with B matching x.
            n = x.shape[0]
            out = []
            for i in range(ts.shape[0]):
                tau_i = ts[i:i+1].expand(n, 1)
                out.append(vfm_time_score_1d(net_b_use, net_eta_use, path, x, tau_i, self._div_fn))
            return torch.stack(out, dim=0)

        # apply EMA, call shared inference, restore networks
        try:
            if self.ema_b is not None:
                self.ema_b.apply_to(self.net_b)
            if self.ema_eta is not None:
                self.ema_eta.apply_to(self.net_eta)

            result = predict_ldr_via_curve(
                time_score_fn=time_score_fn,
                path=self.test_path,
                curve=self.curve,
                integrator=self.integrator,
                n_points=self.integration_steps,
                samples=samples,
            )
            return result
        finally:
            if self.ema_b is not None:
                self.ema_b.restore(self.net_b)
            if self.ema_eta is not None:
                self.ema_eta.restore(self.net_eta)
