"""TriangularVFM: VFM DRE on a barycentric path p0 -> p* -> p1."""
from typing import Optional, Literal
import warnings

import torch

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._cfgs import (
    OptimCfg, SchedCfg, EmaCfg, TimeCfg,
    make_optim, make_sched, make_ema, make_time_sampler,
)
from src.density_ratio_estimation._trainer import train_two_phase
from src.density_ratio_estimation._losses import velo_loss, denoiser_loss
from src.models.common.mlp import MLP
from src.models.flow.div_estimators import exact_div, hutch_div
from src.waypoints.path_1d import VfmPath1D
from src.waypoints.triangular_continuous import BarycentricVfm1D


class TriangularVFM(DensityRatioEstimator):
    """VFM with barycentric path p0 -> p* -> p1.

    trains b and eta sequentially via `train_two_phase`; integrates the time-score
    over tau in [eps, 1-eps] at inference.
    """

    def __init__(
        self,
        input_dim: int,
        path: Optional[VfmPath1D] = None,
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
        # TriVFM-specific (explicit)
        antithetic: bool = True,
        div_method: Literal['hutchinson', 'exact'] = 'hutchinson',
        div_noise: Literal['rademacher', 'gaussian'] = 'rademacher',
        n_hutch_samples: int = 1,
        integration_steps: int = 3000,
        integration_type: Literal['1', '2', '3'] = '2',
        activation: str = "gelu",
        layernorm: str = "off",
        reweight: bool = False,
    ) -> None:
        """two-phase triangular vfm with cfg-based optimization surface.

        args:
            input_dim: dimensionality of data.
            path: optional vfmpath1d; defaults to barycentricvfm1d(k=20.0, vertex=0.5, eps=time.eps).
                  must be positional-or-keyword before * to reflect estimator identity.
            hidden_dim, n_hidden_layers, n_epochs, batch_size: network and training shape.
            optim: required optimcfg (optimizer config with lr, grad_clip_norm, etc.).
            sched: schedcfg for learning rate schedule (default schedcfg()).
            ema: emacfg for exponential moving average (default emacfg()).
            time: timecfg for time sampling and eps (default timecfg()).
            device: torch device (defaults to cuda if available, else cpu).
            antithetic: whether to use antithetic sampling in velo_loss.
            div_method, div_noise, n_hutch_samples: divergence estimation config.
            integration_steps, integration_type: numerical integration at inference.
            activation, layernorm: mlp architecture choices.

        raises:
            valueerror: if time.eps < 1e-3 (boundary regularity requirement).
            valueerror: if path provides sample_tau and time.dist != 'uniform'.
            valueerror: if div_method, div_noise, activation, layernorm invalid.
        """
        # validate time.eps >= 1e-3
        if time.eps < 1e-3:
            raise ValueError(
                f"timecfg.eps must be >= 1e-3 for boundary regularity of b*eta/gamma; "
                f"got eps={time.eps}"
            )

        super().__init__(input_dim)

        # store cfg objects and scalar hyperparams
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optim = optim
        self.sched = sched
        self.ema = ema
        self.time = time
        self.integration_steps = integration_steps
        self.integration_type = integration_type
        self.antithetic = antithetic

        # validate div_method, div_noise, n_hutch_samples
        if div_method not in ('hutchinson', 'exact'):
            raise ValueError(f"div_method must be 'hutchinson' or 'exact'; got {div_method!r}")
        if div_noise not in ('rademacher', 'gaussian'):
            raise ValueError(f"div_noise must be 'rademacher' or 'gaussian'; got {div_noise!r}")
        if n_hutch_samples < 1:
            raise ValueError(f"n_hutch_samples must be >= 1; got {n_hutch_samples}")
        self.div_method = div_method
        self.div_noise = div_noise
        self.n_hutch_samples = n_hutch_samples

        # validate activation, layernorm
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )
        self.activation = activation
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off', 'pre', 'post'}}; got {layernorm!r}")
        self.layernorm = layernorm
        self.reweight = reweight

        # resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # resolve path (default to BarycentricVfm1D if not provided)
        if path is None:
            self.path = BarycentricVfm1D(k=20.0, vertex=0.5, eps=time.eps)
        else:
            self.path = path

        # path/timecfg conflict check:
        # if path provides sample_tau, enforce time.dist == "uniform"
        has_path_sampler = callable(getattr(self.path, "sample_tau", None))
        if has_path_sampler and self.time.dist != "uniform":
            raise ValueError(
                "timecfg.dist must be 'uniform' when path provides sample_tau "
                f"(got dist={self.time.dist!r}; path is {type(self.path).__name__})"
            )

        # initialize network placeholders
        self.net_b = None
        self.net_eta = None
        self.ema_b: Optional[object] = None  # type will be emaWrapper from make_ema
        self.ema_eta: Optional[object] = None

    def init_model(self) -> None:
        """instantiate net_b and net_eta as mlps; drop inline ema construction."""
        self.net_b = MLP(
            self.input_dim,
            self.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            layernorm=self.layernorm
        ).to(self.device)
        self.net_eta = MLP(
            self.input_dim,
            self.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            layernorm=self.layernorm
        ).to(self.device)

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """init b/eta nets and build optimizers/schedulers/emas from cfgs, then train.

        uses shared cfg objects (optim, sched, ema) to create separate optimizer,
        scheduler, and ema for each phase (b and eta).
        """
        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        if n_star < 1:
            raise ValueError(f"samples_pstar must have at least 1 row; got n_star={n_star}")

        if n_star < self.batch_size // 4:
            warnings.warn(
                f"n_star={n_star} is small relative to batch_size={self.batch_size}; "
                f"pstar bootstrap will sample with high replication"
            )

        # move samples to device and cast to float
        samples_p0 = samples_p0.float().to(self.device)  # [n0, D]
        samples_p1 = samples_p1.float().to(self.device)  # [n1, D]
        samples_pstar = samples_pstar.float().to(self.device)  # [n_star, D]

        # initialize model
        self.init_model()

        # create optimizers from shared cfg
        optim_b = make_optim(self.net_b.parameters(), self.optim)
        optim_eta = make_optim(self.net_eta.parameters(), self.optim)

        # create schedulers from shared cfg
        sched_b = make_sched(optim_b, self.n_epochs, self.optim.lr, self.sched)
        sched_eta = make_sched(optim_eta, self.n_epochs, self.optim.lr, self.sched)

        # create emas from shared cfg
        ema_b = make_ema(self.net_b, self.ema)
        ema_eta = make_ema(self.net_eta, self.ema)
        self.ema_b = ema_b
        self.ema_eta = ema_eta

        # time sampler: if path has sample_tau, defer to it; else use timecfg-based sampler.
        # path-conflict guard in __init__ ensures timecfg.dist == "uniform" when sample_tau exists.
        if callable(getattr(self.path, "sample_tau", None)):
            def time_sampler(B, eps, device):
                tau = self.path.sample_tau(B, eps, device)
                return tau, torch.ones(B, 1, device=device)
        else:
            time_sampler = make_time_sampler(self.time)

        train_two_phase(
            model_b=self.net_b,
            model_eta=self.net_eta,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_b=velo_loss,
            loss_eta=denoiser_loss,
            optim_b=optim_b,
            optim_eta=optim_eta,
            n_steps_b=self.n_epochs,
            n_steps_eta=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler_b=sched_b,
            scheduler_eta=sched_eta,
            ema_b=ema_b,
            ema_eta=ema_eta,
            grad_clip_norm_b=self.optim.grad_clip_norm,
            grad_clip_norm_eta=self.optim.grad_clip_norm,
            eps=self.time.eps,
            loss_kwargs_b={"path": self.path, "antithetic": self.antithetic, "reweight": self.reweight},
            loss_kwargs_eta={"path": self.path, "reweight": self.reweight},
        )

        self.net_b.eval()
        self.net_eta.eval()



    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """integrate the time-score over tau in [eps, 1-eps] (chunked vmap); return -integral on cpu."""
        if self.net_b is None or self.net_eta is None:
            raise RuntimeError("TriangularVFM model is not trained. Call fit() before predict_ldr().")

        self.net_b.eval()
        self.net_eta.eval()
        samples = xs.float().to(self.device)  # [n_samples, D]
        n_samples = samples.shape[0]

        # time grid (ensure odd for Simpson's rule)
        n_points = self.integration_steps
        if n_points % 2 == 0:
            n_points += 1
        t_vals = torch.linspace(self.time.eps, 1.0 - self.time.eps, steps=n_points, device=self.device)

        # if ema is active, swap in shadow weights
        if self.ema_b is not None:
            self.ema_b.apply_to(self.net_b)
        if self.ema_eta is not None:
            self.ema_eta.apply_to(self.net_eta)

        try:
            chunk_size = max(1, 100000 // n_samples)
            time_score_fn = torch.vmap(
                self._compute_time_score_single,
                in_dims=(0, None),
                out_dims=0,
                randomness="different",
            )
            chunks = []
            for i in range(0, n_points, chunk_size):
                chunks.append(time_score_fn(t_vals[i:i + chunk_size], samples).detach())
            time_scores = torch.cat(chunks, dim=0)

            if self.integration_type == "2":
                return -torch.trapz(time_scores, t_vals, dim=0).cpu()
            if self.integration_type == "3":
                t_np = t_vals.cpu().numpy()
                h = (t_np[-1] - t_np[0]) / (n_points - 1)
                integrand = time_scores.cpu().numpy()
                integral = integrand[0] + integrand[-1]
                for i in range(1, n_points - 1):
                    integral += (2 if i % 2 == 0 else 4) * integrand[i]
                integral *= h / 3
                return -torch.from_numpy(integral)
            return -time_scores.mean(dim=0).cpu()
        finally:
            if self.ema_b is not None:
                self.ema_b.restore(self.net_b)
            if self.ema_eta is not None:
                self.ema_eta.restore(self.net_eta)

    def _compute_time_score_single(self, t_scalar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """time-score at a single t: -div(b) + <b, eta> / gamma; vmap-ready."""
        n = x.shape[0]
        t_batch = t_scalar.expand(n, 1)

        gamma_t = self.path.gamma(t_scalar)
        if gamma_t.dim() > 0:
            gamma_t = gamma_t.squeeze()

        b_pred = self.net_b(t_batch, x)
        eta_pred = self.net_eta(t_batch, x)

        def b_single(x_single):
            return self.net_b(t_scalar.view(1, 1), x_single.unsqueeze(0)).squeeze(0)

        if self.div_method == "exact":
            div_b = exact_div(b_single, x)
        else:
            div_b = hutch_div(b_single, x, noise=self.div_noise)
            for _ in range(self.n_hutch_samples - 1):
                div_b = div_b + hutch_div(b_single, x, noise=self.div_noise)
            div_b = div_b / self.n_hutch_samples
        b_dot_eta = (b_pred * eta_pred).sum(dim=-1)
        return -div_b + b_dot_eta / gamma_t
