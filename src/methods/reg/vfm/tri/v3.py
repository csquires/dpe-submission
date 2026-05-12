"""TriangularVFM2D: VFM DRE with 2D-time stacked interpolant.

trains two velocity heads (b_1, b_2) and one denoiser (eta) sequentially on a
2D-time stacked interpolant; integrates the time-score along a Curve2D at inference.
"""
from typing import Optional, Literal
import warnings

import torch

from ....common.base import ELDR
from ...common._cfgs import (
    OptimCfg, SchedCfg, EmaCfg, TimeCfg,
    make_optim, make_sched, make_ema,
)
from ...common._weighting import resolve_outer_lambda
from src.models.flow.div_estimators import build_div_fn
from src.waypoints.path_2d import VfmPath2D
from src.waypoints.triangular_continuous_2d import Stacked2DVfm
from src.waypoints.curve_2d import Curve2D
from src.models.time_score_matching.velocity_network_2d import MLP2D


class TriangularVFM2D(ELDR):
    """vfm dre with 2d-time stacked interpolant path.

    trains two velocity heads (b_1, b_2) and one denoiser (eta) sequentially
    on three distributions p_0, p_1, p_*. inference integrates the time-score
    along self.curve from tau=eps to 1-eps. uses cfg-based optimization surface
    with factory-built optimizers, schedulers, and ema wrappers per network.

    contract: fit(samples_p0, samples_p1, samples_pstar) with three [n, d]
    tensors; predict_ldr(xs) returns log(p_0/p_1) as [n_samples] cpu tensor.
    """

    def __init__(
        self,
        input_dim: int,
        path: Optional[VfmPath2D] = None,
        curve: Optional[Curve2D] = None,
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
        # TriangularVFM2D-specific (explicit)
        antithetic: bool = True,
        div_method: Literal['hutchinson', 'exact'] = 'hutchinson',
        div_noise: Literal['rademacher', 'gaussian'] = 'rademacher',
        n_hutch_samples: int = 1,
        integration_steps: int = 200,
        activation: str = "gelu",
        layernorm: str = "off",
        reweight: bool = False,
    ) -> None:
        """initialize triangularvfm2d with cfg-based optimization surface.

        args:
            input_dim: dimensionality of data.
            path: optional vfmpath2d; defaults to stacked2dvfm(eps=time.eps).
                  must be positional-or-keyword before * to reflect estimator identity.
            curve: optional curve2d; defaults to curve2d(path_height=1.0).
            hidden_dim, n_hidden_layers, n_epochs, batch_size: network and training shape.
            optim: required optimcfg (optimizer config with lr, grad_clip_norm, etc.).
            sched: schedcfg for learning rate schedule (default schedcfg()).
            ema: emacfg for exponential moving average (default emacfg()).
            time: timecfg for time sampling and eps (default timecfg()).
            device: torch device (defaults to cuda if available, else cpu).
            antithetic: whether to use antithetic sampling in velocity loss.
            div_method, div_noise, n_hutch_samples: divergence estimation config.
            integration_steps: number of tau quadrature points (uniform grid).
            activation: mlp activation choice.
            reweight: whether to apply path-variance reweighting to losses.

        raises:
            valueerror: if time.eps < 1e-3 (boundary regularity requirement).
            valueerror: if div_method, div_noise, activation invalid.
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

        # validate activation
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

        # resolve path (default to Stacked2DVfm if not provided)
        if path is None:
            self.path = Stacked2DVfm(
                k=20.0, gamma_schedule="linear-stiff", t2_max=0.3, eps=time.eps
            )
        else:
            self.path = path

        # no path-conflict guard: timecfg holds an explicit timesampler instance.
        # users who want path-driven sampling pass timecfg(sampler=pathsampler(path)).

        # resolve curve default
        self.curve = curve if curve is not None else Curve2D(path_height=1.0)

        # bind divergence estimator once; predict_ldr time-score holds no branches.
        self._div_fn = build_div_fn(div_method, noise=div_noise, n_samples=n_hutch_samples)

        # coverage assertion: curve t_2 range must lie within trained t_2 range
        peak = float(self.curve.peak_t2())
        t2_max = float(self.path.t2_max)
        assert peak <= t2_max + 1e-9, (
            f"curve peak_t2 {peak} exceeds path.t2_max {t2_max}"
        )

        # network placeholders
        self.net_b1 = None
        self.net_b2 = None
        self.net_eta = None
        self.ema_b1: Optional[object] = None  # type will be emaWrapper from make_ema
        self.ema_b2: Optional[object] = None
        self.ema_eta: Optional[object] = None

    def init_model(self) -> None:
        """instantiate net_b1, net_b2, and net_eta as mlp2d; drop inline ema construction."""
        self.net_b1 = MLP2D(
            self.input_dim,
            self.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            layernorm=self.layernorm,
        ).to(self.device)
        self.net_b2 = MLP2D(
            self.input_dim,
            self.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            layernorm=self.layernorm,
        ).to(self.device)
        self.net_eta = MLP2D(
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
        """init b1/b2/eta nets and build optimizers/schedulers/emas from cfgs, then train.

        uses shared cfg objects (optim, sched, ema) to create separate optimizer,
        scheduler, and ema for each phase (b and eta).
        """
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
        optim_b = make_optim(
            list(self.net_b1.parameters()) + list(self.net_b2.parameters()),
            self.optim
        )
        optim_eta = make_optim(self.net_eta.parameters(), self.optim)

        # create schedulers from shared cfg
        sched_b = make_sched(optim_b, self.n_epochs, self.optim.lr, self.sched)
        sched_eta = make_sched(optim_eta, self.n_epochs, self.optim.lr, self.sched)

        # create emas from shared cfg
        ema_b1 = make_ema(self.net_b1, self.ema)
        ema_b2 = make_ema(self.net_b2, self.ema)
        ema_eta = make_ema(self.net_eta, self.ema)
        self.ema_b1 = ema_b1
        self.ema_b2 = ema_b2
        self.ema_eta = ema_eta

        # train phases with factory-built objects
        self._train_b_phase(samples_p0, samples_p1, samples_pstar, optim_b, sched_b, ema_b1, ema_b2)
        self._train_eta_phase(samples_p0, samples_p1, samples_pstar, optim_eta, sched_eta, ema_eta)

        # post-training cleanup
        self.net_b1.eval()
        self.net_b2.eval()
        self.net_eta.eval()

    def _train_b_phase(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
        optimizer_b: torch.optim.Optimizer,
        scheduler_b: object,
        ema_b1: object,
        ema_b2: object,
    ) -> None:
        """train b_1 and b_2 jointly with eta frozen.

        velocity matching via inline losses (half-norm minus dot). per-direction
        losses computed separately; sum is back-propagated. antithetic variance
        reduction applied if self.antithetic=true. iw and outer_lambda reweighting
        applied to per-sample losses before mean reduction.
        """
        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        self.net_b1.train()
        self.net_b2.train()
        self.net_eta.eval()

        t2_max = float(self.path.t2_max)
        eps_t = self.time.eps
        path = self.path
        net_b1 = self.net_b1
        net_b2 = self.net_b2
        reweight = self.reweight

        # bind antithetic vs non-antithetic body once; the inner loop calls compute_losses()
        # for both regimes via the same shape.
        if self.antithetic:
            def compute_losses(x0, x1, xstar):
                t1 = torch.rand(self.batch_size, 1, device=self.device) * (1 - 2*eps_t) + eps_t
                t2 = torch.rand(self.batch_size, 1, device=self.device) * (t2_max - eps_t) + eps_t
                z = torch.randn_like(x0)
                outer = resolve_outer_lambda(reweight, t1)
                mu = path.mu(x0, x1, xstar, t1, t2).detach()
                dmu_dt1 = path.dmu_dt1(x0, x1, xstar, t1, t2).detach()
                dmu_dt2 = path.dmu_dt2(x0, x1, xstar, t1, t2).detach()
                gamma_t = path.gamma(t1, t2).detach()
                dgamma_dt1 = path.dgamma_dt1(t1, t2).detach()
                dgamma_dt2 = path.dgamma_dt2(t1, t2).detach()

                x_t_plus = mu + gamma_t * z
                x_t_minus = mu - gamma_t * z
                b1_plus = net_b1(t1, t2, x_t_plus)
                b2_plus = net_b2(t1, t2, x_t_plus)
                b1_minus = net_b1(t1, t2, x_t_minus)
                b2_minus = net_b2(t1, t2, x_t_minus)
                target_1_plus = dmu_dt1 + dgamma_dt1 * z
                target_2_plus = dmu_dt2 + dgamma_dt2 * z
                target_1_minus = dmu_dt1 - dgamma_dt1 * z
                target_2_minus = dmu_dt2 - dgamma_dt2 * z
                per_b1 = (
                    0.25 * (b1_plus ** 2).sum(dim=-1)
                    - 0.5 * (target_1_plus * b1_plus).sum(dim=-1)
                    + 0.25 * (b1_minus ** 2).sum(dim=-1)
                    - 0.5 * (target_1_minus * b1_minus).sum(dim=-1)
                ) * outer
                per_b2 = (
                    0.25 * (b2_plus ** 2).sum(dim=-1)
                    - 0.5 * (target_2_plus * b2_plus).sum(dim=-1)
                    + 0.25 * (b2_minus ** 2).sum(dim=-1)
                    - 0.5 * (target_2_minus * b2_minus).sum(dim=-1)
                ) * outer
                return per_b1.mean() + per_b2.mean()
        else:
            def compute_losses(x0, x1, xstar):
                t1 = torch.rand(self.batch_size, 1, device=self.device) * (1 - 2*eps_t) + eps_t
                t2 = torch.rand(self.batch_size, 1, device=self.device) * (t2_max - eps_t) + eps_t
                z = torch.randn_like(x0)
                outer = resolve_outer_lambda(reweight, t1)
                mu = path.mu(x0, x1, xstar, t1, t2).detach()
                dmu_dt1 = path.dmu_dt1(x0, x1, xstar, t1, t2).detach()
                dmu_dt2 = path.dmu_dt2(x0, x1, xstar, t1, t2).detach()
                gamma_t = path.gamma(t1, t2).detach()
                dgamma_dt1 = path.dgamma_dt1(t1, t2).detach()
                dgamma_dt2 = path.dgamma_dt2(t1, t2).detach()

                x_t = mu + gamma_t * z
                b1_pred = net_b1(t1, t2, x_t)
                b2_pred = net_b2(t1, t2, x_t)
                target_1 = dmu_dt1 + dgamma_dt1 * z
                target_2 = dmu_dt2 + dgamma_dt2 * z
                per_b1 = (
                    0.5 * (b1_pred ** 2).sum(dim=-1) - (target_1 * b1_pred).sum(dim=-1)
                ) * outer
                per_b2 = (
                    0.5 * (b2_pred ** 2).sum(dim=-1) - (target_2 * b2_pred).sum(dim=-1)
                ) * outer
                return per_b1.mean() + per_b2.mean()

        # bind post-step callbacks once.
        grad_clip = self.optim.grad_clip_norm
        if grad_clip is not None and grad_clip > 0:
            b_params = list(net_b1.parameters()) + list(net_b2.parameters())

            def do_clip():
                torch.nn.utils.clip_grad_norm_(b_params, max_norm=grad_clip)
        else:
            def do_clip():
                return None
        do_sched = scheduler_b.step if scheduler_b is not None else (lambda: None)
        do_ema_b1 = (lambda: ema_b1.update(net_b1)) if ema_b1 is not None else (lambda: None)
        do_ema_b2 = (lambda: ema_b2.update(net_b2)) if ema_b2 is not None else (lambda: None)

        for epoch in range(self.n_epochs):
            idx0 = torch.randint(0, n0, (self.batch_size,), device=self.device)
            idx1 = torch.randint(0, n1, (self.batch_size,), device=self.device)
            idx_star = torch.randint(0, n_star, (self.batch_size,), device=self.device)
            loss = compute_losses(samples_p0[idx0], samples_p1[idx1], samples_pstar[idx_star])
            optimizer_b.zero_grad()
            loss.backward()
            do_clip()
            optimizer_b.step()
            do_sched()
            do_ema_b1()
            do_ema_b2()

    def _train_eta_phase(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
        optimizer_eta: torch.optim.Optimizer,
        scheduler_eta: object,
        ema_eta: object,
    ) -> None:
        """train eta with b_1 and b_2 frozen.

        denoising loss (half-norm minus dot with noise). reweighting applied to
        per-sample losses before mean reduction.
        """
        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        self.net_b1.eval()
        self.net_b2.eval()
        self.net_eta.train()

        t2_max = float(self.path.t2_max)
        eps_t = self.time.eps
        path = self.path
        net_eta = self.net_eta
        reweight = self.reweight

        # bind post-step callbacks once.
        grad_clip = self.optim.grad_clip_norm
        if grad_clip is not None and grad_clip > 0:
            eta_params = list(net_eta.parameters())

            def do_clip():
                torch.nn.utils.clip_grad_norm_(eta_params, max_norm=grad_clip)
        else:
            def do_clip():
                return None
        do_sched = scheduler_eta.step if scheduler_eta is not None else (lambda: None)
        do_ema = (lambda: ema_eta.update(net_eta)) if ema_eta is not None else (lambda: None)

        for epoch in range(self.n_epochs):
            idx0 = torch.randint(0, n0, (self.batch_size,), device=self.device)
            idx1 = torch.randint(0, n1, (self.batch_size,), device=self.device)
            idx_star = torch.randint(0, n_star, (self.batch_size,), device=self.device)
            x0 = samples_p0[idx0]
            x1 = samples_p1[idx1]
            xstar = samples_pstar[idx_star]

            t1 = torch.rand(self.batch_size, 1, device=self.device) * (1 - 2*eps_t) + eps_t
            t2 = torch.rand(self.batch_size, 1, device=self.device) * (t2_max - eps_t) + eps_t

            z = torch.randn_like(x0)
            x_t = path.sample(x0, x1, xstar, t1, t2, z).detach()
            outer = resolve_outer_lambda(reweight, t1)
            eta_pred = net_eta(t1, t2, x_t)
            per_sample = (
                0.5 * (eta_pred ** 2).sum(dim=-1) - (z * eta_pred).sum(dim=-1)
            ) * outer
            loss_eta = per_sample.mean()

            optimizer_eta.zero_grad()
            loss_eta.backward()
            do_clip()
            optimizer_eta.step()
            do_sched()
            do_ema()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """estimate log p_0(x) / p_1(x) via time-score line integral on self.curve.

        computes one divergence per velocity head (two total) per integration step.
        the two jacrev calls have different signatures; do NOT collapse them.

        args:
            xs: [n, d] test points (cpu or device); moved to self.device.

        returns:
            [n] log density ratios, cpu float32.

        raises:
            runtimeerror: if any of self.net_b1, self.net_b2, self.net_eta is none.
        """
        if self.net_b1 is None or self.net_b2 is None or self.net_eta is None:
            raise RuntimeError(
                "TriangularVFM2D model is not trained. Call fit() before predict_ldr()."
            )

        self.net_b1.eval()
        self.net_b2.eval()
        self.net_eta.eval()
        samples = xs.float().to(self.device)  # [n_samples, D]
        n_samples = samples.shape[0]

        # create uniform tau grid and pack curve outputs
        n_points = self.integration_steps
        tau_vals = torch.linspace(
            self.time.eps, 1.0 - self.time.eps, steps=n_points, device=self.device
        )  # [n_points]

        curve = self.curve
        tau_list = tau_vals.tolist()  # n_points python floats
        t_data = torch.tensor(
            [
                [curve.t1(tau), curve.t2(tau), curve.dt1(tau), curve.dt2(tau)]
                for tau in tau_list
            ],
            device=self.device,
            dtype=samples.dtype,
        )  # [n_points, 4]

        # apply ema if active
        if self.ema_b1 is not None:
            self.ema_b1.apply_to(self.net_b1)
        if self.ema_b2 is not None:
            self.ema_b2.apply_to(self.net_b2)
        if self.ema_eta is not None:
            self.ema_eta.apply_to(self.net_eta)

        try:
            # chunked inference: vmap over leading dim of t_data to avoid oom
            chunk_size = max(1, 100000 // n_samples)
            compute_vmapped = torch.vmap(
                self._compute_time_score_single,
                in_dims=(0, None),
                out_dims=0,
                randomness="different",  # required for hutchinson noise inside
            )

            time_score_chunks = []
            for i in range(0, n_points, chunk_size):
                t_chunk = t_data[i : i + chunk_size]  # [chunk_len, 4]
                chunk_scores = compute_vmapped(t_chunk, samples).detach()  # [chunk_len, n_samples]
                time_score_chunks.append(chunk_scores)

            time_scores = torch.cat(time_score_chunks, dim=0)  # [n_points, n_samples]

            # integrate via trapezoidal rule
            return -torch.trapezoid(time_scores, tau_vals, dim=0).cpu()  # [n_samples]
        finally:
            # restore original weights if ema was applied
            if self.ema_b1 is not None:
                self.ema_b1.restore(self.net_b1)
            if self.ema_b2 is not None:
                self.ema_b2.restore(self.net_b2)
            if self.ema_eta is not None:
                self.ema_eta.restore(self.net_eta)

    def _compute_time_score_single(
        self, t_tau: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute time-score d log rho / d tau at a single tau via 2D-time decomposition.

        Unpack curve outputs (t_1, t_2, dt_1/dtau, dt_2/dtau), evaluate b_1, b_2, eta,
        compute divergences separately, and combine via chain rule.

        Args:
            t_tau: [4] packed (t_1, t_2, dt_1/dtau, dt_2/dtau) as 0-d slices
                   produced by outer vmap over [n_points, 4].
            x: [n_samples, D] test points (broadcast, not vmapped).

        Returns:
            [n_samples] time scores at the current tau.
        """
        # unpack 0-d tensors from curve
        t1_s = t_tau[0]  # 0-d
        t2_s = t_tau[1]  # 0-d
        dt1_s = t_tau[2]  # 0-d
        dt2_s = t_tau[3]  # 0-d

        n_samples = x.shape[0]

        # expand 0-d tensors to [n_samples, 1] for network interface
        t1_batch = t1_s.view(1, 1).expand(n_samples, 1)  # [n_samples, 1]
        t2_batch = t2_s.view(1, 1).expand(n_samples, 1)  # [n_samples, 1]

        # gamma at scalar (t_1, t_2)
        gamma_t = self.path.gamma(t1_s.view(1, 1), t2_s.view(1, 1)).squeeze()  # 0-d

        # network forwards (full batch)
        b1_pred = self.net_b1(t1_batch, t2_batch, x)  # [n_samples, D]
        b2_pred = self.net_b2(t1_batch, t2_batch, x)  # [n_samples, D]
        eta_pred = self.net_eta(t1_batch, t2_batch, x)  # [n_samples, D]

        # divergence via vmap(jacrev) per head; two separate calls, do NOT collapse
        t1_one = t1_s.view(1, 1)
        t2_one = t2_s.view(1, 1)

        def b1_single(x_single):
            return self.net_b1(t1_one, t2_one, x_single.unsqueeze(0)).squeeze(0)

        def b2_single(x_single):
            return self.net_b2(t1_one, t2_one, x_single.unsqueeze(0)).squeeze(0)

        # divergences via the init-bound _div_fn (no per-tau branch on div_method).
        div_b1 = self._div_fn(b1_single, x)  # [n_samples]
        div_b2 = self._div_fn(b2_single, x)  # [n_samples]

        # dot products
        b1_dot_eta = (b1_pred * eta_pred).sum(dim=-1)  # [n_samples]
        b2_dot_eta = (b2_pred * eta_pred).sum(dim=-1)  # [n_samples]

        # directional time-score components
        s_1 = -div_b1 + b1_dot_eta / gamma_t  # [n_samples]
        s_2 = -div_b2 + b2_dot_eta / gamma_t  # [n_samples]

        # combine via curve derivatives (chain rule)
        time_score = s_1 * dt1_s + s_2 * dt2_s  # [n_samples]
        return time_score
