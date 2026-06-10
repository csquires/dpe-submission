"""Time Score Matching (TSM) density ratio estimator."""

from typing import Callable, Optional

import torch

from ...common.base import DRE
from ..common._trainer import train_loop
from ..common._losses import tsm_loss
from ..common._cfgs import (
    OptimCfg, SchedCfg, EmaCfg, TimeCfg,
    make_optim, make_sched, make_ema, make_time_sampler
)
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D


class TSM(DRE):
    """time-score-matching DRE: trains s_phi(x, tau) under `tsm_loss`, integrates -score over tau."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_steps: int = 1000,
        batch_size: int = 512,
        *,
        optim: OptimCfg,
        sched: SchedCfg = None,
        ema: EmaCfg = None,
        time: TimeCfg = None,
        device: Optional[str] = None,
        # TSM-specific
        reweight: bool = False,
        activation: str = "silu",
        integration_steps: int = 200,
        early_stop_cfg: dict | None = None,
    ) -> None:
        """init TSM with cfg-based surface for optimization, scheduling, EMA, and time sampling.

        Args:
            input_dim: dimension of input space x.
            hidden_dim: width of score network hidden layers.
            n_hidden_layers: depth of score network (default 3).
            n_steps: training epochs.
            batch_size: batch size (default 512).
            optim: required OptimCfg instance (carries lr, grad_clip_norm, etc).
            sched: scheduler config (default SchedCfg() — no scheduling).
            ema: EMA config (default EmaCfg() — no EMA).
            time: time sampling and tau-domain margin config (carries eps, etc).
            device: torch device; auto-detect cuda/cpu if None.
            reweight: apply sample-reweighting in tsm_loss (default False).
            activation: score network activation; must be in {elu, gelu, silu}.
            integration_steps: tau-domain integration resolution for predict_ldr (default 200).
        """
        super().__init__(input_dim)
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )

        # store cfg objects
        self.optim = optim
        self.sched = sched if sched is not None else SchedCfg()
        self.ema = ema if ema is not None else EmaCfg()
        self.time = time if time is not None else TimeCfg()
        self.early_stop_cfg = early_stop_cfg

        # store hyperparams
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.reweight = reweight
        self.activation = activation
        self.integration_steps = integration_steps

        # resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None

    def init_model(self) -> None:
        """instantiate TimeScoreNetwork1D on device."""
        self.model = TimeScoreNetwork1D(
            self.input_dim,
            self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
        ).to(self.device)

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """init model, then delegate to train_loop with cfg-based factories.

        Orchestrates:
        - model instantiation via init_model()
        - optimizer factory (optim cfg -> Adam/SGD/...)
        - scheduler factory (sched cfg -> LR schedule or None)
        - EMA factory (ema cfg -> EMA or None)
        - time sampler factory (time cfg -> sampler callable)

        Args:
            samples_p0: [N0, input_dim] samples from p0.
            samples_p1: [N1, input_dim] samples from p1.
            step_cb: optional callback invoked every step_cb_interval steps with
                (step_index: int, score: float). when None, no instrumentation.
            eval_data: optional dict with keys "pstar" and "true_ldrs" (paired by
                index). used to build eval_fn closure only if step_cb is not None.
                when None, eval_fn is not constructed.
            step_cb_interval: interval (in minibatch updates) for step_cb invocation
                (default 50).
        """
        self.init_model()
        optim_obj = make_optim(self.model.parameters(), self.optim)
        sched_obj = make_sched(optim_obj, self.n_steps, self.optim.lr, self.sched)
        ema_obj = make_ema(self.model, self.ema)
        time_sampler = make_time_sampler(self.time)

        # build eval_fn if both callbacks and eval data are provided.
        eval_fn = None
        if step_cb is not None and eval_data is not None:
            eval_pstar = eval_data["pstar"]
            eval_true_ldrs = eval_data["true_ldrs"]

            def eval_fn(_model) -> torch.Tensor:
                """mae between predict_ldr(eval_pstar) and true_ldrs."""
                predicted = self.predict_ldr(eval_pstar)
                target = eval_true_ldrs.to(predicted.device)
                return torch.abs(predicted - target).mean()

        meta_out: dict = {}
        train_loop(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_fn=tsm_loss,
            optim=optim_obj,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler=sched_obj,
            ema=ema_obj,
            grad_clip_norm=self.optim.grad_clip_norm,
            eps=self.time.eps,
            loss_kwargs={"reweight": self.reweight, "eps": self.time.eps},
            step_cb=step_cb,
            eval_fn=eval_fn,
            step_cb_interval=step_cb_interval,
            early_stop_cfg=self.early_stop_cfg,
            _meta_out=meta_out,
        )
        self._final_step = meta_out.get("final_step", self.n_steps)
        self._stop_reason = meta_out.get("stop_reason", None)

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """trapezoid-integrate -model(xs, tau) over tau in [eps, 1] and return on CPU."""
        if self.model is None:
            raise RuntimeError("TSM is not trained. Call fit() before predict_ldr().")

        self.model.eval()
        xs = xs.float().to(self.device)
        with torch.no_grad():
            ts = torch.linspace(self.time.eps, 1.0, self.integration_steps, device=self.device)
            vals = torch.stack(
                [
                    -self.model(xs, torch.full((xs.shape[0], 1), t.item(), device=self.device)).squeeze(-1)
                    for t in ts
                ],
                dim=0,
            )
            dt = (1.0 - self.time.eps) / (self.integration_steps - 1)
            return torch.trapezoid(vals, dx=dt, dim=0).cpu()
