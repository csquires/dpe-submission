"""flow-matching DRE with classifier-free guidance (S2): integrates the unconditional ratio ODE.

reference: arXiv:2602.24201 (S2 setting).
"""

from typing import Callable, Optional
import warnings
import torch

from ...common.base import DRE
from ..common._losses import make_fm_loss
from ..common._trainer import train_loop
from ..common._cfgs import (
    OptimCfg,
    SchedCfg,
    EmaCfg,
    TimeCfg,
    make_optim,
    make_sched,
    make_ema,
    make_time_sampler,
)
from ..common._precond import endpoint_moments, make_coeffs, make_lambda, wrap_fm
from src.models.flow.cond_vel_score_mlp import CondVelScoreMLP
from src.models.flow.ratio_ode import ratio_ode_s2


class FMDRE_S2(DRE):
    """flow-matching DRE under `fm_loss` with CFG dropout; predict_ldr uses the unconditional trajectory."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_shared_layers: int = 3,
        n_steps: int = 1000,
        batch_size: int = 512,
        *,
        optim: OptimCfg,
        sched: SchedCfg = SchedCfg(),
        ema: EmaCfg = EmaCfg(),
        time: TimeCfg = TimeCfg(),
        device: Optional[str] = None,
        # FMDRE_S2-specific (explicit)
        score_weight: float = 1.0,
        div_method: str = "hutch_rademacher",
        n_hutch_samples: int = 1,
        integration_steps: int = 10000,
        p_uncond: float = 0.1,
        sentinel_cond: float = -1.0,
        reweight: bool = False,
        precond: bool = False,
        early_stop_cfg: dict | None = None,
    ) -> None:
        """construct an FMDRE_S2 estimator with cfg-based hyperparameters.

        Args:
            input_dim: feature dimension.
            hidden_dim: hidden-layer width (default 256).
            n_hidden_layers: MLP depth (default 3); see CondVelScoreMLP for
                exact accounting (output projection not counted).
            n_shared_layers: hidden rounds in the shared backbone (default 3 =
                fully shared, same as pre-split FMDRE_S2). must satisfy
                1 <= n_shared_layers <= n_hidden_layers.
            n_steps: training steps (default 1000).
            batch_size: mini-batch size (default 512).
            optim: optimizer config (required, no default).
            sched: scheduler config (default SchedCfg() disables annealing).
            ema: ema config (default EmaCfg() disables ema).
            time: time-sampling config (default TimeCfg() uses uniform, eps=1e-3).
            device: torch device string or None for auto (default None).
            score_weight: loss weight for score loss (default 1.0).
            div_method: divergence estimator for ode integration (default "hutch_rademacher").
            n_hutch_samples: averaging count for the hutchinson estimator (default 1).
            integration_steps: ode integration steps (default 10000).
            p_uncond: cfg dropout probability in [0, 1] (default 0.1).
            sentinel_cond: sentinel value for unconditional signal (default -1.0).
            reweight: whether to reweight loss (default False).
            precond: whether to apply Karras (EDM) preconditioning (default False).
            early_stop_cfg: early stopping config dict or None (default None).
        """
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_shared_layers = n_shared_layers
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.optim = optim
        self.sched = sched
        self.ema = ema
        self.time = time
        self.score_weight = score_weight
        self.div_method = div_method
        self.n_hutch_samples = n_hutch_samples
        self.integration_steps = integration_steps
        self.p_uncond = p_uncond
        self.sentinel_cond = sentinel_cond
        self.reweight = reweight
        self.precond = precond
        self.early_stop_cfg = early_stop_cfg
        self._moments = None

        if not (0.0 <= p_uncond <= 1.0):
            raise ValueError(f"p_uncond must be in [0.0, 1.0], got {p_uncond}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None

    def init_model(self) -> None:
        """instantiate the conditional velocity MLP on self.device."""
        self.model = CondVelScoreMLP(
            self.input_dim,
            self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            n_shared_layers=self.n_shared_layers,
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
        """train the velocity field on p0 -> p1 flow using fm_loss with cfg dropout.

        procedure:
          1. init_model() builds CondVelScoreMLP.
          2. cast samples to float.
          3. instantiate optim, scheduler, ema, time_sampler from cfgs.
          4. optionally compute endpoint moments and preconditioning coefficients.
          5. delegate to train_loop with fm_loss and cfg guidance (p_uncond > 0).
          6. set model.eval().

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
        meta_out: dict = {}
        self.init_model()
        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()

        optim_obj = make_optim(self.model.parameters(), self.optim)
        sched_obj = make_sched(optim_obj, self.n_steps, self.optim.lr, self.sched)
        ema_obj = make_ema(self.model, self.ema)
        time_sampler = make_time_sampler(self.time)

        # optionally compute endpoint moments and preconditioning coefficients
        if self.precond:
            x_data_all = torch.cat([samples_p0, samples_p1], dim=0)
            self._moments = endpoint_moments({"x_data": x_data_all})
            coeff_v = make_coeffs("fm", self._moments, "velocity")
            coeff_s = make_coeffs("fm", self._moments, "score")
            if self.reweight:
                warnings.warn(
                    "precond=True overrides reweight=True (both normalize loss via "
                    "time-dependent weights; lambda=c_out^-2 replaces 1-tau^2). "
                    "set reweight=False to suppress this warning.",
                    UserWarning,
                )
        else:
            coeff_v = None
            coeff_s = None

        loss_fn = make_fm_loss(
            score_weight=self.score_weight,
            p_uncond=self.p_uncond,
            sentinel_cond=self.sentinel_cond,
            reweight=self.reweight,
            outer_weight=make_lambda(coeff_v) if self.precond else None,
        )

        # resolve the model once: wrap if precond, else raw
        if self.precond:
            model_to_train = wrap_fm(self.model, coeff_v, coeff_s)
        else:
            model_to_train = self.model

        # build eval_fn if both callbacks and eval data are provided.
        # eval_data["pstar"] is the holdout eval inputs; "true_ldrs" the paired
        # ground-truth log-density-ratios. _model arg is ignored; closure captures self.
        eval_fn = None
        if step_cb is not None and eval_data is not None:
            eval_pstar = eval_data["pstar"]
            eval_true_ldrs = eval_data["true_ldrs"]

            def eval_fn(_model) -> torch.Tensor:
                """mae between predict_ldr(eval_pstar) and true_ldrs."""
                predicted = self.predict_ldr(eval_pstar)
                target = eval_true_ldrs.to(predicted.device)
                return torch.abs(predicted - target).mean()

        train_loop(
            model=model_to_train,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_fn=loss_fn,
            optim=optim_obj,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler=sched_obj,
            ema=ema_obj,
            grad_clip_norm=self.optim.grad_clip_norm,
            eps=self.time.eps,
            model_module=self.model,
            step_cb=step_cb,
            eval_fn=eval_fn,
            step_cb_interval=step_cb_interval,
            early_stop_cfg=self.early_stop_cfg,
            _meta_out=meta_out,
        )
        self._final_step = meta_out.get("final_step", self.n_steps)
        self._stop_reason = meta_out.get("stop_reason", None)
        self.model.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """run the S2 (unconditional-trajectory) ratio ODE on xs and return log(p0/p1) on CPU, detached."""
        if self.model is None:
            raise RuntimeError("FMDRE_S2 model is not trained. Call fit() before predict_ldr().")

        self.model.eval()

        # rebuild the preconditioned wrapper if trained with precond=True
        if self.precond:
            if self._moments is None:
                raise RuntimeError(
                    "precond=True but self._moments is None; "
                    "model was not trained with precond=True."
                )
            coeff_v_infer = make_coeffs("fm", self._moments, "velocity")
            coeff_s_infer = make_coeffs("fm", self._moments, "score")
            model_to_infer = wrap_fm(self.model, coeff_v_infer, coeff_s_infer)
        else:
            model_to_infer = self.model

        samples = xs.float().to(self.device)

        ldr = ratio_ode_s2(
            model_to_infer,
            samples,
            steps=self.integration_steps,
            eps=self.time.eps,
            device=str(self.device),
            div_method=self.div_method,
            n_hutch_samples=self.n_hutch_samples,
            uncond_cond=self.sentinel_cond,
            warn_uncond=False,
        )

        return ldr.detach().cpu()
