"""flow-matching DRE: trains a conditional velocity field then integrates a ratio ODE."""

from typing import Callable, Optional
import torch
import warnings

from ...common.base import DRE
from ..common._losses import make_fm_loss
from ..common._precond import endpoint_moments, make_coeffs, make_lambda, wrap_fm
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
from src.models.flow.cond_vel_score_mlp import CondVelScoreMLP
from src.models.flow.ratio_ode import ratio_ode


class FMDRE(DRE):
    """flow-matching DRE under `fm_loss`; predict_ldr runs the ratio ODE."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 3,
        n_shared_layers: int = 3,
        n_epochs: int = 1000,
        batch_size: int = 512,
        *,
        optim: OptimCfg,
        sched: SchedCfg = SchedCfg(),
        ema: EmaCfg = EmaCfg(),
        time: TimeCfg = TimeCfg(),
        device: Optional[str] = None,
        # FMDRE-specific (explicit)
        score_weight: float = 1.0,
        div_method: str = "hutch_rademacher",
        n_hutch_samples: int = 1,
        integration_steps: int = 10000,
        reweight: bool = False,
        precond: bool = False,
    ) -> None:
        """construct an FMDRE estimator with cfg-based hyperparameters.

        Args:
            input_dim: feature dimension.
            hidden_dim: hidden-layer width (default 256).
            n_hidden_layers: MLP depth (default 3); see CondVelScoreMLP for
                exact accounting (output projection not counted).
            n_shared_layers: hidden rounds in the shared backbone (default 3 =
                fully shared, same as pre-split FMDRE). must satisfy
                1 <= n_shared_layers <= n_hidden_layers.
            n_epochs: training steps (default 1000).
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
            precond: optional Karras preconditioning on both heads (velocity, score).
                If True, wraps the network with learned-parameter-free coefficients
                (default False, byte-identical to current behaviour).
        """
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_shared_layers = n_shared_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optim = optim
        self.sched = sched
        self.ema = ema
        self.time = time
        self.score_weight = score_weight
        self.div_method = div_method
        self.n_hutch_samples = n_hutch_samples
        self.reweight = reweight
        self.integration_steps = integration_steps
        self.precond = precond
        self._moments = None

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
        """train the velocity field on p0 -> p1 flow using fm_loss.

        procedure:
          1. init_model() builds CondVelScoreMLP.
          2. cast samples to float.
          3. instantiate optim, scheduler, ema, time_sampler from cfgs.
          4. delegate to train_loop with fm_loss and no CFG guidance.
          5. set model.eval().

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
        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()

        if self.precond:
            samples_p0_device = samples_p0.to(self.device)
            samples_p1_device = samples_p1.to(self.device)
            x_data = torch.cat([samples_p0_device, samples_p1_device], dim=0)
            self._moments = endpoint_moments({"x_data": x_data})
            coeff_v = make_coeffs("fm", self._moments, "velocity")
            coeff_s = make_coeffs("fm", self._moments, "score")
            if self.reweight:
                warnings.warn(
                    "precond=True and reweight=True: reweight is ignored; EDM lambda "
                    "subsumes variance normalization.",
                    UserWarning,
                )

        optim_obj = make_optim(self.model.parameters(), self.optim)
        sched_obj = make_sched(optim_obj, self.n_epochs, self.optim.lr, self.sched)
        ema_obj = make_ema(self.model, self.ema)
        time_sampler = make_time_sampler(self.time)
        outer_weight = make_lambda(coeff_v) if self.precond else None
        loss_fn = make_fm_loss(
            score_weight=self.score_weight,
            p_uncond=0.0,
            sentinel_cond=-1.0,
            reweight=self.reweight,
            outer_weight=outer_weight,
        )

        model_to_train = wrap_fm(self.model, coeff_v, coeff_s) if self.precond else self.model
        model_module = self.model  # raw net is always the parameter-carrying module

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
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler=sched_obj,
            ema=ema_obj,
            grad_clip_norm=self.optim.grad_clip_norm,
            eps=self.time.eps,
            model_module=model_module,
            step_cb=step_cb,
            eval_fn=eval_fn,
            step_cb_interval=step_cb_interval,
        )
        self.model.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """run the ratio ODE on xs and return log(p0/p1) on CPU, detached."""
        if self.model is None:
            raise RuntimeError("FMDRE model is not trained. Call fit() before predict_ldr().")

        self.model.eval()

        if self.precond:
            coeff_v = make_coeffs("fm", self._moments, "velocity")
            coeff_s = make_coeffs("fm", self._moments, "score")
            model_to_infer = wrap_fm(self.model, coeff_v, coeff_s)
        else:
            model_to_infer = self.model

        samples = xs.float().to(self.device)

        ldr = ratio_ode(
            model_to_infer,
            samples,
            steps=self.integration_steps,
            eps=self.time.eps,
            device=str(self.device),
            div_method=self.div_method,
            n_hutch_samples=self.n_hutch_samples,
        )

        return ldr.detach().cpu()
