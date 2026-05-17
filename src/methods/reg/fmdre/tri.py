"""3-class flow-matching DRE with intermediate distribution p*.

reference: arXiv:2602.24201 (triangular setting).
"""

from typing import Callable, Optional
import warnings
import torch
from torch import Tensor

from ...common.base import ELDR
from src.models.flow.multiclass_vel_score_mlp import MultiClassVelScoreMLP
from ..common._trainer import train_loop
from ..common._losses import make_tri_fm_loss
from ..common._precond import endpoint_moments, make_coeffs, make_lambda, wrap_fm
from src.models.flow.ratio_ode import ratio_ode_triangular
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


class TriangularFMDRE(ELDR):
    """3-class flow-matching DRE under `tri_fm_loss`; predict_ldr runs the triangular ratio ODE."""

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
        time: TimeCfg = TimeCfg(),
        device: Optional[str] = None,
        # TriFMDRE-specific (explicit)
        score_weight: float = 1.0,
        div_method: str = "hutch_rademacher",
        integration_steps: int = 10000,
        triangular_p_uncond: float = 0.0,
        layernorm: str = "off",
        reweight: bool = False,
        precond: bool = False,
    ) -> None:
        """construct estimator with cfg-based optimizer, scheduler, EMA, time-sampler.

        Args:
            input_dim: data feature dimension.
            hidden_dim: MLP hidden dimension (default 256).
            n_hidden_layers: number of hidden layers (default 3).
            n_epochs: training steps (default 1000).
            batch_size: mini-batch size (default 512).
            optim: optimizer config (required, no default).
            sched: scheduler config (default SchedCfg() disables annealing).
            ema: EMA config (default EmaCfg() disables EMA).
            time: time-sampler config (default TimeCfg() uses uniform dist, eps=1e-3).
            device: torch device string (default auto-detect cuda).
            score_weight: weight for score-matching loss term (default 1.0).
            div_method: divergence estimator method (default "hutch_rademacher").
            integration_steps: ODE solver steps at predict time (default 10000).
            triangular_p_uncond: probability of dropping class condition (default 0.0).
            layernorm: layer norm mode in {"off", "pre", "post"} (default "off").
            precond: enable Karras preconditioning (default False).
        """
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.score_weight = score_weight
        self.integration_steps = integration_steps
        self.div_method = div_method
        self.n_hidden_layers = n_hidden_layers

        # cfg-based attributes
        self.optim = optim
        self.sched = sched
        self.ema = ema
        self.time = time

        # validate triangular_p_uncond
        if not (0.0 <= triangular_p_uncond <= 1.0):
            raise ValueError(f"triangular_p_uncond must be in [0, 1], got {triangular_p_uncond}")
        self.triangular_p_uncond = float(triangular_p_uncond)

        # validate layernorm
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off', 'pre', 'post'}}; got {layernorm!r}")
        self.layernorm = layernorm
        self.reweight = reweight
        self.precond = precond
        self._moments = None

        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None

    def init_model(self) -> None:
        """instantiate MultiClassVelScoreMLP with K=3 on self.device."""
        self.model = MultiClassVelScoreMLP(
            self.input_dim,
            num_classes=3,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            layernorm=self.layernorm,
        ).to(self.device)

    def fit(
        self,
        samples_p0: Tensor,
        samples_p1: Tensor,
        samples_pstar: Tensor,
        *,
        step_cb: Callable[[int, float], None] | None = None,
        eval_data: dict[str, torch.Tensor] | None = None,
        step_cb_interval: int = 50,
    ) -> None:
        """init model + cfg-based optimizer, scheduler, EMA, time-sampler; delegate to train_loop.

        Args:
            samples_p0: [N0, input_dim] samples from p0.
            samples_p1: [N1, input_dim] samples from p1.
            samples_pstar: [N*, input_dim] samples from p*.
            step_cb: optional callback invoked every step_cb_interval steps with
                (step_index: int, score: float). when None, no instrumentation.
            eval_data: optional dict with keys "pstar" and "true_ldrs" (paired by
                index). used to build eval_fn closure only if step_cb is not None.
                when None, eval_fn is not constructed.
            step_cb_interval: interval (in minibatch updates) for step_cb invocation
                (default 50).
        """
        self.init_model()
        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)
        samples_pstar = samples_pstar.float().to(self.device)

        # compute preconditioning moments from tripled endpoint samples if enabled
        if self.precond:
            x_data_tripled = torch.cat([samples_p0, samples_p1, samples_pstar], dim=0)
            self._moments = endpoint_moments({"x_data": x_data_tripled})

        optim_obj = make_optim(self.model.parameters(), self.optim)
        sched_obj = make_sched(optim_obj, self.n_epochs, self.optim.lr, self.sched)
        ema_obj = make_ema(self.model, self.ema)
        time_sampler = make_time_sampler(self.time)

        # build preconditioning coefficients if enabled; warn if reweight and precond both set
        outer_weight_fn = None
        coeff_v = None
        coeff_s = None
        if self.precond:
            coeff_v = make_coeffs("fm", self._moments, "velocity")
            coeff_s = make_coeffs("fm", self._moments, "score")
            outer_weight_fn = make_lambda(coeff_v)
            if self.reweight:
                warnings.warn(
                    "precond=True and reweight=True: reweight is ignored; "
                    "EDM lambda replaces path_var weighting.",
                    UserWarning,
                )

        loss_fn = make_tri_fm_loss(
            score_weight=self.score_weight,
            triangular_p_uncond=self.triangular_p_uncond,
            reweight=self.reweight if not self.precond else False,
            outer_weight=outer_weight_fn,
        )

        # build eval_fn if both callbacks and eval data are provided
        eval_fn = None
        if step_cb is not None and eval_data is not None:
            eval_pstar = eval_data["pstar"]
            eval_true_ldrs = eval_data["true_ldrs"]

            def eval_fn(_model) -> Tensor:
                """compute MAE between predict_ldr(pstar_eval) and true_ldrs_eval.

                predict_ldr runs ratio_ode_triangular under the hood; integration
                cost is controlled by self.integration_steps (a hyperparameter in HPO
                search space). _model arg is ignored; closure captures self.
                """
                predicted = self.predict_ldr(eval_pstar)
                target = eval_true_ldrs.to(predicted.device)
                return torch.abs(predicted - target).mean()

        # resolve the wrapped model once (if precond); pass raw net as model_module for EMA/optim
        model_for_training = self.model
        if self.precond:
            model_for_training = wrap_fm(
                self.model, coeff_v, coeff_s, onehot=True
            )

        train_loop(
            model=model_for_training,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_fn=loss_fn,
            optim=optim_obj,
            n_steps=self.n_epochs,
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
        )
        self.model.eval()

    def predict_ldr(self, xs: Tensor) -> Tensor:
        """run the triangular ratio ODE on xs and return log(p0/p1) on CPU, detached.

        if precond=True, rebuilds the preconditioned wrapper from stored moments
        and threads it into ratio_ode_triangular for end-to-end differentiation.
        EMA is applied to the raw net first; the wrapper is constructed after.
        """
        if self.model is None:
            raise RuntimeError(
                "TriangularFMDRE model is not trained. Call fit() before predict_ldr()."
            )

        self.model.eval()

        samples = xs.float().to(self.device)

        # thread the preconditioned model into the ODE solver if enabled
        model_for_inference = self.model
        if self.precond:
            model_for_inference = wrap_fm(
                self.model,
                make_coeffs("fm", self._moments, "velocity"),
                make_coeffs("fm", self._moments, "score"),
                onehot=True,
            )

        ldr = ratio_ode_triangular(
            model_for_inference,
            samples,
            steps=self.integration_steps,
            eps=self.time.eps,
            device=str(self.device),
            div_method=self.div_method,
        )

        return ldr.detach().cpu()
