"""flow-matching DRE with classifier-free guidance (S2): integrates the unconditional ratio ODE.

reference: arXiv:2602.24201 (S2 setting).
"""

from typing import Optional
import torch

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._losses import fm_loss
from src.density_ratio_estimation._trainer import train_loop
from src.density_ratio_estimation._cfgs import (
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
from src.models.flow.ratio_ode import ratio_ode_s2


class FMDRE_S2(DensityRatioEstimator):
    """flow-matching DRE under `fm_loss` with CFG dropout; predict_ldr uses the unconditional trajectory."""

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
        # FMDRE_S2-specific (explicit)
        score_weight: float = 1.0,
        div_method: str = "hutch_rademacher",
        integration_steps: int = 10000,
        p_uncond: float = 0.1,
        sentinel_cond: float = -1.0,
        reweight: bool = False,
    ) -> None:
        """construct an FMDRE_S2 estimator with cfg-based hyperparameters.

        Args:
            input_dim: feature dimension.
            hidden_dim: hidden-layer width (default 256).
            n_hidden_layers: MLP depth (default 3).
            n_epochs: training steps (default 1000).
            batch_size: mini-batch size (default 512).
            optim: optimizer config (required, no default).
            sched: scheduler config (default SchedCfg() disables annealing).
            ema: ema config (default EmaCfg() disables ema).
            time: time-sampling config (default TimeCfg() uses uniform, eps=1e-3).
            device: torch device string or None for auto (default None).
            score_weight: loss weight for score loss (default 1.0).
            div_method: divergence estimator for ode integration (default "hutch_rademacher").
            integration_steps: ode integration steps (default 10000).
            p_uncond: cfg dropout probability in [0, 1] (default 0.1).
            sentinel_cond: sentinel value for unconditional signal (default -1.0).
            reweight: whether to reweight loss (default False).
        """
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optim = optim
        self.sched = sched
        self.ema = ema
        self.time = time
        self.score_weight = score_weight
        self.div_method = div_method
        self.integration_steps = integration_steps
        self.p_uncond = p_uncond
        self.sentinel_cond = sentinel_cond
        self.reweight = reweight

        if not (0.0 <= p_uncond <= 1.0):
            raise ValueError(f"p_uncond must be in [0.0, 1.0], got {p_uncond}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None

    def init_model(self) -> None:
        """instantiate the conditional velocity MLP on self.device."""
        self.model = CondVelScoreMLP(self.input_dim, self.hidden_dim, n_hidden_layers=self.n_hidden_layers).to(self.device)

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """train the velocity field on p0 -> p1 flow using fm_loss with cfg dropout.

        procedure:
          1. init_model() builds CondVelScoreMLP.
          2. cast samples to float.
          3. instantiate optim, scheduler, ema, time_sampler from cfgs.
          4. delegate to train_loop with fm_loss and cfg guidance (p_uncond > 0).
          5. set model.eval().
        """
        self.init_model()
        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()

        optim_obj = make_optim(self.model.parameters(), self.optim)
        sched_obj = make_sched(optim_obj, self.n_epochs, self.optim.lr, self.sched)
        ema_obj = make_ema(self.model, self.ema)
        time_sampler = make_time_sampler(self.time)

        train_loop(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_fn=fm_loss,
            optim=optim_obj,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler=sched_obj,
            ema=ema_obj,
            grad_clip_norm=self.optim.grad_clip_norm,
            eps=self.time.eps,
            loss_kwargs={
                "score_weight": self.score_weight,
                "p_uncond": self.p_uncond,
                "sentinel_cond": self.sentinel_cond,
                "reweight": self.reweight,
            },
        )
        self.model.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """run the S2 (unconditional-trajectory) ratio ODE on xs and return log(p0/p1) on CPU, detached."""
        if self.model is None:
            raise RuntimeError("FMDRE_S2 model is not trained. Call fit() before predict_ldr().")

        self.model.eval()

        samples = xs.float().to(self.device)

        ldr = ratio_ode_s2(
            self.model,
            samples,
            steps=self.integration_steps,
            eps=self.time.eps,
            device=str(self.device),
            div_method=self.div_method,
            uncond_cond=self.sentinel_cond,
            warn_uncond=False,
        )

        return ldr.detach().cpu()
