"""3-class flow-matching DRE with intermediate distribution p*.

reference: arXiv:2602.24201 (triangular setting).
"""

from typing import Optional
import torch
from torch import Tensor

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.flow.multiclass_vel_score_mlp import MultiClassVelScoreMLP
from src.density_ratio_estimation._trainer import train_loop
from src.density_ratio_estimation._losses import tri_fm_loss
from src.models.flow.ratio_ode import ratio_ode_triangular
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


class TriangularFMDRE(DensityRatioEstimator):
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

    def fit(self, samples_p0: Tensor, samples_p1: Tensor, samples_pstar: Tensor) -> None:
        """init model + cfg-based optimizer, scheduler, EMA, time-sampler; delegate to train_loop."""
        self.init_model()
        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)
        samples_pstar = samples_pstar.float().to(self.device)

        optim_obj = make_optim(self.model.parameters(), self.optim)
        sched_obj = make_sched(optim_obj, self.n_epochs, self.optim.lr, self.sched)
        ema_obj = make_ema(self.model, self.ema)
        time_sampler = make_time_sampler(self.time)

        train_loop(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_fn=tri_fm_loss,
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
                "triangular_p_uncond": self.triangular_p_uncond,
            },
            model_module=self.model,
        )
        self.model.eval()

    def predict_ldr(self, xs: Tensor) -> Tensor:
        """run the triangular ratio ODE on xs and return log(p0/p1) on CPU, detached."""
        if self.model is None:
            raise RuntimeError(
                "TriangularFMDRE model is not trained. Call fit() before predict_ldr()."
            )

        self.model.eval()

        samples = xs.float().to(self.device)

        ldr = ratio_ode_triangular(
            self.model,
            samples,
            steps=self.integration_steps,
            eps=self.time.eps,
            device=str(self.device),
            div_method=self.div_method,
        )

        return ldr.detach().cpu()
