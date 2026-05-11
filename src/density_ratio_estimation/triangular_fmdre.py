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
from src.models.flow.time_sampler import UniformSampler


class TriangularFMDRE(DensityRatioEstimator):
    """3-class flow-matching DRE under `tri_fm_loss`; predict_ldr runs the triangular ratio ODE."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 2e-3,
        score_weight: float = 1.0,
        eps: float = 0.01,
        device: Optional[str] = None,
        integration_steps: int = 10000,
        div_method: str = "hutch_rademacher",
        verbose: bool = False,
        log_every: int = 100,
        n_hidden_layers: int = 3,
        adam_betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.0,
        cosine_min_factor: float = 1.0,
        triangular_p_uncond: float = 0.0,
        layernorm: str = "off",
    ) -> None:
        """cosine_min_factor=1.0 disables LR annealing; triangular_p_uncond drops the one-hot
        class condition with the given probability."""
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.score_weight = score_weight
        self.eps = eps
        self.integration_steps = integration_steps
        self.div_method = div_method
        self.verbose = verbose
        self.log_every = log_every
        self.n_hidden_layers = n_hidden_layers
        self.adam_betas = tuple(adam_betas)
        self.weight_decay = float(weight_decay)
        if not (0.0 <= cosine_min_factor <= 1.0):
            raise ValueError(f"cosine_min_factor must be in [0, 1], got {cosine_min_factor}")
        self.cosine_min_factor = float(cosine_min_factor)
        if not (0.0 <= triangular_p_uncond <= 1.0):
            raise ValueError(f"triangular_p_uncond must be in [0, 1], got {triangular_p_uncond}")
        self.triangular_p_uncond = float(triangular_p_uncond)
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off', 'pre', 'post'}}; got {layernorm!r}")
        self.layernorm = layernorm

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
    ) -> None:
        """init model + optimizer (+ optional cosine), then delegate to `train_loop`
        with `tri_fm_loss`."""
        self.init_model()
        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)
        samples_pstar = samples_pstar.float().to(self.device)

        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=self.adam_betas,
            eps=1e-8,
            weight_decay=self.weight_decay,
        )

        scheduler = None
        if self.cosine_min_factor < 1.0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.n_epochs, eta_min=self.lr * self.cosine_min_factor
            )

        train_loop(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_fn=tri_fm_loss,
            optim=opt,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=UniformSampler(eps=self.eps),
            scheduler=scheduler,
            loss_kwargs={
                "score_weight": self.score_weight,
                "triangular_p_uncond": self.triangular_p_uncond,
            },
            eps=self.eps,
            model_module=self.model,
        )

        if self.verbose:
            print("[TriangularFMDRE] training complete")
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
            eps=self.eps,
            device=str(self.device),
            div_method=self.div_method,
        )

        return ldr.detach().cpu()
