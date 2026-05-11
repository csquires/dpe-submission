"""flow matching density ratio estimator with triangular p_* trajectory.

estimates log p0(x) / p1(x) via conditional flow matching with three samples
(p0, p1, p*). the estimator trains a velocity field on all three sample sets,
using p* as an intermediate reference point (e.g., geometric-mean Gaussian).
at inference, integrates a ratio ode to estimate log density ratios.

reference: arXiv:2602.24201 (triangular setting with p* trajectory).
"""

from typing import Optional
import torch
from torch import Tensor

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.flow.multiclass_vel_score_mlp import MultiClassVelScoreMLP
from src.density_ratio_estimation._trainer import train_score_flow
from src.density_ratio_estimation._losses import triangular_flow_matching_loss
from src.models.flow.ratio_ode import ratio_ode_triangular
from src.models.flow.time_sampler import UniformTimeSampler


class TriangularFMDRE(DensityRatioEstimator):
    """Flow matching density ratio estimator with triangular p_* trajectory.

    Unlike FMDRE (S1) which trains on two samples (p0, p1) with binary condition,
    TriangularFMDRE trains on three samples (p0, p1, p*) with ternary condition,
    where p* is an intermediate reference point. This enables more flexible density
    ratio estimation via a three-class velocity field.

    Training: delegates to train_score_flow with triangular_flow_matching_loss
    and stratified batch sampling from three distributions.
    Inference: ODE integration from t=eps to t=1-eps via ratio_ode_triangular.
    """

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
        """Initialize triangular FMDRE estimator with hyperparameters.

        Procedure:
          1. Call super().__init__(input_dim) to store input dimension
          2. Store all hyperparameters as instance attributes
          3. Auto-detect device (cuda if available, else cpu)
          4. Initialize model placeholder to None

        Args:
            input_dim: dimensionality of data samples
            hidden_dim: width of hidden layers in velocity mlp (default 256)
            n_epochs: number of training epochs (default 1000)
            batch_size: training batch size (default 512); split 1/3 per class
            lr: adam learning rate (default 2e-3)
            score_weight: weight on score matching loss (default 1.0)
            eps: time clamping to [eps, 1-eps] (default 0.01)
            device: target device ('cuda' or 'cpu'); auto-detects if None
            integration_steps: number of ode integration steps (default 10000)
            div_method: divergence estimation method (default 'hutch_rademacher')
            verbose: print training progress (default False)
            log_every: epoch interval for verbose logging (default 100)
            n_hidden_layers: number of hidden layers in velocity mlp (default 3)
            adam_betas: adam beta parameters (default (0.9, 0.999))
            weight_decay: L2 weight decay (default 0.0)
            cosine_min_factor: cosine scheduler minimum factor in [0, 1] (default 1.0);
                              if < 1.0, enables cosine annealing
            triangular_p_uncond: per-sample probability of CFG-style dropout on class condition
                                (default 0.0); must be in [0, 1]
            layernorm: layernorm placement {'off', 'pre', 'post'} (default 'off')

        Raises:
            ValueError: if cosine_min_factor not in [0, 1] or triangular_p_uncond not in [0, 1]
                        or layernorm not in {'off', 'pre', 'post'}
        """
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
        """Instantiate MultiClassVelScoreMLP with K=3.

        Procedure:
          1. Create MultiClassVelScoreMLP(input_dim, num_classes=3, hidden_dim, n_hidden_layers)
          2. Move model to device
          3. Store in self.model
        """
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
        """Train the velocity field on samples from p0, p1, and p*.

        Procedure:
          1. Call init_model() to instantiate the model
          2. Convert all inputs to float32 and move to device
          3. Create time sampler: UniformTimeSampler(eps=self.eps)
          4. Create optimizer: Adam with adam_betas, weight_decay, eps=1e-8
          5. Create scheduler:
             - If cosine_min_factor < 1.0: CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr*cosine_min_factor)
             - Else: None
          6. Construct loss_kwargs = {"score_weight": self.score_weight, "triangular_p_uncond": self.triangular_p_uncond}
          7. Call train_score_flow with:
               model, samples_p0, samples_p1, samples_pstar, triangular_flow_matching_loss, optimizer,
               n_steps=n_epochs, batch_size=batch_size, time_sampler,
               scheduler, loss_kwargs, eps=self.eps, model_module=self.model
          8. Set model to eval mode
          9. Print completion message if verbose (prefix "[TriangularFMDRE]")

        Args:
            samples_p0: [N0, D] samples from p0
            samples_p1: [N1, D] samples from p1
            samples_pstar: [N*, D] samples from p* (intermediate reference)

        Note: This method extends the base class fit(samples_p0, samples_p1)
              with an additional samples_pstar argument. This is project convention
              for triangular estimators (TriangularCTSM, TriangularVFM).
        """
        self.init_model()

        samples_p0 = samples_p0.float().to(self.device)
        samples_p1 = samples_p1.float().to(self.device)
        samples_pstar = samples_pstar.float().to(self.device)

        # create time sampler
        time_sampler = UniformTimeSampler(eps=self.eps)

        # create optimizer: Adam with given betas, weight_decay, eps=1e-8
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=self.adam_betas,
            eps=1e-8,
            weight_decay=self.weight_decay,
        )

        # create scheduler: cosine annealing if cosine_min_factor < 1.0, else None
        scheduler = None
        if self.cosine_min_factor < 1.0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.n_epochs,
                eta_min=self.lr * self.cosine_min_factor,
            )

        # construct loss kwargs for triangular_flow_matching_loss
        loss_kwargs = {
            "score_weight": self.score_weight,
            "triangular_p_uncond": self.triangular_p_uncond,
        }

        # delegate training to train_score_flow
        train_score_flow(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_fn=triangular_flow_matching_loss,
            optim=optimizer,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler=scheduler,
            loss_kwargs=loss_kwargs,
            eps=self.eps,
            model_module=self.model,
        )

        if self.verbose:
            print("[TriangularFMDRE] Training complete")

        self.model.eval()

    def predict_ldr(self, xs: Tensor) -> Tensor:
        """Estimate log p0(x) / p1(x) via ratio ODE with triangular trajectory.

        Procedure:
          1. Validate model is fitted (raise RuntimeError if not)
          2. Set model to eval mode
          3. Move samples to device and convert to float32
          4. Call ratio_ode_triangular with hyperparameters
          5. Return result on CPU, detached

        Args:
            xs: [B, D] query points

        Returns:
            [B] log density ratio tensor on CPU

        Raises:
            RuntimeError: if model is not trained (self.model is None)
        """
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
