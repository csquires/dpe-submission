"""flow matching density ratio estimator (fmdre).

estimates log p0(x) / p1(x) via conditional flow matching with a ratio ode.
the estimator trains a velocity field on samples from p0 and p1 using conditional
flow matching, then integrates a ratio ode during inference to estimate log density ratios.
"""

from typing import Optional
import warnings
import torch

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._losses import flow_matching_loss
from src.density_ratio_estimation._trainer import train_score_flow
from src.models.flow.cond_vel_score_mlp import CondVelScoreMLP
from src.models.flow.ratio_ode import ratio_ode


class FMDRE(DensityRatioEstimator):
    """flow matching density ratio estimator.

    trains a conditional velocity field on samples from p0 and p1 using
    flow matching, then estimates log density ratios via ratio ode integration.

    hyperparameters control training dynamics (n_epochs, batch_size, lr) and
    inference precision (integration_steps, eps).
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
    ) -> None:
        """
        initialize fmdre estimator with hyperparameters.

        procedure:
          1. call super().__init__(input_dim) to store input dimension
          2. store all hyperparameters as instance attributes
          3. auto-detect device (cuda if available, else cpu)
          4. initialize model placeholder to None

        args:
            input_dim: dimensionality of data samples
            hidden_dim: width of hidden layers in velocity mlp (default 256)
            n_epochs: number of training epochs (default 1000)
            batch_size: training batch size (default 512)
            lr: adam learning rate (default 2e-3)
            score_weight: weight on score matching loss (default 1.0)
            eps: time clamping to [eps, 1-eps] (default 0.01)
            device: target device ('cuda' or 'cpu'); auto-detects if None
            integration_steps: number of ode integration steps (default 10000)
            div_method: divergence estimation method (default 'hutch_rademacher'); one of {'exact', 'hutch_gaussian', 'hutch_rademacher'}
            verbose: print training progress (default False)
            log_every: epoch interval for verbose logging (default 100)
            n_hidden_layers: number of hidden layers in velocity mlp (default 3)
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

        if verbose:
            warnings.warn(
                "verbose=True is deprecated in FMDRE; the new train_score_flow trainer "
                "does not support per-epoch logging. To monitor training, inspect the "
                "model's output directly or add a custom callback.",
                DeprecationWarning,
                stacklevel=2,
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None

    def init_model(self) -> None:
        """instantiate velocity mlp on the device."""
        self.model = CondVelScoreMLP(self.input_dim, self.hidden_dim, n_hidden_layers=self.n_hidden_layers).to(self.device)

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """train the conditional velocity field on samples from p0 and p1.

        procedure:
          1. call init_model() to instantiate the model
          2. convert inputs to float32
          3. create adam optimizer with self.lr
          4. define time_sampler lambda that samples tau uniformly in [eps, 1-eps]
          5. call train_score_flow with flow_matching_loss and hyperparameters
          6. set model to eval mode

        args:
            samples_p0: [N0, D] samples from source distribution
            samples_p1: [N1, D] samples from target distribution

        returns:
            None
        """
        self.init_model()

        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        def time_sampler(B, eps, dev):
            """sample tau uniformly in [eps, 1-eps] with unit importance weights."""
            tau = torch.rand(B, 1, device=dev) * (1.0 - 2 * eps) + eps
            iw = torch.ones(B, 1, device=dev)
            return tau, iw

        train_score_flow(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_fn=flow_matching_loss,
            optim=optim,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            eps=self.eps,
            loss_kwargs={
                "score_weight": self.score_weight,
                "p_uncond": 0.0,
                "sentinel_cond": -1.0,
            },
        )

        self.model.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """estimate log p0(x) / p1(x) via ratio ODE integration over [eps, 1-eps].

        procedure:
          1. validate model is fitted (raise RuntimeError if not)
          2. set model to eval mode
          3. move samples to device and convert to float32
          4. call ratio_ode with hyperparameters
          5. return result on CPU, detached

        args:
            xs: [B, D] query points

        returns:
            [B] log density ratio tensor on CPU
        """
        if self.model is None:
            raise RuntimeError("FMDRE model is not trained. Call fit() before predict_ldr().")

        self.model.eval()

        samples = xs.float().to(self.device)

        ldr = ratio_ode(
            self.model,
            samples,
            steps=self.integration_steps,
            eps=self.eps,
            device=str(self.device),
            div_method=self.div_method,
        )

        return ldr.detach().cpu()
