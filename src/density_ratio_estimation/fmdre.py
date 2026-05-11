"""flow-matching DRE: trains a conditional velocity field then integrates a ratio ODE."""

from typing import Optional
import warnings
import torch

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._losses import fm_loss
from src.density_ratio_estimation._trainer import train_loop
from src.models.flow.cond_vel_score_mlp import CondVelScoreMLP
from src.models.flow.ratio_ode import ratio_ode


class FMDRE(DensityRatioEstimator):
    """flow-matching DRE under `fm_loss`; predict_ldr runs the ratio ODE."""

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
        """verbose is accepted but deprecated (the new trainer has no per-epoch logging)."""
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
                "verbose=True is deprecated in FMDRE; train_loop does not log per-epoch.",
                DeprecationWarning,
                stacklevel=2,
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None

    def init_model(self) -> None:
        """instantiate the conditional velocity MLP on self.device."""
        self.model = CondVelScoreMLP(self.input_dim, self.hidden_dim, n_hidden_layers=self.n_hidden_layers).to(self.device)

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """init model, then delegate to `train_loop` with `fm_loss` (no CFG)."""
        self.init_model()
        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        def time_sampler(B, eps, dev):
            return torch.rand(B, 1, device=dev) * (1.0 - 2 * eps) + eps, torch.ones(B, 1, device=dev)

        train_loop(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_fn=fm_loss,
            optim=opt,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            eps=self.eps,
            loss_kwargs={"score_weight": self.score_weight, "p_uncond": 0.0, "sentinel_cond": -1.0},
        )
        self.model.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """run the ratio ODE on xs and return log(p0/p1) on CPU, detached."""
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
