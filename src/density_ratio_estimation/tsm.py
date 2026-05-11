"""Time Score Matching (TSM) density ratio estimator."""

from typing import Optional
import warnings

import torch
import torch.optim as optim

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._trainer import train_loop
from src.density_ratio_estimation._ema import sample_time_and_iw
from src.density_ratio_estimation._losses import tsm_loss
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D


class TSM(DensityRatioEstimator):
    """time-score-matching DRE: trains s_phi(x, tau) under `tsm_loss`, integrates -score over tau."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1e-3,
        reweight: bool = False,
        eps: float = 1e-5,
        device: Optional[str] = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        n_hidden_layers: int = 3,
        activation: str = "silu",
        integration_steps: int = 200,
    ) -> None:
        """rtol/atol are accepted for HPO compatibility but ignored (warning emitted)."""
        super().__init__(input_dim)
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )

        # emit DeprecationWarning for rtol/atol (no longer used in torch.trapezoid path)
        if rtol != 1e-6:
            warnings.warn(
                "rtol is deprecated; torch.trapezoid does not use ODE tolerances. "
                "Parameter is accepted but ignored. Use integration_steps to control "
                "quadrature precision.",
                DeprecationWarning,
                stacklevel=2,
            )
        if atol != 1e-6:
            warnings.warn(
                "atol is deprecated; torch.trapezoid does not use ODE tolerances. "
                "Parameter is accepted but ignored. Use integration_steps to control "
                "quadrature precision.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.reweight = reweight
        self.eps = eps
        self.rtol = rtol
        self.atol = atol
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.integration_steps = integration_steps

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.optimizer = None

    def init_model(self) -> None:
        """instantiate TimeScoreNetwork1D and Adam optimizer."""
        self.model = TimeScoreNetwork1D(
            self.input_dim,
            self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """init model + optimizer, then delegate to `train_loop` with `tsm_loss`."""
        self.init_model()
        train_loop(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_fn=tsm_loss,
            optim=self.optimizer,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=lambda B, e, d: sample_time_and_iw("uniform", B, e, d),
            eps=self.eps,
            loss_kwargs={"reweight": self.reweight, "eps": self.eps},
        )

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """trapezoid-integrate -model(xs, tau) over tau in [eps, 1] and return on CPU."""
        if self.model is None:
            raise RuntimeError("TSM is not trained. Call fit() before predict_ldr().")

        self.model.eval()
        xs = xs.float().to(self.device)
        with torch.no_grad():
            ts = torch.linspace(self.eps, 1.0, self.integration_steps, device=self.device)
            vals = torch.stack(
                [
                    -self.model(xs, torch.full((xs.shape[0], 1), t.item(), device=self.device)).squeeze(-1)
                    for t in ts
                ],
                dim=0,
            )
            dt = (1.0 - self.eps) / (self.integration_steps - 1)
            return torch.trapezoid(vals, dx=dt, dim=0).cpu()
