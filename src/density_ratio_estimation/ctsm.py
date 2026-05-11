from typing import Optional

import torch
import torch.optim as optim

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._ema import EMA
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D


class CTSM(DensityRatioEstimator):
    """
    Conditional Time Score Matching for density ratio estimation.

    Uses Schrodinger Bridge path with closed-form conditional time score target.
    Training: delegates to train_score_flow with closed_form_sb_loss (path=None).
    Inference: ODE integration from t=eps to t=1-eps via torch.trapezoid.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1e-3,
        sigma: float = 1.0,
        eps: float = 1e-3,
        device: Optional[str] = None,
        integration_steps: int = 200,
        n_hidden_layers: int = 3,
        ema_decay: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        time_dist: str = "uniform",
        activation: str = "elu",
    ) -> None:
        """
        Initialize CTSM estimator.

        Args mostly inherited from base. ema_decay (default None) enables EMA
        of model parameters for inference; if set, must be in (0, 1).
        grad_clip_norm (default None) clips gradient norm at the given value
        before each optimizer step; None disables clipping.
        time_dist: importance sampling time distribution. in {"uniform", "beta_2_2",
        "beta_5_5"}; default "uniform" preserves current behavior.
        activation: score network activation function {"elu", "gelu", "silu"};
        default "elu" preserves byte-identical behavior.
        """
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.sigma = sigma
        self.eps = eps
        self.integration_steps = integration_steps
        self.n_hidden_layers = n_hidden_layers
        self.ema_decay = ema_decay
        self.grad_clip_norm = grad_clip_norm
        if time_dist not in {"uniform", "beta_2_2", "beta_5_5"}:
            raise ValueError(
                f"time_dist must be in {{'uniform', 'beta_2_2', 'beta_5_5'}}; "
                f"got {time_dist!r}"
            )
        self.time_dist = time_dist
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )
        self.activation = activation
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        self.ema: Optional[EMA] = None

    def init_model(self) -> None:
        """
        Initialize neural network and optimizer.

        Instantiate TimeScoreNetwork1D on device, create Adam optimizer, and
        optionally wrap in EMA if ema_decay is set.
        """
        self.model = TimeScoreNetwork1D(
            self.input_dim, self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )
        self.ema = EMA(self.model, self.ema_decay) if self.ema_decay is not None else None

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """
        Train CTSM on paired samples from p0 and p1.

        Initializes model and delegates training to train_score_flow with
        closed_form_sb_loss (path=None). Loss computes SB path via sb_bridge_target.

        Procedure:
          1. Call init_model() to instantiate TimeScoreNetwork1D, optimizer, EMA.
          2. Define time_sampler as a lambda wrapping sample_time_and_iw.
          3. Call train_score_flow(
               model, samples_p0, samples_p1, samples_pstar=None,
               loss_fn=closed_form_sb_loss,
               optim, n_steps=self.n_epochs, batch_size=self.batch_size,
               time_sampler, ema=self.ema, grad_clip_norm=self.grad_clip_norm,
               eps=self.eps, loss_kwargs={"sigma": self.sigma, "path": None}
             ).

        Args:
            samples_p0: [N0, D] samples from p0. Cast to float and moved to device.
            samples_p1: [N1, D] samples from p1. Cast to float and moved to device.

        Returns: None. After fit(), model is in .eval() mode.
        """
        from src.density_ratio_estimation._trainer import train_score_flow
        from src.density_ratio_estimation._losses import closed_form_sb_loss
        from src.density_ratio_estimation._ema import sample_time_and_iw

        self.init_model()

        train_score_flow(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=None,
            loss_fn=closed_form_sb_loss,
            optim=self.optimizer,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=lambda B, eps, dev: sample_time_and_iw(
                self.time_dist, B, eps, dev
            ),
            ema=self.ema,
            grad_clip_norm=self.grad_clip_norm,
            eps=self.eps,
            loss_kwargs={"sigma": self.sigma, "path": None},
        )

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """Estimate log density ratio via trapezoidal quadrature.

        Procedure:
            - verify model is fitted, set eval mode, move xs to device.
            - build uniform tau grid of self.integration_steps points in [eps, 1-eps].
            - evaluate -score(xs, tau) at each grid point (batched over xs).
            - return torch.trapezoid along the tau axis.

        Returns:
            log density ratios as a 1D CPU tensor [N].
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        xs = xs.to(self.device)
        n = xs.shape[0]

        # if EMA is active, evaluate with the shadow weights and restore on exit
        if self.ema is not None:
            self.ema.apply_to(self.model)
        try:
            ts = torch.linspace(self.eps, 1.0 - self.eps, self.integration_steps, device=self.device)
            with torch.no_grad():
                vals = torch.stack([
                    -self.model(xs, torch.full((n, 1), float(t.item()), device=self.device)).squeeze(-1)
                    for t in ts
                ])
            dt = (1.0 - 2.0 * self.eps) / (self.integration_steps - 1)
            return torch.trapezoid(vals, dx=dt, dim=0).cpu()
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
