"""
TriangularCTSM: Continuous-time score matching for triangular density ratio estimation.

V2 (barycentric continuous path via three anchor distributions).
"""
from typing import Callable, Optional

import torch
import torch.optim as optim

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._ema import EMA, sample_time_and_iw
from src.density_ratio_estimation._trainer import train_score_flow
from src.density_ratio_estimation._losses import closed_form_sb_loss
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D
from src.waypoints.path_1d import CtsmPath1D
from src.waypoints.triangular_continuous import BarycentricCtsm1D
from src.waypoints.piecewise_sb import PiecewiseSBCtsm1D


class TriangularCTSM(DensityRatioEstimator):
    """
    Triangular continuous-time score matching for density ratio estimation.

    Uses a continuous barycentric path through p0 -> p* -> p1 (flavor-A path).
    Trains a score network via MSE loss matching target scores from the path.
    Inference: ODE integration from tau=eps to 1-eps yields log(p0/p1).

    Contract: fit(samples_p0, samples_p1, samples_pstar) with three tensors [N, D].
    predict_ldr(xs) returns log density ratios as 1D CPU tensor [N].
    """

    def __init__(
        self,
        input_dim: int,
        path: Optional[CtsmPath1D] = None,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1e-3,
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
        Initialize TriangularCTSM.

        Args:
            input_dim: Input dimension D.
            path: CtsmPath1D instance. If None, instantiate BarycentricCtsm1D(sigma=1.0, vertex=0.5, eps=eps).
            hidden_dim: Hidden layer width for TimeScoreNetwork1D.
            n_epochs: Number of training epochs.
            batch_size: Batch size for stochastic gradient descent.
            lr: Adam learning rate.
            eps: Margin for tau sampling and quadrature bounds. tau in [eps, 1-eps].
            device: Device string ("cuda", "cpu", etc.). If None, auto-detect: cuda if available, else cpu.
            integration_steps: Number of tau quadrature points for predict_ldr (uniform grid).
            n_hidden_layers: Number of hidden layers for TimeScoreNetwork1D.
            time_dist: importance sampling time distribution. in {"uniform", "beta_2_2",
            "beta_5_5"}; default "uniform" preserves current behavior. note: if path has
            sample_tau method, it takes precedence over time_dist.
            activation: score network activation function {"elu", "gelu", "silu"};
            default "elu" preserves byte-identical behavior.
        """
        super().__init__(input_dim)

        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
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

        # device resolution
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # path resolution
        if path is None:
            self.path = BarycentricCtsm1D(sigma=1.0, vertex=0.5, eps=eps)
        else:
            self.path = path

        # model placeholders
        self.model = None
        self.optimizer = None
        self.ema: Optional[EMA] = None

    def init_model(self) -> None:
        """
        Initialize or reinitialize the score network and optimizer.

        Constructs TimeScoreNetwork1D(input_dim, hidden_dim) and moves to device.
        Creates Adam optimizer with standard betas and eps.
        """
        self.model = TimeScoreNetwork1D(self.input_dim, self.hidden_dim, n_hidden_layers=self.n_hidden_layers, activation=self.activation).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.ema = EMA(self.model, self.ema_decay) if self.ema_decay is not None else None

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """
        Train the score network on three-tensor contract.

        Args:
            samples_p0: Samples from p0, shape [N0, D].
            samples_p1: Samples from p1, shape [N1, D].
            samples_pstar: Samples from p* (anchor distribution), shape [Nstar, D].

        Procedure:
          1. Initialize model and optimizer via init_model().
          2. Delegate training to train_score_flow with:
             - model, samples, loss_fn=closed_form_sb_loss
             - loss_kwargs: {"sigma": 1.0, "path": self.path}
             - time_sampler: conditional on self.path.sample_tau vs time_dist
          3. Set model.eval() at completion.

        Notes:
          - V1 path (PiecewiseSBCtsm1D) with sample_tau method: closed_form_sb_loss
            will dynamically override (tau, iw) inside the loss.
          - V2 path (BarycentricCtsm1D) without sample_tau: trainer-provided time_sampler
            is used (respects self.time_dist kwarg).
          - EMA update is handled by train_score_flow if self.ema is not None.
          - Gradient clipping is applied via trainer if grad_clip_norm > 0.
        """
        self.init_model()

        # time sampler: respects path.sample_tau if it exists; otherwise time_dist
        def time_sampler_fn(batch_size: int, eps: float, device) -> tuple[torch.Tensor, torch.Tensor]:
            """Sample (tau, iw) respecting path override or falling back to time_dist."""
            sampler = getattr(self.path, "sample_tau", None)
            if callable(sampler):
                return sampler(batch_size, eps, device), torch.ones(batch_size, 1, device=device)
            else:
                return sample_time_and_iw(self.time_dist, batch_size, eps, device)

        train_score_flow(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_fn=closed_form_sb_loss,
            optim=self.optimizer,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler_fn,
            ema=self.ema,
            grad_clip_norm=self.grad_clip_norm,
            eps=self.eps,
            loss_kwargs={"sigma": 1.0, "path": self.path},
        )

        self.model.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """Predict log density ratios via trapezoidal quadrature.

        Args:
            xs: test samples, shape [N, D], on CPU or device (moved to self.device).

        Returns:
            log density ratios, shape [N], on CPU.

        Procedure:
            - eval mode, move samples to device.
            - uniform tau grid of self.integration_steps points in [eps, 1-eps].
            - evaluate -score(samples, tau) at each grid point (batched over samples).
            - return torch.trapezoid along the tau axis.
        """
        if self.model is None:
            raise RuntimeError(
                "TriangularCTSM is not trained. Call fit() before predict_ldr()."
            )

        self.model.eval()
        samples = xs.to(self.device).float()
        n = samples.shape[0]

        if self.ema is not None:
            self.ema.apply_to(self.model)
        try:
            ts = torch.linspace(self.eps, 1.0 - self.eps, self.integration_steps, device=self.device)
            with torch.no_grad():
                vals = torch.stack([
                    -self.model(
                        samples,
                        torch.full((n, 1), float(t.item()), device=self.device, dtype=torch.float32),
                    ).squeeze(-1)
                    for t in ts
                ])
            dt = (1.0 - 2.0 * self.eps) / (self.integration_steps - 1)
            return torch.trapezoid(vals, dx=dt, dim=0).cpu()
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
