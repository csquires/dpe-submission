"""TriangularCTSM: continuous-time score matching DRE on a barycentric path."""
from typing import Callable, Optional

import torch
import torch.optim as optim

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._ema import EMA, sample_time_and_iw
from src.density_ratio_estimation._trainer import train_loop
from src.density_ratio_estimation._losses import sb_loss
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D
from src.waypoints.path_1d import CtsmPath1D
from src.waypoints.triangular_continuous import BarycentricCtsm1D
from src.waypoints.piecewise_sb import PiecewiseSBCtsm1D


class TriangularCTSM(DensityRatioEstimator):
    """CTSM with barycentric path p0 -> p* -> p1.

    trains s_phi(x, tau) under `sb_loss(path=self.path)`; integrates -score over
    tau in [eps, 1-eps] at inference.
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
        """path defaults to BarycentricCtsm1D(sigma=1.0, vertex=0.5, eps=eps); if path
        exposes sample_tau, it overrides time_dist at training time."""
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
        """instantiate TimeScoreNetwork1D, Adam optimizer, and optional EMA."""
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
        """init model, then delegate to `train_loop` with `sb_loss(path=self.path)`.

        when self.path exposes sample_tau, the loss bypasses time_sampler internally
        (see `sb_loss`); otherwise time_dist is honored.
        """
        self.init_model()

        def time_sampler(B: int, eps: float, device) -> tuple[torch.Tensor, torch.Tensor]:
            sampler = getattr(self.path, "sample_tau", None)
            if callable(sampler):
                return sampler(B, eps, device), torch.ones(B, 1, device=device)
            return sample_time_and_iw(self.time_dist, B, eps, device)

        train_loop(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_fn=sb_loss,
            optim=self.optimizer,
            n_steps=self.n_epochs,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            ema=self.ema,
            grad_clip_norm=self.grad_clip_norm,
            eps=self.eps,
            loss_kwargs={"sigma": 1.0, "path": self.path},
        )
        self.model.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """trapezoid-integrate -model(xs, tau) over tau in [eps, 1-eps]; uses EMA shadow if set."""
        if self.model is None:
            raise RuntimeError("TriangularCTSM is not trained. Call fit() before predict_ldr().")

        self.model.eval()
        samples = xs.to(self.device).float()
        n = samples.shape[0]

        if self.ema is not None:
            self.ema.apply_to(self.model)
        try:
            ts = torch.linspace(self.eps, 1.0 - self.eps, self.integration_steps, device=self.device)
            with torch.no_grad():
                vals = torch.stack([
                    -self.model(samples, torch.full((n, 1), float(t.item()), device=self.device, dtype=torch.float32)).squeeze(-1)
                    for t in ts
                ])
            dt = (1.0 - 2.0 * self.eps) / (self.integration_steps - 1)
            return torch.trapezoid(vals, dx=dt, dim=0).cpu()
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
