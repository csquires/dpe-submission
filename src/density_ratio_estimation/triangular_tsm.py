"""TriangularTSM: time-score matching DRE on a bell-shaped path."""
import warnings
from typing import Optional, Tuple
from math import ceil

import torch
import torch.optim as optim

from src.density_ratio_estimation.base import ELDR
from src.density_ratio_estimation._trainer import train_loop
from src.density_ratio_estimation._losses import tri_tsm_loss
from src.models.time_score_matching.time_score_net_2d import TimeScoreNetwork2D


class TriangularTSM(ELDR):
    """time-score matching DRE under `tri_tsm_loss` on a piecewise-quadratic bell path.

    bell: t' = peak_max (1 - ((tau - vertex)/scale)^2) on (0, vertex) and (vertex, 1).
    integrates -score over a tau grid at inference.
    """

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
        vertex: float = 0.5,
        peak_max: float = 1.0,
        n_hidden_layers: int = 3,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        cosine_min_factor: float = 1.0,
        activation: str = "silu",
    ) -> None:
        """vertex / peak_max define the bell path; cosine_min_factor=1.0 disables LR annealing;
        rtol/atol kept for HPO compatibility but ignored (warning emitted)."""
        super().__init__(input_dim)

        # validate params
        if not 0.0 < vertex < 1.0:
            raise ValueError(f"vertex must be in (0, 1), got {vertex}")
        if not 0.0 < peak_max <= 1.0:
            raise ValueError(f"peak_max must be in (0, 1], got {peak_max}")
        if not 0.0 <= cosine_min_factor <= 1.0:
            raise ValueError(f"cosine_min_factor must be in [0, 1], got {cosine_min_factor}")
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(f"activation must be in {{'elu', 'gelu', 'silu'}}, got {activation!r}")

        # emit deprecation warning for rtol/atol
        if rtol != 1e-6 or atol != 1e-6:
            warnings.warn(
                "rtol and atol are deprecated; torch.trapezoid integration does not use them.",
                DeprecationWarning,
                stacklevel=2,
            )

        # store all hyperparameters
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.reweight = reweight
        self.eps = eps
        self.rtol = rtol
        self.atol = atol
        self.vertex = vertex
        self.peak_max = peak_max
        self.n_hidden_layers = n_hidden_layers
        self.adam_betas = tuple(adam_betas)
        self.weight_decay = float(weight_decay)
        self.cosine_min_factor = float(cosine_min_factor)
        self.activation = activation

        # device handling
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # model and optimizer (lazy-initialized in _init_model)
        self.model = None
        self.optimizer = None

    def _init_model(self) -> None:
        """instantiate TimeScoreNetwork2D and Adam optimizer."""
        self.model = TimeScoreNetwork2D(
            self.input_dim,
            self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=self.adam_betas,
            eps=1e-8,
            weight_decay=self.weight_decay,
        )

    def _path_t_tprime(self, tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """piecewise-quadratic bell at tau: t' is 0 at endpoints and peak_max at vertex."""
        t = torch.clamp(tau, min=self.eps, max=1.0)
        v, m = self.vertex, self.peak_max
        left = m * (2.0 * (tau / v) - (tau / v) ** 2)
        right = m * (1.0 - ((tau - v) / (1.0 - v)) ** 2)
        t_prime = torch.clamp(torch.where(tau <= v, left, right), min=0.0, max=1.0)
        return t, t_prime

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """init model, optional cosine scheduler, then delegate to `train_loop` with `tri_tsm_loss`."""
        self._init_model()
        self.model.train()

        scheduler = None
        if self.cosine_min_factor < 1.0:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.n_epochs,
                eta_min=self.lr * self.cosine_min_factor,
            )

        def time_sampler(B: int, eps: float, device: torch.device):
            tau = eps + torch.rand(B, 1, device=device) * (1.0 - 2.0 * eps)
            return tau, torch.ones(B, 1, device=device)

        min_size = min(len(samples_p0), len(samples_p1), len(samples_pstar))
        n_steps = self.n_epochs * ceil(min_size / self.batch_size)

        train_loop(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_fn=tri_tsm_loss,
            optim=self.optimizer,
            n_steps=n_steps,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler=scheduler,
            eps=self.eps,
            loss_kwargs={
                "reweight": self.reweight,
                "eps": self.eps,
                "vertex": self.vertex,
                "peak_max": self.peak_max,
            },
        )

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """trapezoid-integrate -model(xs, t, t') over a 100-point tau grid in [eps, 1]."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() before predict_ldr().")

        self.model.eval()
        samples = xs.float().to(self.device)
        n = samples.shape[0]
        if n == 0:
            return torch.zeros(0, dtype=samples.dtype, device=self.device)

        with torch.no_grad():
            tau_grid = torch.linspace(self.eps, 1.0, 100, device=self.device)
            scores = []
            for tau_scalar in tau_grid:
                t, t_prime = self._path_t_tprime(tau_scalar.view(1, 1))
                score = self.model(samples, t.expand(n, 1), t_prime.expand(n, 1))
                scores.append(-score.squeeze(-1))
            return torch.trapezoid(torch.stack(scores, dim=0).t(), tau_grid, dim=1)
