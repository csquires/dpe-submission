"""TriangularTSM: 3-source density ratio estimation via score matching on bell path."""
import warnings
from typing import Optional, Tuple
from math import ceil

import torch
import torch.optim as optim

from src.density_ratio_estimation.base import ELDR
from src.density_ratio_estimation._trainer import train_score_flow
from src.density_ratio_estimation._losses import triangular_hyvarinen_time_score_loss
from src.models.time_score_matching.time_score_net_2d import TimeScoreNetwork2D


class TriangularTSM(ELDR):
    """3-source density ratio estimation via score matching on piecewise-quadratic bell path.

    Uses train_score_flow unified trainer and triangular_hyvarinen_time_score_loss.
    The path is a bell interpolant: t' = peak_max * (1 - (|tau - vertex| / scale)^2)
    applied piecewise on (0, vertex) and (vertex, 1).

    Procedure:
      - Constructor: validate params, initialize model=None, optimizer=None.
      - fit(): call _init_model, create scheduler (optional), call train_score_flow.
      - predict_ldr(): integrate -score via torch.trapezoid over tau_grid.
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
        """Initialize TriangularTSM.

        Args:
            input_dim: dimension of input space.
            hidden_dim: hidden layer size in score network.
            n_epochs: number of training epochs.
            batch_size: batch size for training.
            lr: learning rate.
            reweight: if True, apply time-dependent weighting in loss.
            eps: time-domain margin (tau in [eps, 1-eps]).
            device: device (cuda/cpu). None defaults to cuda if available.
            rtol: deprecated (scipy.integrate.solve_ivp param, no longer used).
            atol: deprecated (scipy.integrate.solve_ivp param, no longer used).
            vertex: peak location of bell path in (0, 1).
            peak_max: peak height of bell in (0, 1].
            n_hidden_layers: number of hidden layers in score network.
            adam_betas: Adam beta parameters.
            weight_decay: weight decay in optimizer.
            cosine_min_factor: cosine annealing min factor in [0, 1]; 1.0 disables scheduling.
            activation: activation function (elu, gelu, silu).
        """
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
        """Initialize score network and optimizer.

        Procedure:
          - Create TimeScoreNetwork2D with stored hyperparams.
          - Move to device.
          - Create Adam optimizer.
        """
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
        """Compute (t, t_prime) for bell path at tau.

        Piecewise-quadratic bell: peak at (vertex, peak_max).
        t' = 0 at tau=0 and tau=1, peak_max at tau=vertex.

        Args:
            tau: [B, 1] or broadcastable time parameter.

        Returns:
            t: [same shape as tau] clamped to [eps, 1.0].
            t_prime: [same shape as tau] bell path, clamped to [0, 1].
        """
        # clamp tau to valid range
        t = torch.clamp(tau, min=self.eps, max=1.0)  # [B, 1]

        # piecewise bell: left (0 <= tau <= vertex), right (vertex < tau <= 1)
        v = self.vertex
        m = self.peak_max

        left = m * (2.0 * (tau / v) - (tau / v) ** 2)  # [B, 1]
        right = m * (1.0 - ((tau - v) / (1.0 - v)) ** 2)  # [B, 1]

        # select piece
        t_prime = torch.where(tau <= v, left, right)  # [B, 1]

        # clamp to [0, 1]
        t_prime = torch.clamp(t_prime, min=0.0, max=1.0)  # [B, 1]

        return t, t_prime

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """Fit score network via train_score_flow.

        Procedure:
          1. Initialize model and optimizer.
          2. Create cosine scheduler (if cosine_min_factor < 1.0).
          3. Define time_sampler.
          4. Call train_score_flow with triangular_hyvarinen_time_score_loss.

        Args:
            samples_p0: [N0, D] samples from p0.
            samples_p1: [N1, D] samples from p1.
            samples_pstar: [Nstar, D] samples from p*.
        """
        # initialize model and optimizer
        self._init_model()

        # set model to train mode
        self.model.train()

        # create scheduler if cosine annealing enabled
        scheduler = None
        if self.cosine_min_factor < 1.0:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.n_epochs,
                eta_min=self.lr * self.cosine_min_factor,
            )

        # define time_sampler: uniform tau in (eps, 1-eps), constant iw
        def time_sampler(batch_size: int, eps: float, device: torch.device):
            tau = eps + torch.rand(batch_size, 1, device=device) * (1.0 - 2.0 * eps)  # [B, 1]
            iw = torch.ones(batch_size, 1, device=device)  # [B, 1]
            return tau, iw

        # compute n_steps based on epoch count and dataset size
        min_size = min(len(samples_p0), len(samples_p1), len(samples_pstar))
        n_steps = self.n_epochs * ceil(min_size / self.batch_size)

        # define loss_kwargs for triangular_hyvarinen_time_score_loss
        loss_kwargs = {
            "reweight": self.reweight,
            "eps": self.eps,
            "vertex": self.vertex,
            "peak_max": self.peak_max,
        }

        # call unified trainer
        train_score_flow(
            model=self.model,
            samples_p0=samples_p0,
            samples_p1=samples_p1,
            samples_pstar=samples_pstar,
            loss_fn=triangular_hyvarinen_time_score_loss,
            optim=self.optimizer,
            n_steps=n_steps,
            batch_size=self.batch_size,
            time_sampler=time_sampler,
            scheduler=scheduler,
            ema=None,
            grad_clip_norm=None,
            eps=self.eps,
            loss_kwargs=loss_kwargs,
        )

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """Estimate log density ratio via tau-integration.

        Numerically integrates -score over tau from eps to 1.0 using torch.trapezoid.
        The score is evaluated on the (t, t_prime) bell path at each tau.

        Procedure:
          1. Validate model is trained.
          2. Set eval mode, move samples to device.
          3. Create tau_grid (e.g., 100 points).
          4. For each sample, integrate -score(x, tau).
          5. Return log-ratios [N].

        Args:
            xs: [N, D] evaluation points.

        Returns:
            torch.Tensor: [N] log-density-ratio estimates.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() before predict_ldr().")

        self.model.eval()

        # move to device and ensure float
        samples = xs.float().to(self.device)  # [N, D]
        n_samples = samples.shape[0]

        # handle empty input
        if n_samples == 0:
            return torch.zeros(0, dtype=samples.dtype, device=self.device)

        with torch.no_grad():
            # create tau grid: 100 points from eps to 1.0
            tau_grid = torch.linspace(self.eps, 1.0, 100, device=self.device)  # [T]

            # evaluate -score at each tau
            scores_at_tau = []
            for tau_scalar in tau_grid:
                tau_single = tau_scalar.unsqueeze(0).unsqueeze(1)  # [1, 1]
                t, t_prime = self._path_t_tprime(tau_single)  # both [1, 1]

                # broadcast to batch
                t_batch = t.expand(n_samples, 1)  # [N, 1]
                t_prime_batch = t_prime.expand(n_samples, 1)  # [N, 1]

                # evaluate model [N, 1]
                score = self.model(samples, t_batch, t_prime_batch)
                scores_at_tau.append((-score).squeeze(-1))  # [N]

            # stack: [T, N]
            scores_at_tau = torch.stack(scores_at_tau, dim=0)  # [T, N]

            # integrate via trapezoid: trapezoid(f, x) assumes f shape [..., n] and x shape [n]
            # so we transpose to [N, T] and integrate over dim 1
            scores_at_tau = scores_at_tau.t()  # [N, T]
            log_ratios = torch.trapezoid(scores_at_tau, tau_grid, dim=1)  # [N]

        return log_ratios
