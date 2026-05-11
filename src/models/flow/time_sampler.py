"""Time-sampler helpers for the train_score_flow trainer.

Provides callable adapters that match the trainer's `time_sampler` contract:
`Callable[[int, float, torch.device], tuple[Tensor, Tensor]]` returning
(tau [B,1], iw [B,1]).
"""
import torch

from src.density_ratio_estimation._ema import sample_time_and_iw


class UniformTimeSampler:
    """callable wrapper around sample_time_and_iw with time_dist='uniform'.

    Matches the trainer's time_sampler protocol: sampler(batch_size, eps, device)
    returns (tau, iw) of shape ([B,1], [B,1]).
    """

    def __init__(self, eps: float = 1e-3) -> None:
        self.eps = eps

    def __call__(
        self, batch_size: int, eps: float, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # caller passes eps too; prefer the runtime arg for trainer-injected eps.
        return sample_time_and_iw("uniform", batch_size, eps, device)


class BetaTimeSampler:
    """callable wrapper for time_dist in {'beta_2_2', 'beta_5_5'}."""

    def __init__(self, time_dist: str = "beta_2_2") -> None:
        if time_dist not in ("beta_2_2", "beta_5_5"):
            raise ValueError(f"time_dist must be in {{'beta_2_2', 'beta_5_5'}}; got {time_dist!r}")
        self.time_dist = time_dist

    def __call__(
        self, batch_size: int, eps: float, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return sample_time_and_iw(self.time_dist, batch_size, eps, device)
