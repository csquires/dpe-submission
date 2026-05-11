"""callable time samplers for the `train_loop` trainer.

each sampler matches the protocol
  Callable[[int, float, torch.device], tuple[Tensor, Tensor]]
returning (tau [B,1], iw [B,1]).
"""
import torch

from src.density_ratio_estimation._ema import sample_time_and_iw


class UniformSampler:
    """tau ~ U(eps, 1-eps), iw = 1."""

    def __init__(self, eps: float = 1e-3) -> None:
        self.eps = eps

    def __call__(
        self, batch_size: int, eps: float, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return sample_time_and_iw("uniform", batch_size, eps, device)


class BetaSampler:
    """importance-sampled tau via Beta(2,2) or Beta(5,5)."""

    def __init__(self, dist: str = "beta_2_2") -> None:
        if dist not in ("beta_2_2", "beta_5_5"):
            raise ValueError(f"dist must be 'beta_2_2' or 'beta_5_5'; got {dist!r}")
        self.dist = dist

    def __call__(
        self, batch_size: int, eps: float, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return sample_time_and_iw(self.dist, batch_size, eps, device)
