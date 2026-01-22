import abc

import torch


class KLEstimator(abc.ABC):
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    @abc.abstractmethod
    def estimate_kl(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> float:
        pass