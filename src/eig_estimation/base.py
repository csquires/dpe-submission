import abc

import torch


class EIGEstimation(abc.ABC):
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    @abc.abstractmethod
    def estimate_eig(
        self, 
        samples_theta: torch.Tensor, 
        samples_y: torch.Tensor, 
    ) -> float:
        pass