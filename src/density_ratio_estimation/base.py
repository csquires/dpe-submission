import abc

import torch


class DensityRatioEstimator(abc.ABC):
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    @abc.abstractmethod
    def fit(
        self, 
        samples_p0: torch.Tensor, 
        samples_p1: torch.Tensor
    ) -> float:
        pass

    def predict(
        self,
        xs: torch.Tensor,
    ) -> torch.Tensor:
        pass