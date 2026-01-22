from abc import ABC, abstractmethod

import torch


class DensityRatioEstimator(ABC):
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    @abstractmethod
    def fit(
        self, 
        samples_p0: torch.Tensor, 
        samples_p1: torch.Tensor
    ) -> None:
        pass

    @abstractmethod
    def predict_ldr(
        self,
        xs: torch.Tensor,
    ) -> torch.Tensor:
        pass