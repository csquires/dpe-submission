from abc import ABC, abstractmethod

import torch


class DRE(ABC):
    """abstract base for log-density-ratio estimators.

    Attributes:
        input_dim: dimensionality of the input space.
    """

    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim

    @abstractmethod
    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """fit to samples_p0 [N0, input_dim] and samples_p1 [N1, input_dim]."""
        pass

    @abstractmethod
    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """log(p0(xs) / p1(xs)); shape [N]. device handling is implementation-defined."""
        pass


class ELDR(ABC):
    """abstract base for triangular log-density-ratio estimators that take a reference p*."""

    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim

    @abstractmethod
    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """fit to samples_p0 [N0,D], samples_p1 [N1,D], samples_pstar [N*,D]."""
        pass

    @abstractmethod
    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """log(p0(xs) / p1(xs)); shape [N]. device handling is implementation-defined."""
        pass


DensityRatioEstimator = DRE
