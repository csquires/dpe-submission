from abc import ABC, abstractmethod

import torch


class DRE(ABC):
    """
    Abstract base class for density ratio estimation (DRE).

    Estimates the log-density-ratio log(p0/p1) from two sample populations.

    Attributes:
        input_dim (int): dimensionality of the input space.
    """

    def __init__(self, input_dim: int) -> None:
        """
        Initialize DRE estimator.

        Args:
            input_dim: dimensionality of input space.
        """
        self.input_dim = input_dim

    @abstractmethod
    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """
        Fit density ratio estimator to samples from two distributions.

        Args:
            samples_p0: shape (n_p0, input_dim), samples from first distribution (numerator).
            samples_p1: shape (n_p1, input_dim), samples from second distribution (denominator).

        Returns:
            None.
        """
        pass

    @abstractmethod
    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict log-density-ratio at given points.

        Args:
            xs: shape (n, input_dim), evaluation points.

        Returns:
            shape (n,), log-density-ratio = log(p0(xs) / p1(xs)) at each point in xs.

        Note:
            Device convention (CPU vs GPU) is NOT standardized at this layer.
            Implementations may vary in their device handling.
        """
        pass


class ELDR(ABC):
    """
    Abstract base class for extended log-density-ratio estimation (ELDR).

    Estimates the log-density-ratio log(p0/p1) using a reference distribution p*.
    This allows for more flexible estimation with access to an intermediate distribution.

    Attributes:
        input_dim (int): dimensionality of the input space.
    """

    def __init__(self, input_dim: int) -> None:
        """
        Initialize ELDR estimator.

        Args:
            input_dim: dimensionality of input space.
        """
        self.input_dim = input_dim

    @abstractmethod
    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor
    ) -> None:
        """
        Fit extended density ratio estimator to samples from three distributions.

        Args:
            samples_p0: shape (n_p0, input_dim), samples from first distribution (numerator).
            samples_p1: shape (n_p1, input_dim), samples from second distribution (denominator).
            samples_pstar: shape (n_pstar, input_dim), samples from reference distribution.

        Returns:
            None.
        """
        pass

    @abstractmethod
    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict log-density-ratio at given points.

        Args:
            xs: shape (n, input_dim), evaluation points.

        Returns:
            shape (n,), log-density-ratio = log(p0(xs) / p1(xs)) at each point in xs.

        Note:
            Device convention (CPU vs GPU) is NOT standardized at this layer.
            Implementations may vary in their device handling.
        """
        pass


# Backward compatibility: DensityRatioEstimator is an alias for DRE.
# This will be deprecated in a future release.
DensityRatioEstimator = DRE