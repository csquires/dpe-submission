import torch

from src.density_ratio_estimation.base import DensityRatioEstimator


class DREAlgorithmRunner:
    def __init__(self, algorithms: list[DensityRatioEstimator]):
        self.algorithms = algorithms

    def run(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> torch.Tensor:
        results = []
        for algorithm in self.algorithms:
            result = algorithm.fit(samples_p0, samples_p1)
            results.append(result)
        return results