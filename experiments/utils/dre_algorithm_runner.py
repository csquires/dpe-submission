from tqdm import tqdm
import torch

from src.density_ratio_estimation.base import DensityRatioEstimator


class DREAlgorithmRunner:
    def run(
        self, 
        samples_p0: torch.Tensor, 
        samples_p1: torch.Tensor,
        all_test_samples: list[torch.Tensor],
        algorithm: DensityRatioEstimator
    ) -> torch.Tensor:
        results = []
        algorithm.fit(samples_p0, samples_p1)
        for test_samples in all_test_samples:
            result = algorithm.predict(test_samples)
            results.append(result)
        return results