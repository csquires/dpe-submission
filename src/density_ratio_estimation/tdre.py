from typing import Callable

import torch

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.utils.binary_classifier import BinaryClassifier, build_default_binary_classifier


class TDRE(DensityRatioEstimator):
    def __init__(self, input_dim: int, classifier_builder: Callable[[], BinaryClassifier] = build_default_binary_classifier):
        # self.classifier = classifier_builder(input_dim)
        pass

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        pass
        # self.classifier.fit(samples_p0, samples_p1)

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(xs.shape[0])