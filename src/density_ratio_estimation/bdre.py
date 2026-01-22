from typing import Callable

import torch

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.utils.binary_classifier import BinaryClassifier, build_default_binary_classifier


class BDRE(DensityRatioEstimator):
    def __init__(self, input_dim: int, classifier_builder: Callable[[], BinaryClassifier] = build_default_binary_classifier):
        self.classifier = classifier_builder(input_dim)

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        xs = torch.cat([samples_p0, samples_p1], dim=0)
        p0_labels = torch.zeros((samples_p0.shape[0], 1), dtype=torch.float)
        p1_labels = torch.ones((samples_p1.shape[0], 1), dtype=torch.float)
        ys = torch.cat([p0_labels, p1_labels], dim=0)
        self.classifier.fit(xs, ys)

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        return self.classifier.predict_logits(xs)