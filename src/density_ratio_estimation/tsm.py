from typing import Callable

import torch

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.binary_classification.binary_classifier import BinaryClassifier
from src.models.binary_classification.default_binary_classifier import build_default_binary_classifier


class TSM(DensityRatioEstimator):
    def __init__(self, input_dim: int, classifier_builder: Callable[[], BinaryClassifier] = build_default_binary_classifier):
        pass

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        pass

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        pass