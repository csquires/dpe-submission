from typing import Callable

import torch

from kl_esimtation.base import KLEstimator
from src.utils.regression_model import RegressionModel, build_default_regression_model


class BDRE(KLEstimator):
    def __init__(self, input_dim: int, regressor_model_builder: Callable[[], RegressionModel] = build_default_regression_model):
        self.regressor = classifier_builder(input_dim)

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        self.classifier.fit(samples_p0, samples_p1)

    def predict(self, xs: torch.Tensor) -> torch.Tensor:
        pass