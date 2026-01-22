from typing import Callable

import torch

from kl_estimation.base import KLEstimator
from src.utils.regression_model import RegressionModel, build_default_regression_model


class DV_KLE(KLEstimator):
    def __init__(self, input_dim: int, regression_model_builder: Callable[[], RegressionModel] = build_default_regression_model):
        self.regression_model = regression_model_builder(input_dim)

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        self.regression_model.fit(samples_p0, samples_p1)

    def predict(self, xs: torch.Tensor) -> torch.Tensor:
        pass