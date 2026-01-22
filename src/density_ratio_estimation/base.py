import abc

import numpy as np


class BaseDensityRatioEstimator(abc.ABC):
    @abc.abstractmethod
    def fit(
        self, 
        samples_p0: np.ndarray, 
        samples_p1: np.ndarray
    ) -> float:
        pass

    def predict(
        self,
        xs: np.ndarray,
    ) -> np.ndarray:
        pass