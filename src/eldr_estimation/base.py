import abc

import numpy as np


class BaseELDREstimator(abc.ABC):
    @abc.abstractmethod
    def estimate_eldr(
        self, 
        samples_base: np.ndarray, 
        samples_p0: np.ndarray, 
        samples_p1: np.ndarray
    ) -> float:
        pass