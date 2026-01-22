import abc

import numpy as np


class BaseKLEstimator(abc.ABC):
    @abc.abstractmethod
    def estimate_kl(self, samples_p0: np.ndarray, samples_p1: np.ndarray) -> float:
        pass