import abc

import numpy as np


class ELDREstimator(abc.ABC):
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    @abc.abstractmethod
    def estimate_eldr(
        self, 
        samples_pstar: np.ndarray, 
        samples_p0: np.ndarray, 
        samples_p1: np.ndarray
    ) -> float:
        pass