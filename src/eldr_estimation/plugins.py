import torch

from src.eldr_estimation.base import ELDREstimator
from src.density_ratio_estimation import BDRE, MDRE, TDRE, TSM



class DREPlugin(ELDREstimator):
    def __init__(self, input_dim: int, method: str = "bdre"):
        self.input_dim = input_dim
        self.method = method
        if method == "bdre":
            self.dre = BDRE(input_dim)
        elif method == "mdre":
            self.dre = MDRE(input_dim)
        elif method == "tdre":
            self.dre = TDRE(input_dim)
        elif method == "tsm":
            self.dre = TSM(input_dim)
        else:
            raise ValueError(f"Invalid method: {method}")

    def estimate_eldr(
        self, 
        samples_pstar: torch.Tensor, 
        samples_p0: torch.Tensor, 
        samples_p1: torch.Tensor
    ) -> float:
        self.dre.fit(samples_p0, samples_p1)
        est_ldrs = self.dre.predict_ldr(samples_pstar)
        return torch.mean(est_ldrs)