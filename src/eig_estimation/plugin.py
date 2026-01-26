import torch

from src.eig_estimation.base import EIGEstimation
from src.density_ratio_estimation import BDRE, MDRE, TDRE, TSM



class EIGPlugin(EIGEstimation):
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

    def _create_marginal_samples(self, samples_theta: torch.Tensor, samples_y: torch.Tensor) -> torch.Tensor:
        shuffled_thetas = samples_theta[torch.randperm(samples_theta.shape[0])]
        shuffled_ys = samples_y[torch.randperm(samples_y.shape[0])]
        return torch.cat([shuffled_thetas, shuffled_ys], dim=1)

    def estimate_eig(
        self, 
        samples_theta: torch.Tensor, 
        samples_y: torch.Tensor, 
    ) -> float:
        # p0 samples are from joint distribution and p1 samples are from the product of marginal distributions
        samples_p0 = torch.cat([samples_theta, samples_y], dim=1)
        samples_p1 = self._create_marginal_samples(samples_theta, samples_y)
        # fit density ratio estimator
        self.dre.fit(samples_p0, samples_p1)
        # predict density ratio
        est_ldrs = self.dre.predict_ldr(samples_p0)
        # return estimated eig
        return torch.mean(est_ldrs)