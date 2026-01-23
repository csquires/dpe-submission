from typing import Callable

import torch

from experiments.density_ratio_estimation.step1_create_data import NSAMPLES_TEST
from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.binary_classification.binary_classifier import BinaryClassifier
from src.models.binary_classification.default_binary_classifier import build_default_binary_classifier


class BDRE(DensityRatioEstimator):
    def __init__(
        self, 
        input_dim: int, 
        classifier_builder: Callable[[], BinaryClassifier] = build_default_binary_classifier,
        device: str = "cuda"
    ):
        self.classifier = classifier_builder(input_dim)
        self.device = device
        self.classifier.to(self.device)

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        xs = torch.cat([samples_p0, samples_p1], dim=0)
        p0_labels = torch.ones((samples_p0.shape[0], 1), dtype=torch.float).to(self.device)
        p1_labels = torch.zeros((samples_p1.shape[0], 1), dtype=torch.float).to(self.device)
        ys = torch.cat([p0_labels, p1_labels], dim=0)
        self.classifier.fit(xs, ys)

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        return self.classifier.predict_logits(xs)


if __name__ == '__main__':
    from torch.distributions import MultivariateNormal
    from experiments.utils.two_gaussians_kl import create_two_gaussians_kl
    
    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 10
    KL_DISTANCE = 5

    # === CREATE SYNTHETIC DATA ===
    gaussian_pair = create_two_gaussians_kl(DIM, KL_DISTANCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
    mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)
    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))
    samples_pstar1 = p0.sample((NSAMPLES_TEST,))

    # === DENSITY RATIO ESTIMATION ===
    bdre = BDRE(DIM)
    bdre.fit(samples_p0, samples_p1)

    # === EVALUATION ===
    est_ldrs = bdre.predict_ldr(samples_pstar1)
    true_ldrs = p0.log_prob(samples_pstar1) - p1.log_prob(samples_pstar1)
    mae = torch.mean(torch.abs(est_ldrs - true_ldrs))
    print(f'MAE: {mae}')