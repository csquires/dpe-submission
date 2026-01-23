from typing import Callable

import torch
from einops import rearrange

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.multiclass_classification.multiclass_classifier import MulticlassClassifier
from src.models.multiclass_classification.default_multiclass_classifier import build_default_multiclass_classifier
from src.utils.waypoints import WaypointBuilder, DefaultWaypointBuilder


class MDRE(DensityRatioEstimator):
    def __init__(
        self, 
        input_dim: int, 
        classifier_builder: Callable[[], MulticlassClassifier] = build_default_multiclass_classifier,
        waypoint_builder: WaypointBuilder = DefaultWaypointBuilder(),
        num_waypoints: int = 10
    ):
        self.classifier = classifier_builder(input_dim, num_waypoints)
        self.waypoint_builder = waypoint_builder
        self.num_waypoints = num_waypoints

    def fit(
        self, 
        samples_p0: torch.Tensor,  # [b0, dim]
        samples_p1: torch.Tensor   # [b1, dim]
    ) -> None:
        waypoint_samples = self.waypoint_builder.build_waypoints(samples_p0, samples_p1, self.num_waypoints)  # [w, b, dim]
        b = waypoint_samples.shape[1]
        xs = rearrange(waypoint_samples, 'w b dim -> (w b) dim')  # [n, dim] for n = w * b
        ys = torch.cat([torch.ones(b, dtype=torch.long) * i for i in range(self.num_waypoints)])  # [n]
        self.classifier.fit(xs, ys)

    def predict_ldr(
        self, 
        xs: torch.Tensor  # [b, dim]
    ) -> torch.Tensor:
        logits = self.classifier.predict_logits(xs)
        p1_logits = logits[:, -1]
        p0_logits = logits[:, 0]
        return p0_logits - p1_logits


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
    mdre = MDRE(DIM)
    mdre.fit(samples_p0, samples_p1)

    # === EVALUATION ===
    est_ldrs = mdre.predict_ldr(samples_pstar1)
    true_ldrs = p0.log_prob(samples_pstar1) - p1.log_prob(samples_pstar1)
    mae = torch.mean(torch.abs(est_ldrs - true_ldrs))
    print(f'MAE: {mae}')