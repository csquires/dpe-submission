import torch

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.binary_classification.binary_classifier import BinaryClassifier
from src.waypoints.waypoints1d import WaypointBuilder1D, DefaultWaypointBuilder1D


class TDRE(DensityRatioEstimator):
    def __init__(
        self, 
        classifiers: list[BinaryClassifier],
        waypoint_builder: WaypointBuilder1D = DefaultWaypointBuilder1D(),
        num_waypoints: int = 10,
        device: str = "cuda"
    ):
        # note: the i-th classifier discrimates between waypoint i (in the numerator) and waypoint i+1 (in the denominator)
        self.device = device
        self.classifiers = [classifier.to(self.device) for classifier in classifiers]
        self.waypoint_builder = waypoint_builder
        self.num_waypoints = num_waypoints
        
    def fit(
        self, 
        samples_p0: torch.Tensor,  # [b0, dim]
        samples_p1: torch.Tensor   # [b1, dim]
    ) -> None:
        waypoint_samples = self.waypoint_builder.build_waypoints(samples_p0, samples_p1, self.num_waypoints)  # [w, b, dim]
        b = waypoint_samples.shape[1]
        for i in range(self.num_waypoints - 1):
            xs = torch.cat([waypoint_samples[i], waypoint_samples[i+1]], dim=0)
            p_num_labels = torch.ones((b, 1), dtype=torch.float).to(self.device)
            p_den_labels = torch.zeros((b, 1), dtype=torch.float).to(self.device)
            ys = torch.cat([p_num_labels, p_den_labels], dim=0).to(self.device)
            self.classifiers[i].fit(xs, ys)

    def predict_ldr(
        self, 
        xs: torch.Tensor  # [b, dim]
    ) -> torch.Tensor:
        waypoint_ldrs = torch.zeros(xs.shape[0], self.num_waypoints-1).to(self.device)  # [b, w-1]
        for i in range(self.num_waypoints-1):
            waypoint_ldrs[:, i] = self.classifiers[i].predict_logits(xs)
        return waypoint_ldrs.sum(axis=1)  # [b]


if __name__ == '__main__':
    from torch.distributions import MultivariateNormal
    from experiments.utils.two_gaussians_kl import create_two_gaussians_kl
    from src.models.binary_classification import make_pairwise_binary_classifiers

    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 10
    KL_DISTANCE = 5
    DEVICE = "cuda"

    # === CREATE SYNTHETIC DATA ===
    gaussian_pair = create_two_gaussians_kl(DIM, KL_DISTANCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair['mu0'].to(DEVICE), gaussian_pair['Sigma0'].to(DEVICE)
    mu1, Sigma1 = gaussian_pair['mu1'].to(DEVICE), gaussian_pair['Sigma1'].to(DEVICE)
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)
    samples_p0 = p0.sample((NSAMPLES_TRAIN,)).to(DEVICE)
    samples_p1 = p1.sample((NSAMPLES_TRAIN,)).to(DEVICE)
    samples_pstar1 = p0.sample((NSAMPLES_TEST,)).to(DEVICE)

    # === DENSITY RATIO ESTIMATION ===
    num_waypoints = 10
    classifiers = make_pairwise_binary_classifiers(name="default", num_classes=num_waypoints, input_dim=DIM)
    tdre = TDRE(classifiers, device=DEVICE)
    tdre.fit(samples_p0, samples_p1)

    # === EVALUATION ===
    est_ldrs = tdre.predict_ldr(samples_pstar1)
    true_ldrs = p0.log_prob(samples_pstar1) - p1.log_prob(samples_pstar1)
    mae = torch.mean(torch.abs(est_ldrs - true_ldrs))
    print(f'MAE: {mae}')