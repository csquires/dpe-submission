import torch
from einops import rearrange

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.multiclass_classification.multiclass_classifier import MulticlassClassifier
from src.waypoints.waypoints1d import WaypointBuilder1D, DefaultWaypointBuilder1D


class MDRE(DensityRatioEstimator):
    def __init__(
        self, 
        classifier: MulticlassClassifier,
        waypoint_builder: WaypointBuilder1D = DefaultWaypointBuilder1D(),
        device: str = "cuda"
    ):
        self.device = device
        self.classifier = classifier.to(self.device)
        self.waypoint_builder = waypoint_builder
        self.num_waypoints = self.classifier.num_classes

    def fit(
        self, 
        samples_p0: torch.Tensor,  # [b0, dim]
        samples_p1: torch.Tensor   # [b1, dim]
    ) -> None:
        waypoint_samples = self.waypoint_builder.build_waypoints(samples_p0, samples_p1, self.num_waypoints)  # [w, b, dim]
        b = waypoint_samples.shape[1]
        xs = rearrange(waypoint_samples, 'w b dim -> (w b) dim')  # [n, dim] for n = w * b
        ys = torch.cat([torch.ones(b, dtype=torch.long) * i for i in range(self.num_waypoints)]).to(self.device)  # [n]
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
    from src.models.multiclass_classification import make_multiclass_classifier
    
    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 10
    KL_DIVERGENCE = 5
    DEVICE = "cuda"

    # === CREATE SYNTHETIC DATA ===
    gaussian_pair = create_two_gaussians_kl(DIM, KL_DIVERGENCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair['mu0'].to(DEVICE), gaussian_pair['Sigma0'].to(DEVICE)
    mu1, Sigma1 = gaussian_pair['mu1'].to(DEVICE), gaussian_pair['Sigma1'].to(DEVICE)
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)
    samples_p0 = p0.sample((NSAMPLES_TRAIN,)).to(DEVICE)
    samples_p1 = p1.sample((NSAMPLES_TRAIN,)).to(DEVICE)
    samples_pstar1 = p0.sample((NSAMPLES_TEST,)).to(DEVICE)

    # === DENSITY RATIO ESTIMATION ===
    num_waypoints = 10
    classifier = make_multiclass_classifier(name="default", input_dim=DIM, num_classes=num_waypoints)
    mdre = MDRE(classifier, device=DEVICE)
    mdre.fit(samples_p0, samples_p1)

    # === EVALUATION ===
    est_ldrs = mdre.predict_ldr(samples_pstar1)
    true_ldrs = p0.log_prob(samples_pstar1) - p1.log_prob(samples_pstar1)
    mae = torch.mean(torch.abs(est_ldrs - true_ldrs))
    print(f'MAE: {mae}')