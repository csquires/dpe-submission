import torch

from src.eig_estimation.base import EIGEstimation
from src.density_ratio_estimation.base import DensityRatioEstimator



class EIGPlugin(EIGEstimation):
    def __init__(
        self, 
        density_ratio_estimator: DensityRatioEstimator,
    ):
        self.density_ratio_estimator = density_ratio_estimator

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
        self.density_ratio_estimator.fit(samples_p0, samples_p1)
        # predict density ratio
        est_ldrs = self.density_ratio_estimator.predict_ldr(samples_p0)
        # return estimated eig
        return torch.mean(est_ldrs)


if __name__ == "__main__":
    from src.density_ratio_estimation import BDRE
    from src.models.binary_classification import make_binary_classifier

    DATA_DIM = 2

    # build estimator
    classifier = make_binary_classifier(name="default", input_dim=DATA_DIM+1)
    density_ratio_estimator = BDRE(classifier, device="cuda")
    eig_plugin = EIGPlugin(density_ratio_estimator=density_ratio_estimator)

    # generate fake data
    samples_theta = torch.randn(1000, DATA_DIM).to("cuda")
    samples_y = torch.randn(1000, 1).to("cuda")

    # estimate eig
    est_eig = eig_plugin.estimate_eig(samples_theta, samples_y)
    print(f"Estimated EIG: {est_eig}")