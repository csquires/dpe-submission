from abc import ABC, abstractmethod

import torch


class WaypointBuilder(ABC):
    @abstractmethod
    def build_waypoints(
        self, 
        samples_p0: torch.Tensor, 
        samples_p1: torch.Tensor,
        num_waypoints: int
    ) -> torch.Tensor:
        pass


class DefaultWaypointBuilder(WaypointBuilder):
    def _generate_alphas(self, num_waypoints: int) -> torch.Tensor:
        return torch.linspace(1, 0, num_waypoints-1)

    def build_waypoints(
        self, 
        samples_p0: torch.Tensor,  # [b, dim]
        samples_p1: torch.Tensor,  # [b, dim]
        num_waypoints: int
    ) -> torch.Tensor:
        alphas = self._generate_alphas(num_waypoints)  # [w]
        sqrt_alphas = torch.sqrt(alphas)
        sqrt_1_minus_alphas = torch.sqrt(1 - alphas)  # w

        b, dim = samples_p0.shape
        waypoint_samples = torch.zeros((num_waypoints, b, dim))
        waypoint_samples[0] = samples_p0
        waypoint_samples[-1] = samples_p1
        for i in range(1, num_waypoints-1):
            permuted_samples_p0 = samples_p0[torch.randperm(b)]
            permuted_samples_p1 = samples_p1[torch.randperm(b)]
            waypoint_samples[i] = sqrt_alphas[i] * permuted_samples_p0 + sqrt_1_minus_alphas[i] * permuted_samples_p1
        return waypoint_samples  # [w, b, dim]


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

    # === BUILD WAYPOINTS ===
    waypoint_builder = DefaultWaypointBuilder()
    waypoint_samples = waypoint_builder.build_waypoints(samples_p0, samples_p1, num_waypoints=10)
    
    empirical_means = torch.mean(waypoint_samples, axis=1)
    print(empirical_means)
    alphas = waypoint_builder._generate_alphas(10)
    sqrt_alphas = torch.sqrt(alphas)
    sqrt_1_minus_alphas = torch.sqrt(1 - alphas)
    expected_means = torch.einsum('w,d->wd', sqrt_alphas, mu0) + torch.einsum('w,d->wd', sqrt_1_minus_alphas, mu1)
    print(expected_means)