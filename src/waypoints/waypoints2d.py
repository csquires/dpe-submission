from abc import ABC, abstractmethod

import torch


class WaypointBuilder2D(ABC):
    @abstractmethod
    def build_waypoints(
        self, 
        samples_pstar: torch.Tensor,
        samples_p0: torch.Tensor, 
        samples_p1: torch.Tensor,
        num_waypoints: int
    ) -> torch.Tensor:
        pass


class DefaultWaypointBuilder2D(WaypointBuilder2D):
    def _generate_alphas(self, num_waypoints: int) -> torch.Tensor:
        return torch.linspace(1, 0, num_waypoints-1)

    def build_waypoints(
        self, 
        samples_pstar: torch.Tensor,  # [bstar, dim]
        samples_p0: torch.Tensor,  # [b0, dim]
        samples_p1: torch.Tensor,  # [b1, dim]
        num_waypoints: int
    ) -> torch.Tensor:
        alphas = self._generate_alphas(num_waypoints)  # [w]
        p0_weights = alphas # [w]
        p1_weights = torch.sqrt(1 - alphas ** 2)  # [w]

        bstar, dim = samples_pstar.shape
        b0, dim = samples_p0.shape
        b1, dim = samples_p1.shape
        b = max(bstar, b0, b1)
        waypoint_samples = torch.zeros((num_waypoints, b, dim), device=samples_p0.device)
        waypoint_samples[0] = samples_pstar[torch.randint(0, bstar, (b,))]  # [b, dim], bootstrap draws from pstar
        waypoint_samples[0] = samples_p0[torch.randint(0, b0, (b,))]  # [b, dim], bootstrap draws from p0
        waypoint_samples[-1] = samples_p1[torch.randint(0, b0, (b,))]  # [b, dim], bootstrap draws from p1
        for i in range(1, num_waypoints-1):
            bootstrapped_samples_p0 = samples_p0[torch.randint(0, b0, (b,))]
            bootstrapped_samples_p1 = samples_p1[torch.randint(0, b1, (b,))]
            waypoint_samples[i] = p0_weights[i] * bootstrapped_samples_p0 + p1_weights[i] * bootstrapped_samples_p1
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
    samples_p0 = p0.sample((NSAMPLES_TRAIN//2,))
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