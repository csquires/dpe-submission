import numpy as np
import torch
from torch import logdet, trace

from scipy.special import lambertw


def compute_gaussian_kl_divergence(
    mu0: torch.Tensor,
    Sigma0: torch.Tensor,
    mu1: torch.Tensor,
    Sigma1: torch.Tensor
) -> torch.Tensor:
    dim = mu0.shape[0]
    mean_term = 0.5 * ((mu1 - mu0).T @ torch.linalg.inv(Sigma1) @ (mu1 - mu0))
    cov_term = 0.5 * (trace(Sigma0 @ torch.linalg.inv(Sigma1))  - dim + logdet(Sigma1) - logdet(Sigma0))
    print(mean_term, cov_term)

    return mean_term + cov_term


def create_two_gaussians_kl(
    dim: int,
    k: float,  # KL(p0 || p1) = k
    beta: float = 0.5  # percentage of KL divergence due to covariance inequality
):
    k1 = (1 - beta) * k
    k2 = beta * k
    
    # solve for alpha in terms of k1
    c = 1 + 2 * k2 / dim
    alpha = -np.real(lambertw(-np.exp(-c)))  # maybe take different branch?
    # solve for delta in terms of alpha and k1
    delta = 2 * k1

    mu0 = torch.zeros(dim)
    Sigma0 = torch.Tensor(alpha * np.eye(dim))
    mu1 = torch.Tensor(np.sqrt(delta) * (np.ones(dim) / np.sqrt(dim)))
    Sigma1 = torch.eye(dim)
    breakpoint()

    return dict(mu0=mu0, Sigma0=Sigma0, mu1=mu1, Sigma1=Sigma1)


def create_two_gaussians_kl_range(
    dim: int,
    k: float,  # KL(p0 || p1) = k
    beta_min: float = 0.5,
    beta_max: float = 0.5,
    npairs: int = 100,
):
    betas = np.random.uniform(beta_min, beta_max, npairs)
    results = []
    for beta in betas:
        gaussian_pair = create_two_gaussians_kl(dim, k, beta)
        results.append(gaussian_pair)
    return results


if __name__ == '__main__':
    gaussian_pair = create_two_gaussians_kl(dim=3, k=128, beta=0.8)
    mu0 = gaussian_pair['mu0']
    Sigma0 = gaussian_pair['Sigma0']
    mu1 = gaussian_pair['mu1']
    Sigma1 = gaussian_pair['Sigma1']
    print(compute_gaussian_kl_divergence(mu0, Sigma0, mu1, Sigma1))

    from torch.distributions import MultivariateNormal, kl_divergence
    p0 = MultivariateNormal(mu0, Sigma0)
    p1 = MultivariateNormal(mu1, Sigma1)
    print("As computed by pytorch: ", kl_divergence(p0, p1))

    # gaussian_pairs = create_two_gaussians_kl_range(dim=3, k=128, beta_min=0.3, beta_max=0.7, npairs=100)
    # for gaussian_pair in gaussian_pairs:
    #     mu0 = gaussian_pair['mu0']
    #     Sigma0 = gaussian_pair['Sigma0']
    #     mu1 = gaussian_pair['mu1']
    #     Sigma1 = gaussian_pair['Sigma1']
    #     print(compute_gaussian_kl_divergence(mu0, Sigma0, mu1, Sigma1))