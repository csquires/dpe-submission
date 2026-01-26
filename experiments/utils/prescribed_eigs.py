import torch
from torch import log1p


def compute_gaussian_eig(
    mu_pi: torch.Tensor,  # prior mean
    Sigma_pi: torch.Tensor,  # prior covariance
    xi: torch.Tensor,  # design
    sigma: float = 1.0  # likelihood variance
):
    rho = xi.T @ Sigma_pi @ xi
    return 1/2 * log1p(rho / sigma ** 2)


def create_prior_eig_range(
    dim: int,
    eig_min: float,
    eig_max: float,
    sigma: float = 1.0  # likelihood variance
):
    eigs = torch.linspace(eig_min, eig_max, dim)
    rhos = (sigma ** 2) * (torch.exp(2 * eigs) - 1)
    mu_pi = torch.zeros(dim)
    Sigma_pi = torch.eye(dim) * rhos
    return mu_pi, Sigma_pi


def create_design_eig(
    mu_pi: torch.Tensor,  # prior mean
    Sigma_pi: torch.Tensor,  # prior covariance
    desired_eig: float,  # desired EIG
    sigma: float = 1.0  # likelihood variance
):
    # get eigenvectors and eigenvalues of prior covariance matrix
    if not torch.allclose(Sigma_pi, torch.diag(torch.diagonal(Sigma_pi))):
        raise ValueError("Non-diagonal prior covariance matrix not currently supported (TODO: add eigenvalue decomposition)")
    eigvals = torch.diagonal(Sigma_pi)
    eigvecs = torch.eye(mu_pi.shape[0])

    # transform desired EIG into eigenvalue space
    rho = (sigma ** 2) * (torch.exp(2 * torch.tensor(desired_eig)) - 1)

    # pick upper and lower eigenvalue index at random
    upper_eigval_idxs = torch.where(eigvals > rho)[0]
    lower_eigval_idxs = torch.where(eigvals < rho)[0]
    upper_eigval_idx = upper_eigval_idxs[torch.randint(len(upper_eigval_idxs), (1,))]
    lower_eigval_idx = lower_eigval_idxs[torch.randint(len(lower_eigval_idxs), (1,))]
    
    # compute weights
    lambda_upper = eigvals[upper_eigval_idx]
    lambda_lower = eigvals[lower_eigval_idx]
    t = (rho - lambda_lower) / (lambda_upper - lambda_lower)
    weight_upper = torch.sqrt(t)
    weight_lower = torch.sqrt(1 - t)

    # create the design
    xi = weight_upper * eigvecs[:, upper_eigval_idx] + weight_lower * eigvecs[:, lower_eigval_idx]
    return xi


if __name__ == '__main__':
    mu_pi, Sigma_pi = create_prior_eig_range(dim=3, eig_min=0.1, eig_max=0.9, sigma=1.0)
    xi = create_design_eig(mu_pi, Sigma_pi, desired_eig=0.7, sigma=1.0)
    xi_eig = compute_gaussian_eig(mu_pi, Sigma_pi, xi, sigma=1.0)
    print("EIG(xi) = ", xi_eig)