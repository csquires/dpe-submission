from typing import Tuple

import torch


def get_fractional_posterior(
    mu_pi: torch.Tensor,  # prior mean
    Sigma_pi: torch.Tensor,  # prior covariance
    obs_xi: torch.Tensor,  # observed design
    obs_y: torch.Tensor,  # observed outcome
    alpha: float,  # fractional posterior parameter
    sigma: float = 1.0,  # likelihood variance
) -> Tuple[torch.Tensor, torch.Tensor]:
    if alpha == 0:
        return mu_pi, Sigma_pi
    
    predictive_variance  = (obs_xi.T @ Sigma_pi @ obs_xi) + (sigma ** 2 / alpha)
    # mu_q
    mu_update = Sigma_pi @ obs_xi @ (obs_y - obs_xi.T @ mu_pi) / predictive_variance
    mu_q = mu_pi + mu_update.squeeze()
    # Sigma_q
    Sigma_update = Sigma_pi @ obs_xi @ obs_xi.T @ Sigma_pi / predictive_variance
    Sigma_q = Sigma_pi - Sigma_update

    return mu_q, Sigma_q


if __name__ == "__main__":
    from torch.distributions import MultivariateNormal
    torch.manual_seed(42)

    dim = 2
    mu_pi = torch.zeros(dim)
    Sigma_pi = torch.eye(dim)
    obs_xi = torch.randn((dim, 1))
    obs_y = obs_xi.sum()
    
    alpha = 1.0
    mu_q, Sigma_q = get_fractional_posterior(mu_pi, Sigma_pi, obs_xi, obs_y, alpha)
    print(mu_q)
    print(Sigma_q)

    # sanity check
    prior = MultivariateNormal(mu_pi, covariance_matrix=Sigma_pi)
    posterior = MultivariateNormal(mu_q, covariance_matrix=Sigma_q)
    predictive_mean = mu_pi.T @ obs_xi
    predictive_covariance = obs_xi.T @ Sigma_pi @ obs_xi + 1
    prior_predictive = MultivariateNormal(predictive_mean, covariance_matrix=predictive_covariance)

    def true_log_posterior(thetas):
        residuals = obs_y - thetas @ obs_xi
        log_likelihoods = MultivariateNormal(torch.zeros(1), covariance_matrix=torch.eye(1)).log_prob(residuals)
        log_priors = prior.log_prob(thetas)
        log_predictive = prior_predictive.log_prob(obs_y.reshape(1, 1))
        return log_priors + log_likelihoods - log_predictive
    

    thetas = MultivariateNormal(mu_pi, covariance_matrix=Sigma_pi).sample((5,)).squeeze()
    a1 = true_log_posterior(thetas)
    a2 = posterior.log_prob(thetas)
    print(a1)
    print(a2)