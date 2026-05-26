"""synthetic fixtures for regression method benchmarking.

pure-function factories returning SyntheticFixture instances for layer 1
(single-shot) and layer 3 (watershed) regression benchmarks. each fixture
packages a pair or triplet of distributions, corresponding true likelihood
ratio, and marginal score under bridge-path interpolation.

distributions:
  - gaussian_gaussian: two isotropic Gaussians (stock vfm/ctsm).
  - gaussian_mixture: isotropic N(0,I) vs mixture of Gaussians (layer 3).
  - heteroscedastic: isotropic N(0,I) vs N(0, var_ratio*I) (variance-only).
  - triangular_gaussian: three Gaussians with anchor mu_star (layer 3).
"""
from dataclasses import dataclass
from typing import Callable, Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class SyntheticFixture:
    """synthetic distribution pair(s) and associated callables for benchmarking.

    samples_p0, samples_p1: tensors of shape [n, D], float32, cpu.
    samples_pstar: optional third anchor samples [n, D] (triangular only).
    true_ldr: log likelihood ratio; Tensor[N, D] -> Tensor[N].
    analytic_score: marginal score under bridge path; (x_tau: [B, D], tau: [B, 1],
        path) -> Tensor[B, 1]. path encapsulates gamma(tau), dgamma_dtau(tau),
        and for triangular fixtures, weights(tau).
    meta: dict of kind, D, params, seed, n for logging and serialization.
    """
    samples_p0: Tensor
    samples_p1: Tensor
    samples_pstar: Tensor | None
    true_ldr: Callable[[Tensor], Tensor]
    analytic_score: Callable[[Tensor, Tensor, Any], Tensor]
    meta: dict


def gaussian_gaussian(
    D: int,
    mu: Tensor | None = None,
    sigma_data: float = 1.0,
    seed: int = 0,
    n: int = 1024,
) -> SyntheticFixture:
    """gaussian-to-gaussian distribution pair.

    p_0 = N(0, sigma_data^2 I_D), p_1 = N(mu, sigma_data^2 I_D).
    default mu shifts first coordinate by 1.5 (breaks symmetry).

    args:
      D: dimensionality.
      mu: mean of p_1. if None, defaults to [1.5, 0, ..., 0].
      sigma_data: isotropic std dev of both distributions.
      seed: local generator seed (does not pollute global state).
      n: number of samples per distribution.

    returns: SyntheticFixture with true_ldr and analytic_score closures.
    """
    if mu is None:
        mu = torch.cat([torch.tensor([1.5]), torch.zeros(D - 1)])
    else:
        mu = mu.float()

    # sample via local generator to avoid polluting global state
    gen = torch.Generator(device='cpu').manual_seed(seed)
    samples_p0 = torch.randn(n, D, generator=gen, dtype=torch.float32) * sigma_data
    samples_p1 = torch.randn(n, D, generator=gen, dtype=torch.float32) * sigma_data + mu

    def true_ldr(x: Tensor) -> Tensor:
        """log likelihood ratio p_1 / p_0.

        (x @ mu) / sigma_data^2 - 0.5 * ||mu||^2 / sigma_data^2.
        """
        sigma_sq = sigma_data ** 2
        return (x @ mu) / sigma_sq - 0.5 * (mu @ mu) / sigma_sq

    def analytic_score(x_tau: Tensor, tau: Tensor, path: Any) -> Tensor:
        """marginal score d_tau log p_tau(x_tau) under bridge interpolation.

        p_tau is Gaussian with mean mu_tau = tau * mu and variance
        v(tau) = sigma_data^2 * ((1-tau)^2 + tau^2) + gamma(tau)^2.

        score has three terms (full quadratic derivation):
          1. (v_prime / (2*v^2)) * ||x_tau - mu_tau||^2
          2. (1 / v) * (x_tau - mu_tau) @ mu
          3. -0.5 * D * v_prime / v

        where v_prime = d_tau v = sigma_data^2 * (-2(1-tau) + 2*tau) +
          2*gamma(tau)*dgamma_dtau(tau).
        """
        # broadcast shapes: tau [B, 1], x_tau [B, D], mu [D]
        tau_scalar = tau.squeeze(-1)  # [B]
        mu_tau = tau.unsqueeze(-1) * mu  # [B, 1] * [D] -> [B, D]
        resid = x_tau - mu_tau  # [B, D]

        # variance and its derivative
        sigma_sq = sigma_data ** 2
        gamma_tau = path.gamma(tau_scalar)  # [B]
        dgamma_tau = path.dgamma_dtau(tau_scalar)  # [B]

        # v(tau) = sigma_data^2 * ((1-tau)^2 + tau^2) + gamma(tau)^2
        base_var = sigma_sq * ((1.0 - tau_scalar) ** 2 + tau_scalar ** 2)
        v = base_var + gamma_tau ** 2  # [B]

        # v_prime = sigma_data^2 * (-2(1-tau) + 2*tau) + 2*gamma*dgamma
        v_prime = (
            sigma_sq * (-2.0 * (1.0 - tau_scalar) + 2.0 * tau_scalar)
            + 2.0 * gamma_tau * dgamma_tau
        )  # [B]

        # three terms
        resid_sq = (resid ** 2).sum(dim=1)  # [B]
        resid_mu = (resid @ mu).unsqueeze(-1)  # [B, 1]

        term1 = (v_prime / (2.0 * v ** 2)).unsqueeze(-1) * resid_sq.unsqueeze(-1)  # [B, 1]
        term2 = (1.0 / v).unsqueeze(-1) * resid_mu  # [B, 1]
        term3 = -0.5 * D * (v_prime / v).unsqueeze(-1)  # [B, 1]

        score = term1 + term2 + term3
        return score

    mu_list = mu.tolist() if mu.dim() == 1 else [float(mu)]
    meta = {
        "kind": "gaussian_gaussian",
        "D": D,
        "mu": mu_list,
        "sigma_data": sigma_data,
        "seed": seed,
        "n": n,
    }

    return SyntheticFixture(
        samples_p0=samples_p0,
        samples_p1=samples_p1,
        samples_pstar=None,
        true_ldr=true_ldr,
        analytic_score=analytic_score,
        meta=meta,
    )


def gaussian_mixture(
    D: int,
    components: list[Tensor],
    weights: list[float],
    seed: int = 0,
    n: int = 1024,
) -> SyntheticFixture:
    """isotropic Gaussian vs mixture of Gaussians.

    p_0 = N(0, I_D), p_1 = sum_k w_k N(mu_k, I_D).
    fixture is for layer 3 (watershed regression) only; analytic_score
    is not implemented (numerical differentiation required).

    args:
      D: dimensionality.
      components: list of mean tensors, each [D].
      weights: list of mixture weights (should sum to 1).
      seed: generator seed.
      n: number of samples per distribution.

    returns: SyntheticFixture with NotImplementedError on analytic_score.
    """
    gen = torch.Generator(device='cpu').manual_seed(seed)
    samples_p0 = torch.randn(n, D, generator=gen, dtype=torch.float32)

    # sample from mixture: draw component indicators, then sample from each
    components_t = torch.stack([c.float() for c in components])  # [K, D]
    k_indices = torch.multinomial(
        torch.tensor(weights, dtype=torch.float32),
        n,
        replacement=True,
    )  # [n]
    component_means = components_t[k_indices]  # [n, D]
    samples_p1 = torch.randn(n, D, generator=gen, dtype=torch.float32) + component_means

    def true_ldr(x: Tensor) -> Tensor:
        """logsumexp of component log-ratios.

        log sum_k w_k exp(0.5 * ||x - mu_k||^2 - 0.5 * ||mu_k||^2).
        """
        # compute per-component log-ratio
        log_ratios = []
        for i, (mu_k, w_k) in enumerate(zip(components, weights)):
            mu_k = mu_k.float()
            x_mu_sq = ((x - mu_k) ** 2).sum(dim=1)  # [n]
            mu_sq = (mu_k ** 2).sum()  # scalar
            log_ratio = 0.5 * x_mu_sq - 0.5 * mu_sq + torch.log(torch.tensor(w_k))
            log_ratios.append(log_ratio)
        # logsumexp over components
        log_ratios_stack = torch.stack(log_ratios, dim=1)  # [n, K]
        return torch.logsumexp(log_ratios_stack, dim=1)

    def analytic_score(x_tau: Tensor, tau: Tensor, path: Any) -> Tensor:
        """analytic score not available for mixture targets.

        layer 3 (watershed) regression requires numerical differentiation
        of the marginal or sampling-based score estimation.
        """
        raise NotImplementedError(
            "gaussian_mixture analytic_score: layer 3 watershed only; "
            "use numeric differentiation or skip."
        )

    components_list = [c.tolist() if c.dim() == 1 else [float(c)] for c in components]
    meta = {
        "kind": "gaussian_mixture",
        "D": D,
        "components": components_list,
        "weights": weights,
        "seed": seed,
        "n": n,
    }

    return SyntheticFixture(
        samples_p0=samples_p0,
        samples_p1=samples_p1,
        samples_pstar=None,
        true_ldr=true_ldr,
        analytic_score=analytic_score,
        meta=meta,
    )


def heteroscedastic(
    D: int,
    var_ratio: float = 4.0,
    seed: int = 0,
    n: int = 1024,
) -> SyntheticFixture:
    """variance-only shift: isotropic N(0, I_D) vs N(0, var_ratio * I_D).

    p_0 = N(0, I_D), p_1 = N(0, var_ratio * I_D). zero means; variance
    ratio is the only change.

    args:
      D: dimensionality.
      var_ratio: variance multiplier for p_1.
      seed: generator seed.
      n: number of samples per distribution.

    returns: SyntheticFixture with heteroscedastic analytic_score.
    """
    gen = torch.Generator(device='cpu').manual_seed(seed)
    samples_p0 = torch.randn(n, D, generator=gen, dtype=torch.float32)
    samples_p1 = torch.randn(n, D, generator=gen, dtype=torch.float32) * (var_ratio ** 0.5)

    def true_ldr(x: Tensor) -> Tensor:
        """log p_1 / p_0 under variance shift.

        0.5 * (1 - 1/var_ratio) * ||x||^2 - 0.5 * D * log(var_ratio).
        """
        x_sq = (x ** 2).sum(dim=1)
        return 0.5 * (1.0 - 1.0 / var_ratio) * x_sq - 0.5 * D * torch.log(torch.tensor(var_ratio))

    def analytic_score(x_tau: Tensor, tau: Tensor, path: Any) -> Tensor:
        """marginal score for heteroscedastic path.

        p_tau is Gaussian with mu_tau = 0, variance
        v(tau) = (1-tau)^2 + tau^2*var_ratio + gamma(tau)^2.

        score (simplified; no linear term):
          (v_prime / (2*v^2)) * ||x_tau||^2 - 0.5 * D * v_prime / v
        """
        tau_scalar = tau.squeeze(-1)  # [B]
        gamma_tau = path.gamma(tau_scalar)  # [B]
        dgamma_tau = path.dgamma_dtau(tau_scalar)  # [B]

        # variance: (1-tau)^2 + tau^2*var_ratio + gamma^2
        v = (1.0 - tau_scalar) ** 2 + tau_scalar ** 2 * var_ratio + gamma_tau ** 2  # [B]

        # derivative: -2(1-tau) + 2*tau*var_ratio + 2*gamma*dgamma
        v_prime = (
            -2.0 * (1.0 - tau_scalar) + 2.0 * tau_scalar * var_ratio + 2.0 * gamma_tau * dgamma_tau
        )  # [B]

        # two terms
        x_sq = (x_tau ** 2).sum(dim=1)  # [B]
        term1 = (v_prime / (2.0 * v ** 2)).unsqueeze(-1) * x_sq.unsqueeze(-1)  # [B, 1]
        term2 = -0.5 * D * (v_prime / v).unsqueeze(-1)  # [B, 1]

        score = term1 + term2
        return score

    meta = {
        "kind": "heteroscedastic",
        "D": D,
        "var_ratio": var_ratio,
        "seed": seed,
        "n": n,
    }

    return SyntheticFixture(
        samples_p0=samples_p0,
        samples_p1=samples_p1,
        samples_pstar=None,
        true_ldr=true_ldr,
        analytic_score=analytic_score,
        meta=meta,
    )


def triangular_gaussian(
    D: int,
    mu: Tensor,
    mu_star: Tensor,
    seed: int = 0,
    n: int = 1024,
) -> SyntheticFixture:
    """three-anchor Gaussian path: p_0 ~ N(0, I), p_1 ~ N(mu, I), p_star ~ N(mu_star, I).

    extends gaussian_gaussian with a third distribution (layer 3 watershed).
    true_ldr is p_0 vs p_1 only (ignores p_star). analytic_score uses
    path.weights(tau) to interpolate means and variances.

    args:
      D: dimensionality.
      mu: mean of p_1, shape [D].
      mu_star: mean of p_star (anchor), shape [D].
      seed: generator seed.
      n: number of samples per distribution.

    returns: SyntheticFixture with three sample tensors.
    """
    mu = mu.float()
    mu_star = mu_star.float()

    gen = torch.Generator(device='cpu').manual_seed(seed)
    samples_p0 = torch.randn(n, D, generator=gen, dtype=torch.float32)
    samples_p1 = torch.randn(n, D, generator=gen, dtype=torch.float32) + mu
    samples_pstar = torch.randn(n, D, generator=gen, dtype=torch.float32) + mu_star

    def true_ldr(x: Tensor) -> Tensor:
        """log p_1 / p_0. ignores p_star; same as gaussian_gaussian."""
        return (x @ mu) - 0.5 * (mu @ mu)

    def analytic_score(x_tau: Tensor, tau: Tensor, path: Any) -> Tensor:
        """marginal score under triangular path.

        p_tau is mixture: alpha(tau)*N(0, ...) + beta(tau)*N(mu, ...)
          + w_star(tau)*N(mu_star, ...). marginal mean mu_tau = beta*mu + w_star*mu_star
        and variance v = 1 + gamma^2 (all components unit-variance before gamma).
        includes derivative term from time-varying weights.

        score has four terms: variance quadratic term, linear mean term, mean
        derivative term, and variance derivative term.
        """
        tau_scalar = tau.squeeze(-1)  # [B]
        gamma_tau = path.gamma(tau_scalar)  # [B]
        dgamma_tau = path.dgamma_dtau(tau_scalar)  # [B]
        w = path.weights(tau_scalar)  # TriangularWeights1D

        # variance and its derivative (gamma is the only time-dependent part)
        v = 1.0 + gamma_tau ** 2  # [B]
        v_prime = 2.0 * gamma_tau * dgamma_tau  # [B]

        # marginal mean: beta(tau)*mu + w_star(tau)*mu_star
        mu_tau = w.beta.unsqueeze(-1) * mu + w.w_star.unsqueeze(-1) * mu_star  # [B, D]
        # derivative of marginal mean
        d_mu_tau = w.d_beta.unsqueeze(-1) * mu + w.d_w_star.unsqueeze(-1) * mu_star  # [B, D]

        resid = x_tau - mu_tau  # [B, D]

        # three score terms (quadratic, linear in time-derivative, variance)
        resid_sq = (resid ** 2).sum(dim=1)  # [B]
        resid_d_mu_tau = (resid * d_mu_tau).sum(dim=1)  # [B]

        term1 = (v_prime / (2.0 * v ** 2)).unsqueeze(-1) * resid_sq.unsqueeze(-1)  # [B, 1]
        term2 = (1.0 / v).unsqueeze(-1) * resid_d_mu_tau.unsqueeze(-1)  # [B, 1]
        term3 = -0.5 * D * (v_prime / v).unsqueeze(-1)  # [B, 1]

        score = term1 + term2 + term3
        return score

    mu_list = mu.tolist() if mu.dim() == 1 else [float(mu)]
    mu_star_list = mu_star.tolist() if mu_star.dim() == 1 else [float(mu_star)]

    meta = {
        "kind": "triangular_gaussian",
        "D": D,
        "mu": mu_list,
        "mu_star": mu_star_list,
        "seed": seed,
        "n": n,
    }

    return SyntheticFixture(
        samples_p0=samples_p0,
        samples_p1=samples_p1,
        samples_pstar=samples_pstar,
        true_ldr=true_ldr,
        analytic_score=analytic_score,
        meta=meta,
    )
