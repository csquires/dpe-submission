"""helpers for eig-style density-ratio estimation.

context:
    mutual information between theta and y can be expressed as
    $E_{p(\\theta, y)}[\\log r(\\theta, y)]$ where
    $r = p(\\theta, y) / (p(\\theta) p(y))$. a DRE estimates this $r$ given
    samples from p0 = joint and p1 = product-of-marginals; the latter is
    manufactured by independently shuffling theta and y.

contents:
    joint_and_shuffled: build (p0, p1) from (theta, y).
    true_ldrs_gaussian_linear: closed-form $\\log r$ at the joint samples
        for the gaussian linear model $y = \\theta^\\top \\xi + N(0, \\sigma^2)$
        with $\\theta \\sim N(\\mu_\\pi, \\Sigma_\\pi)$.
"""
from __future__ import annotations

import torch


def joint_and_shuffled(theta: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """concatenate (theta, y) along feature axis to form p0; independently
    permute rows of theta and y to form p1 (product of marginals).

    args:
        theta: shape (N, D)
        y:     shape (N, K)
    returns:
        (p0, p1) each of shape (N, D + K).
    """
    n = theta.shape[0]
    p0 = torch.cat([theta, y], dim=1)
    perm_t = torch.randperm(n, device=theta.device)
    perm_y = torch.randperm(n, device=theta.device)
    p1 = torch.cat([theta[perm_t], y[perm_y]], dim=1)
    return p0, p1


def true_ldrs_gaussian_linear(
    theta: torch.Tensor,
    y: torch.Tensor,
    mu_pi: torch.Tensor,
    Sigma_pi: torch.Tensor,
    xi: torch.Tensor,
    sigma2: float = 1.0,
) -> torch.Tensor:
    """closed-form per-sample $\\log r(\\theta, y)$ for the gaussian linear model.

    derivation:
        $\\log r(\\theta, y)
         = \\log p(y \\mid \\theta) - \\log p(y)$,
        $p(y \\mid \\theta) = N(\\theta^\\top \\xi, \\sigma^2)$,
        $p(y) = N(\\mu_\\pi^\\top \\xi, \\xi^\\top \\Sigma_\\pi \\xi + \\sigma^2)$.
    expectation under the joint recovers the standard eig:
        $E[\\log r] = 0.5 \\log(1 + \\xi^\\top \\Sigma_\\pi \\xi / \\sigma^2)$.

    args:
        theta:    (N, D)
        y:        (N, 1)
        mu_pi:    (D,) or (D, 1)
        Sigma_pi: (D, D)
        xi:       (D, 1) or (D,)
        sigma2:   noise variance.
    returns:
        true_ldrs: (N,)
    """
    xi_flat = xi.reshape(-1)
    mu_pi_flat = mu_pi.reshape(-1)
    mean_cond = (theta * xi_flat).sum(dim=1, keepdim=True)
    mu_y = (mu_pi_flat * xi_flat).sum()
    quad = xi_flat @ Sigma_pi @ xi_flat
    var_y = quad + sigma2
    y_col = y.reshape(-1, 1)
    log_var_ratio = 0.5 * torch.log(var_y / sigma2)
    err_term = 0.5 * ((y_col - mu_y) ** 2 / var_y - (y_col - mean_cond) ** 2 / sigma2)
    return (log_var_ratio + err_term).reshape(-1)
