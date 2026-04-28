import numpy as np
from scipy import linalg, special


def bellman_occupancy(
    P: np.ndarray,
    mu0: np.ndarray,
    pi: np.ndarray,
    gamma: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute closed-form discounted state-action occupancy d^pi(s, a)
    via Bellman flow inversion.

    Solves the linear system:
      d^pi(s, a) = (1 - gamma) * (I - gamma * (P^pi)^T)^{-1} * (mu_0(s) * pi(a | s))

    where P^pi is the s-a-transition kernel: P^pi[(s, a), (s', a')] = P(s' | s, a) * pi(a' | s').

    Args:
        P: shape [|S|, |A|, |S|], transition dynamics.
        mu0: shape [|S|], initial state distribution.
        pi: shape [|S|, |A|], stochastic policy.
        gamma: discount factor, typically 0.99.
        eps: numerical tolerance, default 1e-12.

    Returns:
        d: shape [|S|, |A|], float64, with d.sum() ~= 1 (within eps).

    Algorithm:
      1. extract dimensions from shapes.
      2. build P^pi as [|S|*|A|, |S|*|A|] via broadcast and reshape.
      3. form RHS: (1 - gamma) * mu_0 \otimes pi, flattened.
      4. construct system matrix A = I - gamma * P_pi^T.
      5. solve A * d_flat = rho_init.
      6. reshape to [|S|, |A|].
      7. sanity checks: assert d.sum() ~ 1, clip negatives, renormalize if needed.
      8. return as float64.
    """
    n_states, n_actions = pi.shape

    # build P^pi: [|S|, |A|, |S|, |A|] -> [|S|*|A|, |S|*|A|]
    P_pi_4d = P[:, :, :, None] * pi[None, None, :, :]  # broadcast
    P_pi = P_pi_4d.reshape(n_states * n_actions, n_states * n_actions)

    # RHS: (1 - gamma) * mu_0(s) * pi(a|s), broadcast to [S, A] then flatten to [S*A]
    rho_init = ((1.0 - gamma) * mu0[:, None] * pi).reshape(-1)

    # system matrix: A = I - gamma * P_pi^T
    A = np.eye(n_states * n_actions) - gamma * P_pi.T

    # solve A * d_flat = rho_init
    d_flat = linalg.solve(A, rho_init, overwrite_a=True, check_finite=False)

    # reshape to [|S|, |A|]
    d = d_flat.reshape(n_states, n_actions)

    # sanity: d should sum to 1
    d_sum = d.sum()
    if not np.abs(d_sum - 1.0) <= eps:
        raise ValueError("occupancy does not integrate to 1")

    # clip negative entries (numerical drift)
    if (d < -eps).any():
        d[d < -eps] = 0.0

    # renormalize if clipped
    if (d < 0).any():
        d /= d.sum()

    return d.astype(np.float64)


def kl_occupancy(
    d_p: np.ndarray,
    d_q: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute KL divergence KL(p || q) between two occupancies.

    Formula: KL(p || q) = sum_{s,a} p(s,a) * log(p(s,a) / (q(s,a) + eps))

    Uses scipy.special.xlogy for numerical safety at p = 0.

    Args:
        d_p: shape [|S|, |A|], float64, normalized occupancy.
        d_q: shape [|S|, |A|], float64, normalized occupancy.
        eps: numerical floor for division, default 1e-12.

    Returns:
        kl_value: float, non-negative.

    Algorithm:
      1. flatten d_p and d_q to 1-D.
      2. compute ratio r = d_p / (d_q + eps).
      3. apply xlogy(d_p_flat, r).sum() to get KL.
    """
    d_p_flat = d_p.reshape(-1)
    d_q_flat = d_q.reshape(-1)

    ratio = d_p_flat / (d_q_flat + eps)
    kl_value = special.xlogy(d_p_flat, ratio).sum()

    return float(kl_value)


def mixture_policy(
    pi_a: np.ndarray,
    pi_b: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Compute convex mixture of two policies: pi_mix = (1 - beta) * pi_a + beta * pi_b.

    Args:
        pi_a: shape [|S|, |A|], float64, stochastic policy A.
        pi_b: shape [|S|, |A|], float64, stochastic policy B.
        beta: blending weight for B, in [0, 1].

    Returns:
        pi: shape [|S|, |A|], float64, with rows summing to 1 within eps.

    Algorithm:
      1. validate 0 <= beta <= 1; raise ValueError if not.
      2. compute pi_mix = (1 - beta) * pi_a + beta * pi_b.
      3. sanity check: assert pi_a and pi_b rows sum to ~1.
      4. sanity check: assert pi_mix rows sum to ~1.
      5. return pi_mix as float64.
    """
    eps = 1e-12

    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1]")

    # sanity: pi_a and pi_b are valid stochastic policies
    assert np.allclose(pi_a.sum(axis=1), 1.0, atol=eps), "pi_a rows must sum to 1"
    assert np.allclose(pi_b.sum(axis=1), 1.0, atol=eps), "pi_b rows must sum to 1"

    # mixture
    pi_mix = (1.0 - beta) * pi_a + beta * pi_b

    # sanity: pi_mix is also a valid stochastic policy
    assert np.allclose(pi_mix.sum(axis=1), 1.0, atol=eps), "pi_mix rows must sum to 1"

    return pi_mix.astype(np.float64)
