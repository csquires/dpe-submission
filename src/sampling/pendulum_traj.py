"""
vectorized pendulum trajectory rollout, log-density, and KL estimation.

provides closed-form trajectory log-density computation under a policy,
direct Monte Carlo KL divergence estimation with per-trajectory standard
error diagnostics, and trajectory packing for downstream serialization.
all operations use NumPy on CPU.
"""
import numpy as np
from scipy.special import logsumexp
from typing import Callable, Tuple
import warnings

# forward-reference type annotation only; no runtime dependency on policy details.
from src.utils.pendulum import PendulumCfg


def rollout(
    sampler: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    F: Callable[[np.ndarray, np.ndarray, "PendulumCfg"], np.ndarray],
    sample_mu0: Callable[[int, "PendulumCfg", np.random.Generator], np.ndarray],
    T: int,
    N: int,
    env_cfg: "PendulumCfg",
    gen: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    vectorized rollout of N trajectories for T+1 timesteps under a sampler policy.

    procedure:
        (1) sample initial states s_0 ~ mu_0: [N, 2]
        (2) allocate state_buf [N, T+1, 2] and action_buf [N, T+1, 1]
        (3) for t in range(T):
                a = sampler(s, gen)  # [N], unclipped
                action_buf[:, t, 0] = a
                s = F(s, a, env_cfg)  # [N, 2]; F clips a + wraps theta internally
                state_buf[:, t+1, :] = s
        (4) a = sampler(s, gen)  # [N]
            action_buf[:, T, 0] = a

    returns:
        (states: [N, T+1, 2], actions: [N, T+1, 1]) both float64

    inputs:
        sampler: (s: [N, 2], gen) -> a: [N]. unclipped scalar actions.
        F: (s: [N, 2], a: [N], cfg) -> s_next: [N, 2]. clips a internally.
        sample_mu0: (N, cfg, gen) -> s_0: [N, 2].
        T: int, horizon. loop runs T iterations; output has T+1 timesteps.
        N: int, batch size.
        env_cfg: PendulumCfg instance.
        gen: np.random.Generator.
    """
    # allocate buffers once before loop
    state_buf = np.zeros((N, T + 1, 2), dtype=np.float64)
    action_buf = np.zeros((N, T + 1, 1), dtype=np.float64)

    # sample initial states and store at t=0
    s = sample_mu0(N, env_cfg, gen)  # [N, 2]
    state_buf[:, 0, :] = s

    # rollout: loop t=0 to t=T-1 (T iterations)
    for t in range(T):
        a = sampler(s, gen)  # [N], unclipped
        action_buf[:, t, 0] = a
        s = F(s, a, env_cfg)  # [N, 2]; F clips internally
        state_buf[:, t + 1, :] = s  # [N, 2]

    # sample final action a_T
    a = sampler(s, gen)  # [N]
    action_buf[:, T, 0] = a

    return state_buf, action_buf


def log_density(
    states: np.ndarray,
    actions: np.ndarray,
    logprob: Callable[[np.ndarray, np.ndarray], np.ndarray],
    log_mu0: Callable[[np.ndarray, "PendulumCfg"], np.ndarray],
    env_cfg: "PendulumCfg",
) -> np.ndarray:
    """
    closed-form log-density of trajectories under a policy.

    log p(tau) = log mu_0(s_0) + sum_{t=0}^{T} log pi(a_t | s_t)

    procedure:
        (1) log_p_s0 = log_mu0(states[:, 0, :], env_cfg)  # [N]
        (2) actions_squeezed = actions[..., 0]  # [N, T+1] (remove trailing-1 axis)
        (3) log_p_a = logprob(actions_squeezed, states)  # [N, T+1]
        (4) log_p = log_p_s0 + log_p_a.sum(axis=-1)  # [N]

    returns:
        log_p: [N] float64. trajectory log-density per sample.

    inputs:
        states: [N, T+1, 2] float64. full state trajectory.
        actions: [N, T+1, 1] float64. action trajectory (unclipped).
        logprob: (a: [N, T+1], s: [N, T+1, 2]) -> log_p: [N, T+1].
                 per-step log-probability; policy's logprob callable.
                 must handle batched-with-time shape.
        log_mu0: (s: [N, 2], cfg) -> log_p: [N]. log-density of initial state.
        env_cfg: PendulumCfg instance.
    """
    # log-density of initial state
    log_p_s0 = log_mu0(states[:, 0, :], env_cfg)  # [N]

    # squeeze actions from [N, T+1, 1] to [N, T+1]
    actions_squeezed = actions[..., 0]

    # per-step log-probability
    log_p_a = logprob(actions_squeezed, states)  # [N, T+1]

    # trajectory log-density: initial + sum of per-step
    log_p = log_p_s0 + log_p_a.sum(axis=-1)  # [N]

    return log_p


def traj_kl_mc(
    p_sampler: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    p_logprob: Callable[[np.ndarray, np.ndarray], np.ndarray],
    q_logprob: Callable[[np.ndarray, np.ndarray], np.ndarray],
    F: Callable[[np.ndarray, np.ndarray, "PendulumCfg"], np.ndarray],
    sample_mu0: Callable[[int, "PendulumCfg", np.random.Generator], np.ndarray],
    log_mu0: Callable[[np.ndarray, "PendulumCfg"], np.ndarray],
    T: int,
    N_mc: int,
    env_cfg: "PendulumCfg",
    gen: np.random.Generator,
    kl_se_warn_threshold: float = 0.1,
) -> dict:
    """
    trajectory-level KL divergence estimation via direct Monte Carlo.

    KL(p || q) = E_{tau ~ p}[ log p(tau) - log q(tau) ]
              ≈ (1/N_mc) sum_i [ log p(tau_i) - log q(tau_i) ],   tau_i ~ p

    trajectories are sampled from p; the estimator is a plain MC mean of
    per-trajectory log-ratios. there is NO importance weight; do NOT compute
    IS-ESS.

    procedure:
        (1) rollout N_mc trajectories under p: states, actions = rollout(p_sampler, ...)
        (2) log_p = log_density(states, actions, p_logprob, log_mu0, cfg)  # [N_mc]
        (3) log_q = log_density(states, actions, q_logprob, log_mu0, cfg)  # [N_mc]
        (4) log_ratio = log_p - log_q  # [N_mc]
        (5) kl_hat = log_ratio.mean()  # scalar
        (6) kl_se = log_ratio.std(ddof=1) / sqrt(N_mc)  # per-trajectory SE
        (7) relative SE diagnostic: rel_se = kl_se / max(abs(kl_hat), 1e-12).
            if rel_se > kl_se_warn_threshold: warnings.warn(...) with context.

    returns:
        dict with keys:
            - 'kl_hat': float. mean log-ratio (KL estimate).
            - 'kl_se': float. per-trajectory standard error.
            - 'log_ratio_samples': [N_mc] np.ndarray. raw log-ratio samples (for reuse).

    inputs:
        p_sampler: (s: [N, 2], gen) -> a: [N]. policy p's sampler.
        p_logprob: (a: [N, T+1], s: [N, T+1, 2]) -> log_p: [N, T+1]. policy p's log-prob.
        q_logprob: (a: [N, T+1], s: [N, T+1, 2]) -> log_q: [N, T+1]. policy q's log-prob.
        F: dynamics function (same as rollout).
        sample_mu0: initial state sampler (same as rollout).
        log_mu0: initial state log-density (same as log_density).
        T: int, horizon.
        N_mc: int, number of MC samples (trajectories to rollout).
        env_cfg: PendulumCfg instance.
        gen: np.random.Generator.
        kl_se_warn_threshold: float >= 0. emit warning if rel_se > threshold.
    """
    # rollout N_mc trajectories under p
    states, actions = rollout(p_sampler, F, sample_mu0, T, N_mc, env_cfg, gen)

    # log-density under p and q
    log_p = log_density(states, actions, p_logprob, log_mu0, env_cfg)  # [N_mc]
    log_q = log_density(states, actions, q_logprob, log_mu0, env_cfg)  # [N_mc]

    # per-trajectory log-ratio
    log_ratio = log_p - log_q  # [N_mc]

    # KL estimate: mean log-ratio
    kl_hat = log_ratio.mean()

    # per-trajectory standard error (not per-step)
    # SE of trajectory-level sum, not step-level
    kl_se = log_ratio.std(ddof=1) / np.sqrt(N_mc)

    # relative SE diagnostic: warn if high
    rel_se = kl_se / np.maximum(np.abs(kl_hat), 1e-12)
    if rel_se > kl_se_warn_threshold:
        warnings.warn(
            f"high relative SE in KL estimate: kl_se / |kl_hat| = {rel_se:.4f} > {kl_se_warn_threshold}",
            stacklevel=2,
        )

    return {
        "kl_hat": float(kl_hat),
        "kl_se": float(kl_se),
        "log_ratio_samples": log_ratio,
    }


def pack(
    states: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """
    concatenate states and actions along the feature axis (no flattening).

    procedure:
        (1) concatenate states [N, T+1, 2] and actions [N, T+1, 1] along last axis
            -> traj: [N, T+1, 3]
        (2) cast to float32 for hdf5-friendly downstream consumers.

    returns:
        traj: [N, T+1, 3] float32

    inputs:
        states: [N, T+1, 2] float64.
        actions: [N, T+1, 1] float64.

    note on flattening: if a downstream consumer wants a flat [N, (T+1)*3] vector,
    they call pack(states, actions).reshape(N, -1) themselves. pack does NOT flatten.
    """
    traj = np.concatenate([states, actions], axis=-1)  # [N, T+1, 3]
    return traj.astype(np.float32)
