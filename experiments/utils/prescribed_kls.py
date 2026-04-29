"""prescribed KL: construct (p, q) pairs achieving prescribed KL divergence.

three flavors share the "prescribed KL" idiom:
  1. gaussian (analytic): closed-form construction via Lambert W.
  2. occupancy (SMODICE/gridworld): Bellman occupancy grid with inversion.
  3. trajectory (pendulum): Monte-Carlo rollouts with grid interpolation.

each flavor follows the same workflow:
  - define a parametric policy family (alpha, beta, etc.).
  - precompute KL divergence grids over the parameter space.
  - invert prescribed target KL values to recover the parameters.
"""

import numpy as np
import torch
from torch import logdet, trace
from scipy.special import lambertw
import h5py
import hashlib
import json
import warnings
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple

from src.utils.occupancy import bellman_occupancy, kl_occupancy, mixture_policy
from src.utils.gridworld import value_iteration, softmax_policy, build_gridworld, reward_to_goal
from src.utils.pendulum_q import load_or_build_q
from src.utils.pendulum_policies import GaussPolicy, MixPolicy
from src.sampling.pendulum_traj import traj_kl_mc
from src.utils.io import _write_hdf5_atomic


# === core inversion helpers (shared by occupancy + trajectory) ===

def assert_monotone(table: np.ndarray, axis: int = -1, kind: str = "increasing") -> None:
    """
    assert that table is strictly monotone along axis.

    raises ValueError with index of first violation if monotonicity is broken.

    args:
        table: array of any shape.
        axis: axis along which to check (default -1 = last axis).
        kind: "increasing" or "decreasing".

    raises:
        ValueError: with message like "table not strictly increasing at indices [i, i+1]".
    """
    # tolerance for "ties": small numerical noise should not flag as a violation,
    # and exact-zero plateaus (e.g. alpha=0 where d_O == d_E) are acceptable for
    # np.interp inversion. enforce *non-strict* monotonicity.
    diffs = np.diff(table, axis=axis)
    tol = 1e-9 * max(1.0, np.abs(table).max())

    if kind == "increasing":
        violation_mask = diffs < -tol
    elif kind == "decreasing":
        violation_mask = diffs > tol
    else:
        raise ValueError(f"kind must be 'increasing' or 'decreasing', got {kind}")

    if np.any(violation_mask):
        # find first violation along the specified axis
        violation_indices = np.where(violation_mask)
        first_idx = violation_indices[axis][0]
        raise ValueError(
            f"table not strictly {kind} at indices [{first_idx}, {first_idx + 1}]"
        )


def prescribe(
    KL1: np.ndarray,
    KL2: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    K1: float,
    K2: float,
) -> Dict[str, Any]:
    """
    invert prescribed (K1, K2) targets to (alpha*, beta*) via monotone tables.

    workflow:
      step 6a (invert K1):
        - assert KL1 is strictly increasing.
        - check if K1 is in [KL1[0], KL1[-1]].
        - invert via np.interp(K1, KL1, alphas).

      step 6b (invert K2 at alpha*):
        - locate alpha* in alphas via np.searchsorted.
        - linearly blend KL2 along alpha (5-line formula).
        - assert blended KL2_slice is decreasing.
        - check if K2 is in the blended range.
        - invert via np.interp(K2, KL2_slice[::-1], betas[::-1]).

      step 6c (realize values):
        - re-evaluate KL1 and KL2 at (alpha*, beta*).

    args:
        KL1: [G_alpha], monotone increasing.
        KL2: [G_alpha, G_beta], monotone decreasing in beta.
        alphas: [G_alpha], strictly increasing.
        betas: [G_beta], strictly increasing.
        K1, K2: target KL values.

    returns:
        dict with keys: alpha_star, beta_star, realized_K1, realized_K2, feasible, reason.
    """
    # step 6a: invert K1
    assert_monotone(KL1, axis=0, kind="increasing")

    if K1 < KL1[0] or K1 > KL1[-1]:
        return {
            "alpha_star": None,
            "beta_star": None,
            "realized_K1": None,
            "realized_K2": None,
            "feasible": False,
            "reason": f"K1={K1} outside [{KL1[0]:.6f}, {KL1[-1]:.6f}]",
        }

    alpha_star = np.interp(K1, KL1, alphas)  # scalar in [0, 1]

    # step 6b: invert K2 at alpha_star via 2-D blend
    # locate alpha_star in the grid and blend KL2 along alpha
    i_low = np.searchsorted(alphas, alpha_star) - 1
    i_low = np.clip(i_low, 0, len(alphas) - 2)
    i_high = i_low + 1

    # 5-line blend formula
    if i_low == len(alphas) - 1:
        # at or past the last grid point
        KL2_slice = KL2[i_low, :]
    else:
        w = (alpha_star - alphas[i_low]) / (alphas[i_low + 1] - alphas[i_low])
        KL2_slice = (1 - w) * KL2[i_low, :] + w * KL2[i_high, :]

    # assert that the blended slice is decreasing in beta
    assert_monotone(KL2_slice, axis=0, kind="decreasing")

    # check K2 feasibility
    # KL2_slice decreases from KL2_slice[0] to KL2_slice[-1]
    if K2 < KL2_slice[-1] or K2 > KL2_slice[0]:
        return {
            "alpha_star": alpha_star,
            "beta_star": None,
            "realized_K1": None,
            "realized_K2": None,
            "feasible": False,
            "reason": f"K2={K2} outside [{KL2_slice[-1]:.6f}, {KL2_slice[0]:.6f}]",
        }

    # invert K2 using reversed tables for decreasing function
    beta_star = np.interp(K2, KL2_slice[::-1], betas[::-1])

    # step 6c: realize values
    realized_K1 = np.interp(alpha_star, alphas, KL1)

    # re-blend KL2 for realized_K2
    if i_low == len(alphas) - 1:
        KL2_slice_realized = KL2[i_low, :]
    else:
        w = (alpha_star - alphas[i_low]) / (alphas[i_low + 1] - alphas[i_low])
        KL2_slice_realized = (1 - w) * KL2[i_low, :] + w * KL2[i_high, :]
    realized_K2 = np.interp(beta_star, betas, KL2_slice_realized)

    return {
        "alpha_star": float(alpha_star),
        "beta_star": float(beta_star),
        "realized_K1": float(realized_K1),
        "realized_K2": float(realized_K2),
        "feasible": True,
        "reason": None,
    }


def feasible_region(KL1: np.ndarray, KL2: np.ndarray) -> Dict[str, Any]:
    """
    compute achievable (K1, K2) bounds given the KL grid.

    args:
        KL1: [G_alpha], monotone increasing.
        KL2: [G_alpha, G_beta], monotone decreasing in beta at each alpha.

    returns:
        dict with K1_min, K1_max, K2_min_per_alpha, K2_max_per_alpha.
    """
    K1_min = float(np.min(KL1))
    K1_max = float(np.max(KL1))

    # for each alpha, KL2 is decreasing in beta
    # so min is at the end, max is at the beginning
    K2_min_per_alpha = np.min(KL2, axis=1)  # [G_alpha]
    K2_max_per_alpha = np.max(KL2, axis=1)  # [G_alpha]

    return {
        "K1_min": K1_min,
        "K1_max": K1_max,
        "K2_min_per_alpha": K2_min_per_alpha,
        "K2_max_per_alpha": K2_max_per_alpha,
    }


# === gaussian (analytic) ===

def compute_gaussian_kl_divergence(
    mu0: torch.Tensor,
    Sigma0: torch.Tensor,
    mu1: torch.Tensor,
    Sigma1: torch.Tensor
) -> torch.Tensor:
    dim = mu0.shape[0]
    mean_term = 0.5 * ((mu1 - mu0).T @ torch.linalg.inv(Sigma1) @ (mu1 - mu0))
    cov_term = 0.5 * (trace(Sigma0 @ torch.linalg.inv(Sigma1))  - dim + logdet(Sigma1) - logdet(Sigma0))

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
    alpha = -np.real(lambertw(-np.exp(-c), k=-1))
    # solve for delta in terms of alpha and k1
    delta = 2 * k1

    mu0 = torch.zeros(dim)
    Sigma0 = torch.Tensor(alpha * np.eye(dim))
    mu1 = torch.Tensor(np.sqrt(delta) * (np.ones(dim) / np.sqrt(dim)))
    Sigma1 = torch.eye(dim)

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


# === occupancy (SMODICE / gridworld) ===

def hash_mdp_config(mdp_cfg: Dict[str, Any]) -> str:
    """
    canonical sha256 hash of MDP configuration.

    serializes relevant fields as JSON with sorted keys, then sha256.

    args:
        mdp_cfg: dict with keys L, p_slip, gamma, terminals, expert_goal,
                 anti_goal, tau, G_alpha, G_beta.

    returns:
        first 16 hex characters of sha256 hash.
    """
    canonical = {
        "L": mdp_cfg["L"],
        "p_slip": mdp_cfg["p_slip"],
        "gamma": mdp_cfg["gamma"],
        "terminals": sorted([list(t) for t in mdp_cfg.get("terminals", [])]),
        "expert_goal": list(mdp_cfg.get("expert_goal", [])),
        "anti_goal": list(mdp_cfg.get("anti_goal", [])),
        "tau": mdp_cfg["tau"],
        "G_alpha": mdp_cfg["G_alpha"],
        "G_beta": mdp_cfg["G_beta"],
        "mu0_kind": mdp_cfg.get("mu0_kind", "uniform"),
        "mu0_centers": sorted([list(c) for c in mdp_cfg.get("mu0_centers", [])]),
        "reward_kind": mdp_cfg.get("reward_kind", "sparse"),
        "reward_sigma": float(mdp_cfg.get("reward_sigma", 1.0)),
    }

    s = json.dumps(canonical, sort_keys=True)
    h = hashlib.sha256(s.encode()).hexdigest()
    return h[:16]


def build_kl_grid(
    mdp: "MDP",
    r_E: np.ndarray,
    r_anti: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    tau: float,
) -> Dict[str, Any]:
    """
    precompute 2-D KL-divergence grid for SMODICE policy family.

    workflow:
      1. compute expert policy pi_E from r_E.
      2. for each alpha, interpolate reward and compute occupancy d_O(alpha).
      3. log KL1[alpha] = KL(d_E || d_O(alpha)).
      4. for each beta, mix pi_O and pi_E, log KL2[alpha, beta] = KL(d_mix || d_E).
      5. assert KL1 strictly increasing; KL2 strictly decreasing in beta at each alpha.

    args:
        mdp: gridworld MDP with P [|S|, |A|, |S|], mu0 [|S|], gamma.
        r_E: expert reward [|S|, |A|].
        r_anti: anti-expert reward [|S|, |A|].
        alphas: grid points in [0, 1], strictly increasing.
        betas: grid points in [0, 1], strictly increasing.
        tau: softmax temperature for policy smoothing.

    returns:
        dict with keys: KL1, KL2, alphas, betas, d_E, d_O_grid, pi_E.
    """
    # validate grid size
    if len(alphas) < 2 or len(betas) < 2:
        raise ValueError("grid too coarse: G_alpha and G_beta must be >= 2")

    # validate sorting
    assert np.all(np.diff(alphas) > 0), "alphas must be strictly increasing"
    assert np.all(np.diff(betas) > 0), "betas must be strictly increasing"

    # compute expert policy and occupancy
    Q_E = value_iteration(mdp.P, r_E, mdp.gamma)  # [|S|, |A|]
    pi_E = softmax_policy(Q_E, tau)               # [|S|, |A|]
    d_E = bellman_occupancy(mdp.P, mdp.mu0, pi_E, mdp.gamma)  # [|S|, |A|]

    n_states, n_acts = r_E.shape
    G_alpha = len(alphas)
    G_beta = len(betas)

    # allocate grids
    KL1 = np.zeros(G_alpha)
    KL2 = np.zeros((G_alpha, G_beta))
    d_O_grid = np.zeros((G_alpha, n_states, n_acts))

    # double loop: alpha, then beta
    for i, alpha in enumerate(alphas):
        # interpolate reward
        r_O = (1 - alpha) * r_E + alpha * r_anti  # [|S|, |A|]

        # compute occupancy at this alpha
        Q_O = value_iteration(mdp.P, r_O, mdp.gamma)  # [|S|, |A|]
        pi_O = softmax_policy(Q_O, tau)               # [|S|, |A|]
        d_O = bellman_occupancy(mdp.P, mdp.mu0, pi_O, mdp.gamma)  # [|S|, |A|]

        # log KL(d_E || d_O)
        KL1[i] = kl_occupancy(d_E, d_O)
        d_O_grid[i] = d_O

        # loop over betas
        for j, beta in enumerate(betas):
            pi_mix = mixture_policy(pi_O, pi_E, beta)  # [|S|, |A|]
            d_mix = bellman_occupancy(mdp.P, mdp.mu0, pi_mix, mdp.gamma)  # [|S|, |A|]
            KL2[i, j] = kl_occupancy(d_mix, d_E)

    # assert monotonicity
    assert_monotone(KL1, axis=0, kind="increasing")
    for i in range(G_alpha):
        assert_monotone(KL2[i, :], axis=0, kind="decreasing")

    return {
        "KL1": KL1,
        "KL2": KL2,
        "alphas": alphas,
        "betas": betas,
        "d_E": d_E,
        "d_O_grid": d_O_grid,
        "pi_E": pi_E,
    }


def load_or_build_grid(
    mdp_cfg: Dict[str, Any],
    cache_dir: str,
    rebuild: bool = False,
) -> Dict[str, Any]:
    """
    high-level orchestrator: load or build the KL grid.

    workflow:
      - hash the MDP config.
      - check cache_dir / grid_{hash}.h5.
      - if cache hit and not rebuild: load and return.
      - if cache miss or rebuild: build via build_kl_grid, write HDF5, return.

    args:
        mdp_cfg: dict with MDP shape and policy parameters.
        cache_dir: directory to store cache files (created if missing).
        rebuild: if True, force rebuild even if cache exists.

    returns:
        dict from build_kl_grid PLUS mdp, r_E, r_anti (rebuilt).

    raises:
        RuntimeError: if HDF5 cache is corrupt and rebuild=False.
    """
    # ensure cache_dir exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # compute hash and cache filename
    cfg_hash = hash_mdp_config(mdp_cfg)
    cache_file = Path(cache_dir) / f"grid_{cfg_hash}.h5"

    # always rebuild mdp + rewards (cheap; consumed downstream)
    mdp = build_gridworld(
        L=mdp_cfg["L"],
        p_slip=mdp_cfg["p_slip"],
        terminals=mdp_cfg["terminals"],
        gamma=mdp_cfg["gamma"],
        mu0_kind=mdp_cfg.get("mu0_kind", "uniform"),
        mu0_centers=mdp_cfg.get("mu0_centers"),
    )
    reward_kind = mdp_cfg.get("reward_kind", "sparse")
    reward_sigma = float(mdp_cfg.get("reward_sigma", 1.0))
    from src.utils.gridworld import shaped_reward_to_goal
    r_E = shaped_reward_to_goal(
        mdp_cfg["L"], mdp_cfg["expert_goal"], mdp_cfg["terminals"],
        kind=reward_kind, sigma=reward_sigma,
    )
    r_anti = shaped_reward_to_goal(
        mdp_cfg["L"], mdp_cfg["anti_goal"], mdp_cfg["terminals"],
        kind=reward_kind, sigma=reward_sigma,
    )

    # try to load from cache if not rebuild
    if not rebuild and cache_file.exists():
        try:
            with h5py.File(cache_file, "r") as f:
                result = {
                    "KL1": f["KL1"][:],
                    "KL2": f["KL2"][:],
                    "alphas": f["alphas"][:],
                    "betas": f["betas"][:],
                    "d_E": f["d_E"][:],
                    "d_O_grid": f["d_O_grid"][:],
                    "pi_E": f["pi_E"][:],
                    "mdp": mdp,
                    "r_E": r_E,
                    "r_anti": r_anti,
                }
            return result
        except (KeyError, OSError, h5py.Error) as e:
            # cache corrupt; fall through to rebuild
            print(f"Warning: cache corrupted at {cache_file}, rebuilding: {e}")

    # build the grid
    alphas = np.linspace(0, 1, mdp_cfg["G_alpha"])
    betas = np.linspace(0, 1, mdp_cfg["G_beta"])

    result = build_kl_grid(mdp, r_E, r_anti, alphas, betas, mdp_cfg["tau"])
    result["mdp"] = mdp
    result["r_E"] = r_E
    result["r_anti"] = r_anti

    # write to HDF5
    with h5py.File(cache_file, "w") as f:
        f.create_dataset("KL1", data=result["KL1"])
        f.create_dataset("KL2", data=result["KL2"])
        f.create_dataset("alphas", data=result["alphas"])
        f.create_dataset("betas", data=result["betas"])
        f.create_dataset("d_E", data=result["d_E"])
        f.create_dataset("d_O_grid", data=result["d_O_grid"])
        f.create_dataset("pi_E", data=result["pi_E"])

        # metadata
        f.attrs["hash"] = cfg_hash
        f.attrs["L"] = mdp_cfg["L"]
        f.attrs["p_slip"] = mdp_cfg["p_slip"]
        f.attrs["gamma"] = mdp_cfg["gamma"]
        f.attrs["tau"] = mdp_cfg["tau"]
        f.attrs["G_alpha"] = mdp_cfg["G_alpha"]
        f.attrs["G_beta"] = mdp_cfg["G_beta"]

    return result


# === trajectory (pendulum) ===

def hash_pendulum_cfg(cfg: Dict[str, Any]) -> str:
    """canonical sha256 over pendulum config for cache keying.

    plan:
      1. extract env constants into dict with canonical field order.
      2. convert alphas, betas to lists (JSON serializable).
      3. flatten q_cfg.
      4. create canonical dict with all fields.
      5. JSON serialize with sort_keys=True.
      6. sha256 hash; return first 16 hex chars.

    args:
      cfg: dict with env constants (g, ell, m, dt, action_clip, theta_dot_clip),
           mu0_box, T, sigma_pi, r_E_name, r_anti_name, alphas, betas, M, q_cfg.

    returns:
      str, 16 hex characters (mirrors hash_mdp_config from kl_prescriber.py).
    """
    env_cfg = cfg['env_cfg']
    q_cfg = cfg['q_cfg']

    canonical = {
        'g': float(env_cfg.g),
        'ell': float(env_cfg.ell),
        'm': float(env_cfg.m),
        'dt': float(env_cfg.dt),
        'action_clip': float(env_cfg.action_clip),
        'theta_dot_clip': float(env_cfg.theta_dot_clip),
        'mu0_box': [list(cfg['env_cfg'].mu0_box[0]), list(cfg['env_cfg'].mu0_box[1])],
        'T': int(cfg['T']),
        'sigma_pi': float(cfg['sigma_pi']),
        'r_E_name': str(cfg['r_E_name']),
        'r_anti_name': str(cfg['r_anti_name']),
        'alphas': [float(a) for a in cfg['alphas']],
        'betas': [float(b) for b in cfg['betas']],
        'M': int(cfg['M']),
        'N_theta': int(q_cfg.N_theta),
        'N_theta_dot': int(q_cfg.N_theta_dot),
        'N_action': int(q_cfg.N_action),
        'gamma': float(q_cfg.gamma),
        'fqi_max_iter': int(q_cfg.fqi_max_iter),
        'fqi_tol': float(q_cfg.fqi_tol),
    }

    s = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    h = hashlib.sha256(s.encode()).hexdigest()
    return h[:16]


def build_traj_kl_grid(
    env_cfg: Any,
    q_cfg: Any,
    F: Callable,
    sample_mu0: Callable,
    log_mu0: Callable,
    r_E: Callable,
    r_E_name: str,
    r_anti: Callable,
    r_anti_name: str,
    alphas: np.ndarray,
    betas: np.ndarray,
    sigma_pi: float,
    T: int,
    M: int,
    cache_dir: str,
    gen: np.random.Generator,
    kl_se_warn_threshold: float = 0.1,
) -> Dict[str, Any]:
    """build 2-D KL grid via MC rollouts; return dict with KL1, KL2, SEs, and residuals.

    plan:
      1. load expert q-function and construct expert policy pi_E.
      2. allocate KL1, KL2, SE arrays.
      3. loop over alphas:
         a. use named-factory make_r_O to defeat python late-binding.
         b. load or build q_O; store residual.
         c. seed sub-generator for this alpha row (CRN).
         d. compute KL1[i] = KL(p^pi_E || p^pi_O) via traj_kl_mc.
         e. loop over betas:
            i. construct mixture pi_mix.
            ii. fresh sub-generator per (i, j).
            iii. compute KL2[i, j] via traj_kl_mc.
      4. check monotonicity: KL1 increasing in alpha, KL2 decreasing in beta (per alpha).
      5. return dict with all arrays, residuals, and monotone flags.

    args:
      env_cfg: PendulumCfg.
      q_cfg: QGridCfg.
      F, sample_mu0, log_mu0: dynamics and sampling.
      r_E, r_anti: reward functions.
      r_E_name, r_anti_name: reward identifiers for caching.
      alphas, betas: grid points, strictly increasing in [0, 1].
      sigma_pi: policy std.
      T: trajectory length.
      M: MC rollout count.
      cache_dir: directory for Q-table caching.
      gen: global RNG for seeding sub-generators.
      kl_se_warn_threshold: warning threshold for relative SE in KL estimates.

    returns:
      dict with KL1, KL2, KL1_se, KL2_se, alphas, betas, q_E, q_E_residual,
      q_O_grid, q_O_residuals, monotone_alpha, monotone_beta_per_alpha.
    """
    # load expert q-function
    q_E_result = load_or_build_q(
        env_cfg, r_E, r_E_name, q_cfg, cache_dir,
        F=F, alpha=None, r_E_name=None, r_anti_name=None
    )
    q_E = q_E_result['Q']  # [N_theta, N_theta_dot, N_action]
    q_E_residual = float(q_E_result['bellman_residual'])
    pi_E = GaussPolicy(q_E, sigma_pi, env_cfg, q_cfg)

    # allocate grids
    G_alpha = len(alphas)
    G_beta = len(betas)
    N_theta, N_theta_dot, N_action = q_E.shape

    KL1 = np.zeros(G_alpha)  # [G_alpha]
    KL2 = np.zeros((G_alpha, G_beta))  # [G_alpha, G_beta]
    KL1_se = np.zeros(G_alpha)  # [G_alpha]
    KL2_se = np.zeros((G_alpha, G_beta))  # [G_alpha, G_beta]
    q_O_grid = np.zeros((G_alpha, N_theta, N_theta_dot, N_action))  # [G_alpha, ...]
    q_O_residuals = np.zeros(G_alpha)  # [G_alpha]

    # named factory: defeats python late-binding of alpha in closures
    def make_r_O(alpha_val: float, r_E_fn: Callable, r_anti_fn: Callable) -> Callable:
        """construct r_O closure for a specific alpha value."""
        def r_O(s: np.ndarray, a: np.ndarray, s_next: np.ndarray, cfg: Any) -> np.ndarray:
            return (1.0 - alpha_val) * r_E_fn(s, a, s_next, cfg) + alpha_val * r_anti_fn(s, a, s_next, cfg)
        return r_O

    # loop over alphas
    for i, alpha in enumerate(alphas):
        r_O = make_r_O(float(alpha), r_E, r_anti)

        # load or build q_O; pass r_E_name + r_anti_name to avoid cache collision
        q_O_result = load_or_build_q(
            env_cfg, r_O, 'r_O', q_cfg, cache_dir,
            F=F, alpha=float(alpha), r_E_name=r_E_name, r_anti_name=r_anti_name
        )
        q_O = q_O_result['Q']  # [N_theta, N_theta_dot, N_action]
        q_O_grid[i] = q_O
        q_O_residuals[i] = float(q_O_result['bellman_residual'])

        # construct policy for this alpha
        pi_O = GaussPolicy(q_O, sigma_pi, env_cfg, q_cfg)

        # CRN: sub-generator for this alpha row
        alpha_seed = gen.integers(0, 2**63 - 1)
        alpha_gen = np.random.default_rng(alpha_seed)

        # KL1[i] = KL(p^pi_E || p^pi_O) under rollouts from pi_E
        res_KL1 = traj_kl_mc(
            pi_E.sample, pi_E.log_prob, pi_O.log_prob,
            F, sample_mu0, log_mu0, T, M, env_cfg, alpha_gen,
            kl_se_warn_threshold=kl_se_warn_threshold,
        )
        KL1[i] = res_KL1['kl_hat']
        KL1_se[i] = res_KL1['kl_se']

        # loop over betas
        for j, beta in enumerate(betas):
            pi_mix = MixPolicy(pi_O, pi_E, float(beta))

            # CRN: fresh sub-generator per (i, j)
            beta_seed = alpha_gen.integers(0, 2**63 - 1)
            beta_gen = np.random.default_rng(beta_seed)

            # KL2[i, j] = KL(p^pi_mix || p^pi_E) under rollouts from pi_mix
            res_KL2 = traj_kl_mc(
                pi_mix.sample, pi_mix.log_prob, pi_E.log_prob,
                F, sample_mu0, log_mu0, T, M, env_cfg, beta_gen,
                kl_se_warn_threshold=kl_se_warn_threshold,
            )
            KL2[i, j] = res_KL2['kl_hat']
            KL2_se[i, j] = res_KL2['kl_se']

    # check monotonicity
    monotone_alpha = True
    try:
        assert_monotone(KL1, axis=0, kind='increasing')
    except ValueError:
        monotone_alpha = False
        warnings.warn('KL1 non-monotone in alpha')

    monotone_beta_per_alpha = np.zeros(G_alpha, dtype=bool)
    for i in range(G_alpha):
        try:
            assert_monotone(KL2[i, :], axis=0, kind='decreasing')
            monotone_beta_per_alpha[i] = True
        except ValueError:
            monotone_beta_per_alpha[i] = False
            warnings.warn(f'KL2[{i}, :] non-monotone in beta')

    return {
        'KL1': KL1,
        'KL2': KL2,
        'KL1_se': KL1_se,
        'KL2_se': KL2_se,
        'alphas': alphas,
        'betas': betas,
        'q_E': q_E,
        'q_E_residual': q_E_residual,
        'q_O_grid': q_O_grid,
        'q_O_residuals': q_O_residuals,
        'monotone_alpha': monotone_alpha,
        'monotone_beta_per_alpha': monotone_beta_per_alpha,
    }


def prescribe_traj(grid: Dict[str, Any], K1: float, K2: float) -> Dict[str, Any]:
    """invert prescribed (K1, K2) targets via monotone bisection or grid-argmin fallback.

    plan:
      1. check monotonicity flags.
      2. if all monotone: delegate to existing prescribe() function; snap alpha to grid.
      3. else: grid-argmin fallback; find closest alpha to K1, then closest beta at that alpha to K2.
      4. compute tolerance as 0.5 * mean grid step.
      5. check feasibility: achieved values within tolerance of targets.
      6. return dict with alpha_star, beta_star, realized_K1, realized_K2, feasible, reason, i_snap.

    args:
      grid: output of build_traj_kl_grid.
      K1, K2: target KL values.

    returns:
      dict with alpha_star, beta_star, realized_K1, realized_K2, feasible, reason, i_snap.
    """
    # check monotonicity
    all_monotone = (
        grid['monotone_alpha'] and
        np.all(grid['monotone_beta_per_alpha'])
    )

    if all_monotone:
        # use bisection
        result = prescribe(
            grid['KL1'], grid['KL2'],
            grid['alphas'], grid['betas'],
            K1, K2
        )
        # snap alpha to grid index
        i_snap = int(np.argmin(np.abs(grid['alphas'] - result['alpha_star'])))
        result['i_snap'] = i_snap
        return result

    else:
        # grid-argmin fallback
        # find closest alpha to K1
        i_star = int(np.argmin(np.abs(grid['KL1'] - K1)))
        alpha_star = float(grid['alphas'][i_star])
        realized_K1 = float(grid['KL1'][i_star])

        # find closest beta at that alpha to K2
        j_star = int(np.argmin(np.abs(grid['KL2'][i_star, :] - K2)))
        beta_star = float(grid['betas'][j_star])
        realized_K2 = float(grid['KL2'][i_star, j_star])

        # compute tolerance: mean step in grids
        k1_step = np.mean(np.diff(grid['KL1']))
        k2_step = np.mean(np.diff(grid['KL2'][i_star, :]))
        tol_k1 = 0.5 * k1_step
        tol_k2 = 0.5 * k2_step

        # check feasibility
        feasible = (
            abs(realized_K1 - K1) <= tol_k1 and
            abs(realized_K2 - K2) <= tol_k2
        )
        reason = None
        if not feasible:
            reason = (
                f"non-monotone fallback: got K1={realized_K1:.4f} "
                f"(target {K1:.4f}, diff {abs(realized_K1 - K1):.4f}), "
                f"K2={realized_K2:.4f} (target {K2:.4f}, diff {abs(realized_K2 - K2):.4f})"
            )

        result = {
            'alpha_star': alpha_star,
            'beta_star': beta_star,
            'realized_K1': realized_K1,
            'realized_K2': realized_K2,
            'feasible': feasible,
            'reason': reason,
            'i_snap': int(i_star),
        }
        return result


def load_or_build_traj_grid(
    cfg: Dict[str, Any],
    F: Callable,
    sample_mu0: Callable,
    log_mu0: Callable,
    r_E: Callable,
    r_anti: Callable,
    cache_dir: str,
    rebuild: bool = False
) -> Dict[str, Any]:
    """high-level orchestrator: load or build trajectory KL grid; cache to HDF5.

    plan:
      1. ensure cache_dir exists.
      2. compute hash; construct cache filename.
      3. try to load from cache if not rebuild.
      4. (optional) build diagnostic coarse grid (10x10) if requested.
      5. build full grid.
      6. log feasible region.
      7. write to HDF5 atomically.
      8. return result dict.

    args:
      cfg: dict with env_cfg, q_cfg, T, sigma_pi, M, alphas, betas,
           r_E_name, r_anti_name, traj_kl_grid (sub-dict with diagnostic_grid, kl_se_warn_threshold).
      F, sample_mu0, log_mu0, r_E, r_anti: dynamics, sampling, rewards.
      cache_dir: directory for Q-table caching (created if missing).
      rebuild: if True, force rebuild even if cache exists.

    returns:
      dict from build_traj_kl_grid (KL1, KL2, ..., monotone_*).
    """
    # ensure cache_dir exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # compute hash and cache filename
    cfg_hash = hash_pendulum_cfg(cfg)
    cache_file = Path(cache_dir) / f'traj_grid_{cfg_hash}.h5'

    # try to load from cache
    if not rebuild and cache_file.exists():
        try:
            with h5py.File(cache_file, 'r') as f:
                result = {
                    'KL1': f['KL1'][:],
                    'KL2': f['KL2'][:],
                    'KL1_se': f['KL1_se'][:],
                    'KL2_se': f['KL2_se'][:],
                    'alphas': f['alphas'][:],
                    'betas': f['betas'][:],
                    'q_E': f['q_E'][:],
                    'q_E_residual': float(f.attrs['q_E_residual']),
                    'q_O_grid': f['q_O_grid'][:],
                    'q_O_residuals': f['q_O_residuals'][:],
                    'monotone_alpha': bool(f.attrs['monotone_alpha']),
                    'monotone_beta_per_alpha': f['monotone_beta_per_alpha'][:],
                }
            return result
        except (KeyError, OSError, h5py.Error) as e:
            warnings.warn(f'cache corrupted at {cache_file}, rebuilding: {e}')

    # (optional) diagnostic grid
    kl_se_threshold = cfg.get('traj_kl_grid', {}).get('kl_se_warn_threshold', 0.1)

    if cfg.get('traj_kl_grid', {}).get('diagnostic_grid', False):
        alphas_diag = np.linspace(0, 1, 10)
        betas_diag = np.linspace(0, 1, 10)
        seed_diag = 42
        gen_diag = np.random.default_rng(seed_diag)
        result_diag = build_traj_kl_grid(
            cfg['env_cfg'], cfg['q_cfg'], F, sample_mu0, log_mu0,
            r_E, cfg['r_E_name'], r_anti, cfg['r_anti_name'],
            alphas_diag, betas_diag, cfg['sigma_pi'], cfg['T'], cfg['M'],
            cache_dir, gen_diag, kl_se_warn_threshold=kl_se_threshold
        )
        print(
            f"diagnostic grid built; monotone_alpha={result_diag['monotone_alpha']}, "
            f"monotone_beta_per_alpha={result_diag['monotone_beta_per_alpha']}"
        )
        if not result_diag['monotone_alpha'] or not np.all(result_diag['monotone_beta_per_alpha']):
            warnings.warn('diagnostic grid non-monotone; check reward functions')

    # build full grid
    gen = np.random.default_rng(cfg.get('seed', 1729))
    result = build_traj_kl_grid(
        cfg['env_cfg'], cfg['q_cfg'], F, sample_mu0, log_mu0,
        r_E, cfg['r_E_name'], r_anti, cfg['r_anti_name'],
        cfg['alphas'], cfg['betas'], cfg['sigma_pi'], cfg['T'], cfg['M'],
        cache_dir, gen, kl_se_warn_threshold=kl_se_threshold
    )

    # log feasible region
    region = feasible_region(result['KL1'], result['KL2'])
    print(
        f"feasible (K1, K2) region:\n"
        f"  K1 in [{region['K1_min']:.4f}, {region['K1_max']:.4f}]\n"
        f"  K2 per alpha:"
    )
    for i, alpha in enumerate(cfg['alphas']):
        print(
            f"    alpha={alpha:.4f}: K2 in [{region['K2_min_per_alpha'][i]:.4f}, "
            f"{region['K2_max_per_alpha'][i]:.4f}]"
        )
    print("use these bounds to populate config.kl_targets")

    # write to HDF5 atomically. cfg may pass alphas/betas as python lists
    # (step1 calls .tolist() before constructing cfg), so coerce to arrays.
    datasets = {
        'KL1': np.asarray(result['KL1']),
        'KL2': np.asarray(result['KL2']),
        'KL1_se': np.asarray(result['KL1_se']),
        'KL2_se': np.asarray(result['KL2_se']),
        'alphas': np.asarray(result['alphas']),
        'betas': np.asarray(result['betas']),
        'q_E': np.asarray(result['q_E']),
        'q_O_grid': np.asarray(result['q_O_grid']),
        'q_O_residuals': np.asarray(result['q_O_residuals']),
        'monotone_beta_per_alpha': np.asarray(result['monotone_beta_per_alpha']).astype(np.uint8),
    }

    attrs = {
        'hash': cfg_hash,
        'g': float(cfg['env_cfg'].g),
        'ell': float(cfg['env_cfg'].ell),
        'm': float(cfg['env_cfg'].m),
        'dt': float(cfg['env_cfg'].dt),
        'action_clip': float(cfg['env_cfg'].action_clip),
        'theta_dot_clip': float(cfg['env_cfg'].theta_dot_clip),
        'T': int(cfg['T']),
        'sigma_pi': float(cfg['sigma_pi']),
        'M': int(cfg['M']),
        'G_alpha': len(result['alphas']),
        'G_beta': len(result['betas']),
        'monotone_alpha': result['monotone_alpha'],
        'r_E_name': cfg['r_E_name'],
        'r_anti_name': cfg['r_anti_name'],
        'q_E_residual': float(result['q_E_residual']),
    }

    _write_hdf5_atomic(str(cache_file), datasets, attrs)

    return result


if __name__ == '__main__':
    gaussian_pair = create_two_gaussians_kl(dim=3, k=256, beta=0.8)
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
