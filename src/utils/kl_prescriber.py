"""
kl_prescriber: precompute KL divergence grid and invert prescribed targets.

workflow:
  1. build_kl_grid: 2-D grid of KL(d_E || d_O(alpha)) and KL(d_mix || d_E).
  2. assert_monotone: guard that KL1 is increasing, KL2 decreasing in beta.
  3. prescribe: invert (K1, K2) targets to (alpha*, beta*) via np.interp + 2-D blend.
  4. feasible_region: bounds on achievable (K1, K2).
  5. hash_mdp_config: deterministic sha256 of MDP config.
  6. load_or_build_grid: HDF5 cache orchestration.
"""

import numpy as np
import h5py
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

from src.utils.occupancy import bellman_occupancy, kl_occupancy, mixture_policy
from src.utils.gridworld import value_iteration, softmax_policy


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
    diffs = np.diff(table, axis=axis)

    if kind == "increasing":
        violation_mask = diffs <= 0
    elif kind == "decreasing":
        violation_mask = diffs >= 0
    else:
        raise ValueError(f"kind must be 'increasing' or 'decreasing', got {kind}")

    if np.any(violation_mask):
        # find first violation along the specified axis
        violation_indices = np.where(violation_mask)
        first_idx = violation_indices[axis][0]
        raise ValueError(
            f"table not strictly {kind} at indices [{first_idx}, {first_idx + 1}]"
        )


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
    }

    s = json.dumps(canonical, sort_keys=True)
    h = hashlib.sha256(s.encode()).hexdigest()
    return h[:16]


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
    from src.utils.gridworld import build_gridworld, reward_to_goal

    mdp = build_gridworld(
        L=mdp_cfg["L"],
        p_slip=mdp_cfg["p_slip"],
        terminals=mdp_cfg["terminals"],
        gamma=mdp_cfg["gamma"],
    )
    r_E = reward_to_goal(mdp_cfg["L"], mdp_cfg["expert_goal"], mdp_cfg["terminals"])
    r_anti = reward_to_goal(mdp_cfg["L"], mdp_cfg["anti_goal"], mdp_cfg["terminals"])

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
