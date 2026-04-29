"""
tile-coded fitted-Q-iteration on 3-D Q-table over (theta, theta_dot, action).
provides fast vectorized lookup and vectorized argmax_a. disk-cached per
(env constants, reward, q grid cfg, alpha if reward is interpolated).
"""

import numpy as np
from scipy.ndimage import map_coordinates
import h5py
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Any, Optional


@dataclass(frozen=True)
class QGridCfg:
    """
    Q-table discretization and fitted-Q-iteration parameters.

    fields:
        N_theta: int = 128. number of theta grid points; linspace(-pi, pi, N_theta, endpoint=False).
        N_theta_dot: int = 128. number of theta_dot grid points; linspace(-clip, clip, N_theta_dot).
        N_action: int = 21. number of action grid points; linspace(-clip, clip, N_action).
        gamma: float = 0.99. discount factor.
        fqi_max_iter: int = 1000. max bellman backup passes.
        fqi_tol: float = 1e-4. convergence tolerance on bellman residual.
    """
    N_theta: int = 128
    N_theta_dot: int = 128
    N_action: int = 21
    gamma: float = 0.99
    fqi_max_iter: int = 1000
    fqi_tol: float = 1e-4


def _axes(env_cfg: Any, q_cfg: QGridCfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    construct grid axes for tile-coding lookup.

    pseudocode:
        theta_grid = linspace(-pi, pi, N_theta, endpoint=False).
        thdot_grid = linspace(-theta_dot_clip, theta_dot_clip, N_theta_dot).
        a_grid = linspace(-action_clip, action_clip, N_action).

    args:
        env_cfg: environment config with theta_dot_clip, action_clip.
        q_cfg: QGridCfg with N_theta, N_theta_dot, N_action.

    returns:
        (theta_grid, thdot_grid, a_grid), each shape (N,) float64.
    """
    theta_grid = np.linspace(-np.pi, np.pi, q_cfg.N_theta, endpoint=False, dtype=np.float64)
    thdot_grid = np.linspace(-env_cfg.theta_dot_clip, env_cfg.theta_dot_clip, q_cfg.N_theta_dot, dtype=np.float64)
    a_grid = np.linspace(-env_cfg.action_clip, env_cfg.action_clip, q_cfg.N_action, dtype=np.float64)

    return theta_grid, thdot_grid, a_grid


def q_lookup(
    Q: np.ndarray, s: np.ndarray, a: np.ndarray,
    env_cfg: Any, q_cfg: QGridCfg
) -> np.ndarray:
    """
    trilinear lookup on Q-table via scipy.ndimage.map_coordinates.

    affine index-space transforms:
        theta in [-pi, pi) maps to [0, N_theta):
            i_theta = (theta + pi) / (2*pi) * N_theta.
        theta_dot in [-clip, clip] maps to [0, N_theta_dot - 1]:
            i_thdot = (theta_dot + clip) / (2*clip) * (N_theta_dot - 1).
        a in [-clip, clip] maps to [0, N_action - 1]:
            i_a = (a + clip) / (2*clip) * (N_action - 1).

    vectorized lookup:
        1. flatten s to [...,2], compute i_theta, i_thdot (broadcast together).
        2. flatten a to [...], compute i_a.
        3. stack [i_theta.ravel(), i_thdot.ravel(), i_a.ravel()] -> [3, total_size].
        4. call map_coordinates(Q, coords, order=1, mode='nearest').
        5. reshape back to s.shape[:-1].

    args:
        Q: [N_theta, N_theta_dot, N_action] float64. Q-values.
        s: [..., 2] float64. continuous states; s[..., 0] = theta in [-pi, pi), s[..., 1] = theta_dot.
        a: [...] float64. continuous actions in [-action_clip, action_clip].
        env_cfg: provides action_clip, theta_dot_clip.
        q_cfg: provides N_theta, N_theta_dot, N_action.

    returns:
        float64 array, shape s.shape[:-1] (state leading dims).
    """
    # affine transforms: state and action to index space
    # theta: [...] in [-pi, pi) -> [0, N_theta)
    i_theta = (s[..., 0] + np.pi) / (2.0 * np.pi) * q_cfg.N_theta
    # theta_dot: [...] in [-clip, clip] -> [0, N_theta_dot - 1]
    i_thdot = (s[..., 1] + env_cfg.theta_dot_clip) / (2.0 * env_cfg.theta_dot_clip) * (q_cfg.N_theta_dot - 1)
    # action: [...] in [-clip, clip] -> [0, N_action - 1]
    i_a = (a + env_cfg.action_clip) / (2.0 * env_cfg.action_clip) * (q_cfg.N_action - 1)

    # flatten to [total_size], stack to [3, total_size] for map_coordinates
    orig_shape = s.shape[:-1]  # leading batch dims
    coords = np.stack([
        i_theta.ravel(),
        i_thdot.ravel(),
        i_a.ravel()
    ], axis=0)  # [3, total_size]

    # trilinear interpolation with nearest-neighbor fallback at boundaries
    result = map_coordinates(Q, coords, order=1, mode='nearest')
    result = result.reshape(orig_shape)  # [...]

    return result


def argmax_a(
    Q: np.ndarray, s: np.ndarray,
    env_cfg: Any, q_cfg: QGridCfg
) -> np.ndarray:
    """
    vectorized action argmax with parabolic refinement.

    pseudocode:
        1. tile s repeated N_action times: s_tile [..., N_action, 2].
           (state-dim stays on last axis so q_lookup's `s[..., 0]` keeps picking theta.)
        2. get action grid a_grid [N_action].
        3. broadcast a_grid to [..., N_action].
        4. vectorized q_lookup -> q_vals [..., N_action].
        5. coarse argmax: k_max [...]; q_max [...].
        6. parabolic refinement (interior indices only):
           - q_lo, q_mid, q_hi from neighbors.
           - denom = q_lo - 2*q_mid + q_hi.
           - where denom != 0: delta = 0.5 * (q_lo - q_hi) / denom, refined_k = k_max + delta.
           - clamp refined_k to [0, N_action - 1].
        7. convert refined index back to continuous action.
        8. clamp a_star to [-action_clip, action_clip].

    args:
        Q: [N_theta, N_theta_dot, N_action] float64.
        s: [..., 2] float64. continuous states.
        env_cfg, q_cfg: configuration.

    returns:
        a_star [...] float64, clipped to [-action_clip, action_clip].
    """
    _, _, a_grid = _axes(env_cfg, q_cfg)

    # tile s along a new N_action axis. keep state-dim on last so q_lookup's
    # `s[..., 0]`/`s[..., 1]` continue to pick theta and theta_dot.
    orig_shape = s.shape[:-1]
    s_tile = np.broadcast_to(s[..., np.newaxis, :], orig_shape + (q_cfg.N_action, 2))

    # broadcast a_grid to [..., N_action]
    a_broadcast = np.broadcast_to(a_grid, orig_shape + (q_cfg.N_action,))

    # vectorized q_lookup: q_vals [..., N_action]
    q_vals = q_lookup(Q, s_tile, a_broadcast, env_cfg, q_cfg)

    # coarse argmax: k_max [...], q_max [...]
    k_max = np.argmax(q_vals, axis=-1)
    q_max = q_vals[..., k_max] if orig_shape else q_vals[k_max]

    # parabolic refinement for interior indices (0 < k_max < N_action - 1)
    refined_k = k_max.astype(np.float64)

    # mask for interior indices
    interior = (k_max > 0) & (k_max < q_cfg.N_action - 1)

    if np.any(interior):
        # per-row neighbor lookup. plain `q_vals[..., k_max - 1]` would fancy-index
        # along the last axis with the full k_max array, broadcasting it into every
        # leading row and producing shape (..., N_action) instead of (...,).
        # take_along_axis does the per-row pick.
        if orig_shape:
            k_lo  = np.clip(k_max - 1, 0, q_cfg.N_action - 1)[..., np.newaxis]
            k_mid = k_max[..., np.newaxis]
            k_hi  = np.clip(k_max + 1, 0, q_cfg.N_action - 1)[..., np.newaxis]
            q_lo  = np.take_along_axis(q_vals, k_lo,  axis=-1).squeeze(-1)
            q_mid = np.take_along_axis(q_vals, k_mid, axis=-1).squeeze(-1)
            q_hi  = np.take_along_axis(q_vals, k_hi,  axis=-1).squeeze(-1)
        else:
            q_lo = q_vals[k_max - 1]
            q_mid = q_vals[k_max]
            q_hi = q_vals[k_max + 1]

        denom = q_lo - 2.0 * q_mid + q_hi

        # only refine where denom != 0
        refine_mask = interior & (np.abs(denom) > 1e-12)
        if np.any(refine_mask):
            delta = np.where(
                refine_mask,
                0.5 * (q_lo - q_hi) / denom,
                0.0
            )
            refined_k = np.where(refine_mask, k_max + delta, refined_k)

    # clamp refined_k to [0, N_action - 1]
    refined_k = np.clip(refined_k, 0, q_cfg.N_action - 1)

    # convert refined index back to continuous action
    a_star = a_grid[0] + refined_k / (q_cfg.N_action - 1) * (a_grid[-1] - a_grid[0])

    # clamp a_star to [-action_clip, action_clip]
    a_star = np.clip(a_star, -env_cfg.action_clip, env_cfg.action_clip)

    return a_star


def build_q(
    F: Callable, r_fn: Callable,
    env_cfg: Any, q_cfg: QGridCfg
) -> Dict[str, Any]:
    """
    vectorized fitted-Q-iteration over entire (s, a) grid until convergence.

    pseudocode:
        1. precompute state grid: theta_grid, thdot_grid.
           s_grid = cartesian product [N_theta * N_theta_dot, 2].
           a_grid = linspace(-action_clip, action_clip, N_action).
        2. initialize Q_prev = zeros([N_theta, N_theta_dot, N_action]).
        3. for it in range(fqi_max_iter):
           a. construct (s_flat, a_flat) covering all (state, action) cells:
              s_flat: [N_theta * N_theta_dot * N_action, 2].
              a_flat: [N_theta * N_theta_dot * N_action].
           b. dynamics & reward:
              s_next = F(s_flat, a_flat, env_cfg).
              r = r_fn(s_flat, a_flat, s_next, env_cfg).
           c. compute V_next = max_{a'} Q(s_next, a') for each s_next:
              shape V_next: [N_theta * N_theta_dot * N_action].
           d. bellman update:
              Q_flat_new = r + gamma * V_next.
              Q_new = reshape(Q_flat_new).
           e. convergence check:
              residual = max(abs(Q_new - Q_prev)).
              if residual < fqi_tol: break.
           f. Q_prev = Q_new.
        4. return {'Q': Q_new, 'bellman_residual': float(residual), 'iterations': int(it + 1)}.

    args:
        F: Callable(s, a, cfg) -> s_next. dynamics function.
        r_fn: Callable(s, a, s_next, cfg) -> r. reward function.
        env_cfg: environment config.
        q_cfg: QGridCfg.

    returns:
        dict with keys 'Q' [N_theta, N_theta_dot, N_action], 'bellman_residual', 'iterations'.
    """
    theta_grid, thdot_grid, a_grid = _axes(env_cfg, q_cfg)

    # construct state grid: cartesian product [N_theta * N_theta_dot, 2]
    theta_mesh, thdot_mesh = np.meshgrid(theta_grid, thdot_grid, indexing='ij')  # [N_theta, N_theta_dot]
    s_grid = np.stack([theta_mesh.ravel(), thdot_mesh.ravel()], axis=-1)  # [N_theta * N_theta_dot, 2]

    # initialize Q
    Q_prev = np.zeros((q_cfg.N_theta, q_cfg.N_theta_dot, q_cfg.N_action), dtype=np.float64)

    residual = float('inf')
    iterations = 0

    for it in range(q_cfg.fqi_max_iter):
        # construct (s_flat, a_flat) covering all (state, action) cells
        # tile s_grid to [N_theta * N_theta_dot * N_action, 2]
        s_flat = np.repeat(s_grid, q_cfg.N_action, axis=0)  # [N_theta * N_theta_dot * N_action, 2]
        # tile a_grid to [N_theta * N_theta_dot * N_action]
        a_flat = np.tile(a_grid, q_cfg.N_theta * q_cfg.N_theta_dot)  # [N_theta * N_theta_dot * N_action]

        # dynamics & reward
        s_next = F(s_flat, a_flat, env_cfg)  # [N_theta * N_theta_dot * N_action, 2]
        r = r_fn(s_flat, a_flat, s_next, env_cfg)  # [N_theta * N_theta_dot * N_action]

        # compute V_next = max_{a'} Q(s_next, a') for each s_next
        # reshape s_next to [N_theta * N_theta_dot, N_action, 2] for argmax_a
        s_next_grid = s_next.reshape(q_cfg.N_theta * q_cfg.N_theta_dot, q_cfg.N_action, 2)
        # vectorized argmax_a over leading batch dimension
        v_next_flat = np.zeros(s_next.shape[0], dtype=np.float64)
        for i in range(s_next_grid.shape[0]):
            a_opt = argmax_a(Q_prev, s_next_grid[i], env_cfg, q_cfg)  # [N_action]
            # look up Q values at optimal actions
            q_at_opt = q_lookup(Q_prev, s_next_grid[i], a_opt, env_cfg, q_cfg)  # [N_action]
            v_next_flat[i * q_cfg.N_action:(i + 1) * q_cfg.N_action] = q_at_opt

        # bellman update
        Q_flat_new = r + q_cfg.gamma * v_next_flat  # [N_theta * N_theta_dot * N_action]
        Q_new = Q_flat_new.reshape(q_cfg.N_theta, q_cfg.N_theta_dot, q_cfg.N_action)  # [N_theta, N_theta_dot, N_action]

        # convergence check
        residual = np.max(np.abs(Q_new - Q_prev))

        if residual < q_cfg.fqi_tol:
            iterations = it + 1
            break

        Q_prev = Q_new
        iterations = it + 1

    return {
        'Q': Q_new,
        'bellman_residual': float(residual),
        'iterations': int(iterations)
    }


def hash_q_cfg(
    env_cfg: Any, r_name: str, q_cfg: QGridCfg,
    alpha: Optional[float] = None,
    r_E_name: Optional[str] = None,
    r_anti_name: Optional[str] = None
) -> str:
    """
    canonical sha256 hash of Q-table configuration.

    when r_name == "r_O", asserts r_E_name and r_anti_name are not None
    to prevent cache collisions across experiments with different reward bases.

    pseudocode:
        1. build canonical dict with all config parameters.
        2. when r_name == "r_O", assert r_E_name is not None and r_anti_name is not None.
        3. json.dumps(cfg, sort_keys=True) -> s.
        4. hashlib.sha256(s.encode()).hexdigest() -> h.
        5. return h[:16].

    args:
        env_cfg: PendulumCfg. extract g, ell, m, dt, action_clip, theta_dot_clip.
        r_name: str. reward name (e.g., "upright", "swingdown", "r_O").
        q_cfg: QGridCfg.
        alpha: Optional[float]. interpolation weight if reward is blended; None for primary rewards.
        r_E_name, r_anti_name: Optional[str]. base reward names; required when r_name == "r_O".

    returns:
        str, first 16 hex characters of sha256.
    """
    # when r_name == "r_O", assert r_E_name and r_anti_name are not None
    if r_name == "r_O":
        assert r_E_name is not None, "r_E_name required when r_name == 'r_O'"
        assert r_anti_name is not None, "r_anti_name required when r_name == 'r_O'"

    cfg = {
        "g": env_cfg.g,
        "ell": env_cfg.ell,
        "m": env_cfg.m,
        "dt": env_cfg.dt,
        "action_clip": env_cfg.action_clip,
        "theta_dot_clip": env_cfg.theta_dot_clip,
        "r_name": r_name,
        "N_theta": q_cfg.N_theta,
        "N_theta_dot": q_cfg.N_theta_dot,
        "N_action": q_cfg.N_action,
        "gamma": q_cfg.gamma,
        "fqi_max_iter": q_cfg.fqi_max_iter,
        "fqi_tol": q_cfg.fqi_tol,
        "alpha": alpha,
        "r_E_name": r_E_name,
        "r_anti_name": r_anti_name,
    }

    s = json.dumps(cfg, sort_keys=True)
    h = hashlib.sha256(s.encode()).hexdigest()
    return h[:16]


def load_or_build_q(
    env_cfg: Any, r_fn: Callable, r_name: str, q_cfg: QGridCfg,
    cache_dir: str, F: Optional[Callable] = None,
    alpha: Optional[float] = None,
    r_E_name: Optional[str] = None,
    r_anti_name: Optional[str] = None,
    rebuild: bool = False
) -> Dict[str, Any]:
    """
    high-level cache orchestrator: load or build Q-table; write hdf5 atomically.

    pseudocode:
        1. mkdir -p cache_dir.
        2. compute cache key: h = hash_q_cfg(...).
        3. cache_file = Path(cache_dir) / f"q_{h}.h5".
        4. if not rebuild and cache_file.exists():
           - try: load hdf5, return dict.
           - except: log warning, fall through to rebuild.
        5. call build_q(F, r_fn, env_cfg, q_cfg).
        6. write hdf5 atomically:
           - write to tmp file.
           - atomic rename.
        7. return build_result.

    args:
        env_cfg: PendulumCfg.
        r_fn: reward function.
        r_name: reward name for cache key.
        q_cfg: QGridCfg.
        cache_dir: str. directory for hdf5 cache.
        F: Optional[Callable]. dynamics function (required for build, optional for load).
        alpha: Optional[float]. interpolation weight; None if primary reward.
        r_E_name, r_anti_name: Optional[str]. base reward names; required when r_name == "r_O".
        rebuild: bool. if True, force rebuild even if cache exists.

    returns:
        dict with keys 'Q', 'bellman_residual', 'iterations'.
    """
    # mkdir -p cache_dir
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # compute cache key
    h = hash_q_cfg(env_cfg, r_name, q_cfg, alpha, r_E_name, r_anti_name)
    cache_file = cache_path / f"q_{h}.h5"

    # try to load from cache if not rebuilding
    if not rebuild and cache_file.exists():
        try:
            with h5py.File(cache_file, 'r') as f:
                Q = f['Q'][:]  # [..., ...]
                bellman_residual = float(f.attrs['bellman_residual'])
                iterations = int(f.attrs['iterations'])
            return {
                'Q': Q,
                'bellman_residual': bellman_residual,
                'iterations': iterations
            }
        except (KeyError, OSError, h5py.Error):
            # silently fall through to rebuild
            pass

    # build Q-table
    assert F is not None, "F (dynamics function) required for build"
    build_result = build_q(F, r_fn, env_cfg, q_cfg)

    # write hdf5 atomically
    tmp_path = cache_file.parent / (cache_file.name + ".tmp")
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("Q", data=build_result['Q'])
        f.attrs['bellman_residual'] = build_result['bellman_residual']
        f.attrs['iterations'] = build_result['iterations']
    tmp_path.rename(cache_file)

    return build_result
