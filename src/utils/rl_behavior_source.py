"""
online soft Q-learning on tile-coded Q with continuous Boltzmann exploration
and replay collection. NO dependencies beyond numpy, scipy, h5py.

implements behavior source: soft Q-learning training + Boltzmann policy.
"""

import hashlib
import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional, Any

from src.utils.pendulum_q import q_lookup, _axes, hash_q_cfg
from src.utils.pendulum import F, sample_mu0
from src.utils.io import _set_seed, _write_hdf5_atomic


@dataclass(frozen=True)
class SoftQTrainerCfg:
    """Training configuration for soft Q-learning.

    Binding defaults: n_steps=200_000, lr=0.05, minibatch=32, td_update_freq=1,
    beta_init=0.3, beta_final=10.0, beta_schedule="linear", episode_len=200,
    warmup=1_000, stream_target=500_000.

    fields:
        n_steps: int. total online steps (excluding warm reset of Q).
        lr: float. TD step size (multiplicative on temporal-difference).
        minibatch: int. batch size for random replay samples.
        td_update_freq: int. number of env steps per TD update (stride in replay buffer).
        beta_init, beta_final: float. inverse temp at t=0 and t=n_steps.
        beta_schedule: str. "linear" or "constant"; if "constant" use beta_init.
        episode_len: int. steps between episodic resets from sample_mu0.
        warmup: int. steps before replay collection begins (optional; collection
          starts at step 1 if warmup > n_steps).
        stream_target: int. subsample replay to this many (s, a) pairs, stratified
          across phase (early/mid/late thirds).
        phase_weights: tuple of 3 floats. RELATIVE subsampling weights for the
          early/mid/late phases ((1,1,1) = proportional balance). down-weighting
          the early exploration phase raises cross-policy divergence: the broad
          near-uniform early mass is shared by every policy's flow and caps K1.
    """
    n_steps: int = 200_000
    lr: float = 0.05
    minibatch: int = 32
    td_update_freq: int = 1
    beta_init: float = 0.3
    beta_final: float = 10.0
    beta_schedule: str = "linear"
    episode_len: int = 200
    warmup: int = 1_000
    stream_target: int = 500_000
    phase_weights: tuple = (1.0, 1.0, 1.0)


def sample_boltzmann_continuous(
    Q: np.ndarray,
    s: np.ndarray,
    beta: float,
    env_cfg: Any,
    q_cfg: Any,
    gen: np.random.Generator
) -> np.ndarray:
    """
    Continuous piecewise-exponential inverse-CDF sampler.

    implements pi(a|s) propto exp(beta*Q(s,a)) over N_action=21 knot points.
    support is [-action_clip, action_clip]; each adjacent pair of knots forms
    a segment with piecewise-constant density exponent.

    procedure:
    1. extract Q(s, :) -> q_vals [B, N_action] via q_lookup.
    2. for each state, compute segment endpoints and log-integral (see binding
       numerics below).
    3. sample u ~ U(0,1) once per state via gen.uniform([B]).
    4. invert CDF via segment location and residual mass, using guarded
       arithmetic (expm1, log1p) for numerical stability.
    5. clamp to [-action_clip, action_clip].

    binding numerics (segment j from a_j to a_{j+1}):
    - q_lo, q_hi = endpoints of Q values; Delta_j = a_{j+1} - a_j.
    - d_j = beta * (q_hi - q_lo); per-segment exponent width.
    - log I_j (unnormalized) = beta*q_hi + log(Delta_j) + log(-expm1(-d_j)) - log(d_j)
      for d_j > 0 (mirror for d_j < 0).
    - flat branch (|d_j| < 1e-6): log I_j = beta*q_lo + log(Delta_j).
    - global max-subtraction: c = max over j of (beta*q_hi if d_j > 0 else beta*q_lo).
    - never evaluate exp(beta*q) raw; use c - offset terms in logsumexp.
    - inversion within j (residual mass w in [0,1]):
      a = a_j + Delta_j * log1p(w*expm1(d_j))/d_j.
    - guards for extreme |d_j|:
      * d_j >= 30: log1p term simplifies to d_j + log(w + (1-w)*exp(-d_j)).
      * d_j <= -30: log(1 - w + w*exp(d_j)).
      * |d_j| < 1e-8: a = a_j + Delta_j*w (linear).
    - batched segment location: seg = (cum_probs < u[:, None]).sum(axis=1),
      clipped to [0, n_seg-1]. (torch.searchsorted unavailable; boolean-sum idiom.)
    - one u = gen.uniform([B]) per call (single RNG array call, monotone coupling).

    args:
        Q: [N_theta, N_theta_dot, N_action] float64. Q-values.
        s: [B, 2] float64. continuous states.
        beta: float. inverse temperature.
        env_cfg: environment config with action_clip.
        q_cfg: QGridCfg with N_action.
        gen: np.random.Generator.

    returns:
        [B] float64. actions in [-action_clip, action_clip].
    """
    _, _, a_grid = _axes(env_cfg, q_cfg)
    B = s.shape[0]
    N_action = q_cfg.N_action
    N_seg = N_action - 1

    # extract Q(s, :) -> q_vals [B, N_action]; tile states across the action
    # grid so q_lookup broadcasts (same pattern as pendulum_q.argmax_a)
    s_tile = np.broadcast_to(s[:, None, :], (B, N_action, 2))
    a_tile = np.broadcast_to(a_grid, (B, N_action))
    q_vals = q_lookup(Q, s_tile, a_tile, env_cfg, q_cfg)  # [B, N_action]

    # segment data: endpoints and widths
    a_lo = a_grid[:-1]  # [N_seg]
    a_hi = a_grid[1:]   # [N_seg]
    Delta = a_hi - a_lo  # [N_seg]

    q_lo = q_vals[:, :-1]  # [B, N_seg]
    q_hi = q_vals[:, 1:]   # [B, N_seg]
    d = beta * (q_hi - q_lo)  # [B, N_seg]

    # log-integral computation with global max-subtraction
    # c = max over all segments of (beta*q_hi if d > 0 else beta*q_lo)
    c_candidates = np.where(d > 0, beta * q_hi, beta * q_lo)  # [B, N_seg]
    c = np.max(c_candidates, axis=1, keepdims=True)  # [B, 1]

    # compute log-unnormalized segment integrals: [B, N_seg]
    # use np.where to handle different formulas per-element

    # identify segment types
    flat = np.abs(d) < 1e-6
    pos = d > 0
    neg = d < 0
    extreme_pos = d >= 30
    extreme_neg = d <= -30

    # precompute log(Delta) for all segments [N_seg]
    log_Delta = np.log(Delta)  # [N_seg]

    # vectorized log-integral computation. np.where evaluates every branch
    # eagerly, so masked lanes produce benign invalid-log noise: silence it.
    _errstate = np.errstate(divide="ignore", invalid="ignore")
    _errstate.__enter__()

    # case 1: flat (|d| < 1e-6)
    log_I_flat = beta * q_lo + log_Delta  # [B, N_seg]

    # case 2: normal positive (d > 0, not flat, not extreme)
    log_I_pos = (beta * q_hi + log_Delta +
                 np.log(-np.expm1(-d)) - np.log(d))

    # case 3: normal negative (d < 0, not flat, not extreme)
    log_I_neg = (beta * q_lo + log_Delta +
                 np.log(-np.expm1(d)) - np.log(-d))

    # case 4: extreme positive (d >= 30)
    log_I_ext_pos = beta * q_hi + log_Delta + d - np.log(d)

    # case 5: extreme negative (d <= -30)
    log_I_ext_neg = beta * q_lo + log_Delta - np.log(np.abs(d))

    # combine using np.where
    log_I = np.where(flat, log_I_flat,
            np.where(extreme_pos, log_I_ext_pos,
            np.where(extreme_neg, log_I_ext_neg,
            np.where(pos, log_I_pos, log_I_neg))))  # [B, N_seg]
    _errstate.__exit__(None, None, None)

    # apply global max-subtraction
    log_I = log_I - c

    # normalize and compute cumulative probabilities
    # prevent log-sum-exp overflow with stabilization
    log_norm = np.logaddexp.reduce(log_I, axis=1, keepdims=True)  # [B, 1]
    weights = np.exp(log_I - log_norm)  # [B, N_seg]
    cum_probs = np.cumsum(weights, axis=1)  # [B, N_seg]

    # sample u ~ U(0,1) once per state (single RNG call)
    u = gen.uniform(size=B)  # [B]

    # batched segment location: seg = (cum_probs < u[:, None]).sum(axis=1)
    seg = (cum_probs < u[:, None]).sum(axis=1)  # [B]
    seg = np.clip(seg, 0, N_seg - 1)  # [B]

    # extract per-segment parameters
    a_j = a_lo[seg]  # [B]
    Delta_j = Delta[seg]  # [B]
    d_j = d[np.arange(B), seg]  # [B]
    q_lo_j = q_lo[np.arange(B), seg]  # [B]

    # compute residual mass w in [0, 1] within segment
    cum_before = np.where(seg > 0, cum_probs[np.arange(B), seg - 1], 0.0)  # [B]
    cum_current = cum_probs[np.arange(B), seg]  # [B]
    w = np.clip((u - cum_before) / (cum_current - cum_before + 1e-10), 0.0, 1.0)  # [B]

    # inversion formula with guards; eager branches divide by masked-lane
    # zeros, so silence invalid/divide noise for this block too
    _err2 = np.errstate(divide="ignore", invalid="ignore")
    _err2.__enter__()
    # compute all candidates first, then select with np.where
    flat = np.abs(d_j) < 1e-8
    extreme_pos = d_j >= 30
    extreme_neg = d_j <= -30

    # flat: a = a_j + Delta_j*w
    a_flat = a_j + Delta_j * w

    # extreme positive: a = a_j + Delta_j*(d + log(w + (1-w)*exp(-d)))/d
    a_ext_pos = a_j + Delta_j * (d_j + np.log(w + (1.0 - w) * np.exp(-d_j))) / d_j

    # extreme negative: a = a_j + Delta_j*log(1 - w + w*exp(d))/d
    a_ext_neg = a_j + Delta_j * np.log(1.0 - w + w * np.exp(d_j)) / d_j

    # normal: a = a_j + Delta_j*log1p(w*expm1(d))/d
    a_normal = a_j + Delta_j * np.log1p(w * np.expm1(d_j)) / d_j

    # select using np.where
    a_invert = np.where(flat, a_flat,
                np.where(extreme_pos, a_ext_pos,
                np.where(extreme_neg, a_ext_neg, a_normal)))

    _err2.__exit__(None, None, None)

    # clamp to [-action_clip, action_clip]
    actions = np.clip(a_invert, -env_cfg.action_clip, env_cfg.action_clip)

    return actions


def replay_cache_path(env_cfg, q_cfg, r_name, alpha, trainer_cfg, seed, cache_dir):
    """content-addressed replay path; pure path computation, no side effects.

    key folds in the q-config hash, the FULL trainer config, and the rl seed
    so distinct runs never collide on one file. single source of truth for
    the replay path (step0 resolves paths through this, never by rebuilding
    the hash itself).

    args:
        env_cfg, q_cfg, r_name, alpha, trainer_cfg, seed: as in
            load_or_collect_replay.
        cache_dir: str.

    returns:
        str. replay h5 path (may not exist yet).
    """
    # blended r_O needs the reward-base names for a collision-free hash
    # (hash_q_cfg asserts them); pendulum fixes them to upright/swingdown.
    kw = {"r_E_name": "upright", "r_anti_name": "swingdown"} if r_name == "r_O" else {}
    base = hash_q_cfg(env_cfg, r_name, q_cfg, alpha, **kw)
    tc = json.dumps(asdict(trainer_cfg), sort_keys=True)
    key = hashlib.sha256(f"{base}|{tc}|{seed}".encode()).hexdigest()[:16]
    return str(Path(cache_dir) / f"replay_{key}.h5")


def load_or_collect_replay(
    env_cfg: Any,
    q_cfg: Any,
    r_name: str,
    alpha: float,
    trainer_cfg: SoftQTrainerCfg,
    seed: int,
    cache_dir: str,
    force: bool = False
) -> str:
    """
    Online soft Q-learning run; return h5 path with replay data.

    side effects: sets numpy seed via _set_seed(seed).

    returns h5 path with datasets + attrs:
    - states [N, 2] float64: continuous (theta, theta_dot).
    - actions [N] float64: continuous actions in [-action_clip, action_clip].
    - phase [N] int8: stratum label (0=early, 1=mid, 2=late third of n_steps).

    attrs:
    - cfg_sha256: str, hash_q_cfg(env_cfg, r_name, q_cfg, alpha, ...) [:16].
    - r_name: str.
    - alpha: float.
    - seed: int.
    - n_steps: int (total steps executed).

    cache:
    - cache_key = hash_q_cfg(...); cache_file = cache_dir / f"replay_{key}.h5".
    - if not force and cache_file exists, load and return path.
    - else, run training, write atomically via _write_hdf5_atomic.

    training loop pseudocode:
    1. gen = np.random.default_rng(seed).
    2. Q = zeros([N_theta, N_theta_dot, N_action]) (copy for in-place TD).
    3. (s, a, r_sum) buffers: lists, no allocation overhead.
    4. s_t = sample_mu0(1, env_cfg, gen)[0]; step_count = 0.
    5. for step in range(n_steps):
       a. beta(t) = beta_init + (beta_final - beta_init) * (step / n_steps)
          if beta_schedule == "linear", else beta_init.
       b. a_t = sample_boltzmann_continuous(Q, s_t[np.newaxis], beta, ...) [0].
       c. s_next = F(s_t[np.newaxis], a_t, env_cfg)[0].
       d. r_t = r_fn(s_t, a_t, s_next, env_cfg).
       e. if step >= warmup: record (s_t, a_t, step_count % 3 phase).
       f. if step % td_update_freq == 0 and step >= warmup:
          - sample minibatch of (s, a) from buffer.
          - q_curr = q_lookup(Q, s_batch, a_batch).
          - s_next_batch = F(s_batch, a_batch, env_cfg).
          - q_next = max_a q_lookup(Q, s_next_batch, a[:, np.newaxis]) for all a in grid.
          - td = r_batch + gamma * q_next - q_curr.
          - Q_update = q_lookup indices via argmin distance in state-action space.
          - Q[nearest_cell] += lr * td (scalar in-place scatter, or bilinear weight scatter).
       g. episodic reset: if (step + 1) % episode_len == 0: s_t = sample_mu0(...).
       h. else: s_t = s_next.
    6. subsample (s, a, phase) -> (s_samp, a_samp, phase_samp) to stream_target,
       stratified by phase (preserve 0/1/2 balance). use gen.choice with
       replacement=False, per-phase quotas.
    7. write h5: states, actions, phase; attrs: cfg_sha256, r_name, alpha, seed, n_steps.

    args:
        env_cfg: PendulumCfg.
        q_cfg: QGridCfg.
        r_name: str. reward name (e.g., "r_upright").
        alpha: float. reward mixture weight.
        trainer_cfg: SoftQTrainerCfg.
        seed: int.
        cache_dir: str.
        force: bool. if True, skip cache and rebuild.

    returns:
        str. path to h5 file.
    """
    # check cache (content-addressed; see replay_cache_path)
    cache_file = Path(replay_cache_path(env_cfg, q_cfg, r_name, alpha,
                                        trainer_cfg, seed, cache_dir))
    cache_key = cache_file.stem.removeprefix("replay_")

    if not force and cache_file.exists():
        return str(cache_file)

    # initialize training
    _set_seed(seed)
    gen = np.random.default_rng(seed)

    Q = np.zeros((q_cfg.N_theta, q_cfg.N_theta_dot, q_cfg.N_action), dtype=np.float64)

    # resolve reward: expert task or the alpha-blended r_O used for the
    # ladder runs (r_O(alpha) = (1-alpha) r_upright + alpha r_swingdown)
    from src.utils.pendulum import r_upright, r_swingdown
    if r_name == "r_upright":
        r_fn = r_upright
    elif r_name == "r_O":
        if alpha is None:
            raise ValueError("r_O requires alpha")
        a_blend = float(alpha)

        def r_fn(s, a, s_next, cfg):
            return ((1.0 - a_blend) * r_upright(s, a, s_next, cfg)
                    + a_blend * r_swingdown(s, a, s_next, cfg))
    else:
        raise ValueError(f"unknown r_name: {r_name}")

    # buffers
    states_buf = []
    actions_buf = []
    phase_buf = []

    s_t = sample_mu0(1, env_cfg, gen)[0]  # [2]
    step_count = 0

    for step in range(trainer_cfg.n_steps):
        # compute beta(t). schedules: "linear" (reach beta_final at the end);
        # "linear_hold:<frac>" (reach beta_final at frac*n_steps, then hold
        # so most of the stream is sharp); else constant.
        if trainer_cfg.beta_schedule == "linear":
            beta = trainer_cfg.beta_init + (trainer_cfg.beta_final - trainer_cfg.beta_init) * (step / trainer_cfg.n_steps)
        elif trainer_cfg.beta_schedule.startswith("linear_hold:"):
            frac = float(trainer_cfg.beta_schedule.split(":", 1)[1])
            ramp = max(1, int(frac * trainer_cfg.n_steps))
            beta = min(trainer_cfg.beta_final,
                       trainer_cfg.beta_init + (trainer_cfg.beta_final - trainer_cfg.beta_init) * (step / ramp))
        else:
            beta = trainer_cfg.beta_init

        # sample action
        a_t = sample_boltzmann_continuous(
            Q, s_t[np.newaxis], beta, env_cfg, q_cfg, gen
        )[0]

        # step environment
        s_next = F(s_t[np.newaxis], a_t, env_cfg)[0]

        # compute reward
        r_t = r_fn(s_t, a_t, s_next, env_cfg)

        # record if step >= warmup
        if step >= trainer_cfg.warmup:
            states_buf.append(s_t.copy())
            actions_buf.append(a_t)
            phase_label = (step - trainer_cfg.warmup) // max(1, (trainer_cfg.n_steps - trainer_cfg.warmup) // 3)
            phase_label = min(phase_label, 2)
            phase_buf.append(phase_label)
            step_count += 1

        # TD update
        if step % trainer_cfg.td_update_freq == 0 and step >= trainer_cfg.warmup and len(states_buf) > 0:
            # sample minibatch
            buf_size = len(states_buf)
            mb_indices = gen.choice(buf_size, size=min(trainer_cfg.minibatch, buf_size), replace=False)
            s_batch = np.array([states_buf[i] for i in mb_indices])  # [minibatch, 2]
            a_batch = np.array([actions_buf[i] for i in mb_indices])  # [minibatch]

            # current Q
            q_curr = q_lookup(Q, s_batch, a_batch, env_cfg, q_cfg)  # [minibatch]

            # next state
            s_next_batch = F(s_batch, a_batch, env_cfg)  # [minibatch, 2]

            # max_a Q(s_next, a) by scanning all action grid points
            _, _, a_grid = _axes(env_cfg, q_cfg)
            mb_size = s_batch.shape[0]
            q_vals_all = q_lookup(
                Q,
                np.repeat(s_next_batch, q_cfg.N_action, axis=0),  # [minibatch * N_action, 2]
                np.tile(a_grid, mb_size),  # [minibatch * N_action]
                env_cfg, q_cfg
            )  # [minibatch * N_action]
            q_vals_all = q_vals_all.reshape(mb_size, q_cfg.N_action)  # [minibatch, N_action]
            q_next = np.max(q_vals_all, axis=1)  # [minibatch]

            # TD error
            r_batch = np.array([r_fn(states_buf[i], actions_buf[i], F(states_buf[i][np.newaxis], actions_buf[i], env_cfg)[0], env_cfg) for i in mb_indices])
            td = r_batch + q_cfg.gamma * q_next - q_curr  # [minibatch]

            # nearest-cell update: for each (s, a) in batch, find nearest cell and update
            for i in range(mb_size):
                s_i = s_batch[i]  # [2]
                a_i = a_batch[i]
                td_i = td[i]

                # map (s, a) to nearest cell index
                i_theta = (s_i[0] + np.pi) / (2.0 * np.pi) * q_cfg.N_theta
                i_thdot = (s_i[1] + env_cfg.theta_dot_clip) / (2.0 * env_cfg.theta_dot_clip) * (q_cfg.N_theta_dot - 1)
                i_a = (a_i + env_cfg.action_clip) / (2.0 * env_cfg.action_clip) * (q_cfg.N_action - 1)

                # round to nearest cell
                cell_idx = (
                    int(np.round(i_theta) % q_cfg.N_theta),
                    int(np.clip(np.round(i_thdot), 0, q_cfg.N_theta_dot - 1)),
                    int(np.clip(np.round(i_a), 0, q_cfg.N_action - 1))
                )

                # in-place update
                Q[cell_idx] += trainer_cfg.lr * td_i

        # episodic reset
        if (step + 1) % trainer_cfg.episode_len == 0:
            s_t = sample_mu0(1, env_cfg, gen)[0]
        else:
            s_t = s_next

    # subsample replay to stream_target
    if len(states_buf) > trainer_cfg.stream_target:
        states_arr = np.array(states_buf)
        actions_arr = np.array(actions_buf)
        phase_arr = np.array(phase_buf, dtype=np.int8)

        # stratify by phase with configurable weights (see phase_weights)
        w = np.asarray(trainer_cfg.phase_weights, dtype=np.float64)
        assert w.shape == (3,) and np.all(w >= 0) and w.sum() > 0
        sample_indices = []
        for ph in [0, 1, 2]:
            ph_mask = phase_arr == ph
            ph_indices = np.where(ph_mask)[0]
            if len(ph_indices) > 0 and w[ph] > 0:
                frac = (w[ph] * np.sum(ph_mask)) / float((w * np.bincount(phase_arr, minlength=3)).sum())
                ph_quota = max(1, int(trainer_cfg.stream_target * frac))
                ph_sample = gen.choice(len(ph_indices), size=min(ph_quota, len(ph_indices)), replace=False)
                sample_indices.extend(ph_indices[ph_sample])

        sample_indices = np.array(sample_indices[:trainer_cfg.stream_target])
        states_final = states_arr[sample_indices]
        actions_final = actions_arr[sample_indices]
        phase_final = phase_arr[sample_indices]
    else:
        states_final = np.array(states_buf, dtype=np.float64)
        actions_final = np.array(actions_buf, dtype=np.float64)
        phase_final = np.array(phase_buf, dtype=np.int8)

    # write HDF5 atomically
    datasets = {
        'states': states_final,
        'actions': actions_final,
        'phase': phase_final,
    }
    attrs = {
        'cfg_sha256': cache_key,
        'r_name': r_name,
        'alpha': float(alpha) if alpha is not None else np.nan,  # h5 attrs reject None
        'seed': seed,
        'n_steps': trainer_cfg.n_steps,
    }

    _write_hdf5_atomic(str(cache_file), datasets, attrs)

    return str(cache_file)
