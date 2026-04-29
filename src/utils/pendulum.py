"""canonical pendulum-v1 environment primitives (numpy, stateless)."""
import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PendulumCfg:
    """pendulum environment configuration (frozen, hashable).

    parameters for canonical pendulum-v1 dynamics (theta=0 = upright),
    euler step, action clipping, and initial state distribution box.
    """
    g: float = 10.0
    ell: float = 1.0
    m: float = 1.0
    dt: float = 0.05
    action_clip: float = 2.0
    theta_dot_clip: float = 8.0
    mu0_box: tuple[tuple[float, float], tuple[float, float]] = field(
        default_factory=lambda: ((-np.pi, np.pi), (-1.0, 1.0))
    )


def angle_norm(theta: np.ndarray) -> np.ndarray:
    """wrap angle to [-pi, pi) via modular arithmetic.

    vectorized over arbitrary leading dims.
    one-liner; used by F and reward functions.
    """
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def F(
    s: np.ndarray,
    a: np.ndarray,
    cfg: PendulumCfg
) -> np.ndarray:
    """canonical pendulum-v1 euler step with action clipping and angle wrapping.

    theta=0 = upright; dynamics are sin(theta) torque + action.

    state is stored in canonical [-pi, pi) range (theta is wrapped inside F).
    action is clipped to ±action_clip before applying dynamics.
    angular velocity is clipped to ±theta_dot_clip after euler step.

    vectorized over arbitrary leading dimensions via numpy broadcasting.
    action can be passed as [...] or [..., 1]; both are valid.
    """
    # squeeze action if [..., 1]
    if a.ndim > 0 and a.shape[-1:] == (1,):
        a = np.squeeze(a, axis=-1)

    # clip action
    a_clip = np.clip(a, -cfg.action_clip, cfg.action_clip)

    # extract state components
    theta = s[..., 0]
    theta_dot = s[..., 1]

    # compute angular acceleration
    accel = (
        3 * cfg.g / (2 * cfg.ell) * np.sin(theta)
        + 3 / (cfg.m * cfg.ell**2) * a_clip
    )

    # euler step for angular velocity (with clipping)
    theta_dot_next = np.clip(
        theta_dot + accel * cfg.dt,
        -cfg.theta_dot_clip,
        cfg.theta_dot_clip
    )

    # euler step for angle
    theta_next_raw = theta + theta_dot_next * cfg.dt

    # wrap theta to [-pi, pi)
    theta_next = angle_norm(theta_next_raw)

    # stack and return (preserves input dtype)
    s_next = np.stack([theta_next, theta_dot_next], axis=-1)
    return s_next.astype(s.dtype)


def sample_mu0(
    N: int,
    cfg: PendulumCfg,
    gen: np.random.Generator
) -> np.ndarray:
    """draw N initial states uniformly from mu0 box.

    returns unclipped samples (F will wrap/clip as needed downstream).
    """
    theta = gen.uniform(cfg.mu0_box[0][0], cfg.mu0_box[0][1], size=N)
    theta_dot = gen.uniform(cfg.mu0_box[1][0], cfg.mu0_box[1][1], size=N)
    return np.stack([theta, theta_dot], axis=-1)


def log_mu0(
    s: np.ndarray,
    cfg: PendulumCfg
) -> np.ndarray:
    """log-density of uniform initial-state distribution (closed-form).

    assumes s is in-support (i.e., already sampled or valid trajectory state).
    does NOT return -inf for out-of-support points; caller is responsible
    for ensuring trajectories remain valid (F wraps theta; states initialized
    via sample_mu0 are by construction in support).
    """
    volume = (
        (cfg.mu0_box[0][1] - cfg.mu0_box[0][0])
        * (cfg.mu0_box[1][1] - cfg.mu0_box[1][0])
    )
    log_p = -np.log(volume)
    # broadcast scalar to shape s.shape[:-1]
    return np.full(s.shape[:-1], log_p, dtype=np.float64)


def r_upright(
    s: np.ndarray,
    a: np.ndarray,
    s_next: np.ndarray,
    cfg: PendulumCfg
) -> np.ndarray:
    """canonical pendulum-v1 reward: penalize angle deviation and velocity magnitude.

    goal is to balance upright (theta ~ 0).
    r(s, a, s') = -(theta'^2 + 0.1*theta_dot'^2 + 0.001*a_clip^2)
    where a_clip = clip(a, ±action_clip).
    vectorized over leading dims.

    uses angle_norm(theta_next) to ensure wrapped-angle reward.
    action a is unclipped on input; reward uses clipped version for penalty term.
    """
    # squeeze action if [..., 1]
    if a.ndim > 0 and a.shape[-1:] == (1,):
        a = np.squeeze(a, axis=-1)

    # clip action
    a_clip = np.clip(a, -cfg.action_clip, cfg.action_clip)

    # extract next state
    theta_next = s_next[..., 0]
    theta_dot_next = s_next[..., 1]

    # reward
    r = -(
        angle_norm(theta_next)**2
        + 0.1 * theta_dot_next**2
        + 0.001 * a_clip**2
    )
    return r.astype(np.float64)


def r_swingdown(
    s: np.ndarray,
    a: np.ndarray,
    s_next: np.ndarray,
    cfg: PendulumCfg
) -> np.ndarray:
    """variant of r_upright: goal is to swing down (theta ~ pi = upside-down).

    r(s, a, s') = -((theta' - pi)'^2 + 0.1*theta_dot'^2 + 0.001*a_clip^2)
    where angle norm wraps (theta' - pi) to [-pi, pi).
    vectorized over leading dims.

    mirrors r_upright with theta_next shifted by pi before wrapping.
    useful for curriculum or alternative task definitions.
    """
    # squeeze action if [..., 1]
    if a.ndim > 0 and a.shape[-1:] == (1,):
        a = np.squeeze(a, axis=-1)

    # clip action
    a_clip = np.clip(a, -cfg.action_clip, cfg.action_clip)

    # extract next state
    theta_next = s_next[..., 0]
    theta_dot_next = s_next[..., 1]

    # reward (goal is theta ~ pi)
    r = -(
        angle_norm(theta_next - np.pi)**2
        + 0.1 * theta_dot_next**2
        + 0.001 * a_clip**2
    )
    return r.astype(np.float64)
