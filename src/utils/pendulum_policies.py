"""
Two numpy-only policy classes for the pendulum domain.

GaussPolicy: 1-D Gaussian around argmax of Q-table.
MixPolicy: 2-component Gaussian mixture with Bernoulli gating.
"""

import numpy as np
from scipy.special import logsumexp
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.pendulum_q import QGridCfg, argmax_a
    from src.utils.pendulum import PendulumCfg


@dataclass(frozen=False)
class GaussPolicy:
    """
    Gaussian policy around argmax of Q(s, a).

    Plan:
      - store precomputed Q-table and env/q config.
      - at sample/log_prob time: compute mu = argmax_a(Q, s, env_cfg, q_cfg)
        via vectorized line search.
      - sample: mu + sigma * N(0, 1); log_prob: standard gaussian pdf at unclipped a.

    Vectorization: leading dims of s can be arbitrary ([N, 2], [N, T+1, 2], etc.).
    Last axis of s is always 2; output shapes match s.shape[:-1].
    Actions are returned unclipped (env clips internally in F).
    """

    Q: np.ndarray           # shape [N_theta, N_theta_dot, N_action]
    sigma: float            # gaussian std dev (required)
    env_cfg: 'PendulumCfg'  # for dynamics signature only
    q_cfg: 'QGridCfg'       # for argmax_a and q_lookup calls

    def __post_init__(self):
        """precompute log constant for efficiency."""
        self._log_sigma_const = np.log(self.sigma) + 0.5 * np.log(2.0 * np.pi)

    def sample(self, s: np.ndarray, gen: np.random.Generator) -> np.ndarray:
        """
        Draw action from N(mu, sigma) where mu = argmax Q at state s.

        Input:
          s: [..., 2], arbitrary leading dims. last axis is (theta, theta_dot).
          gen: np.random.Generator.

        Output:
          a: [...], scalar action per state.

        Plan:
          mu = argmax_a(self.Q, s, self.env_cfg, self.q_cfg)  # shape s.shape[:-1]
          noise = gen.standard_normal(mu.shape)               # iid N(0, 1)
          a = mu + self.sigma * noise                         # unclipped
          return a
        """
        # runtime import to avoid circular dependency
        from src.utils.pendulum_q import argmax_a

        # compute mean action at each state
        mu = argmax_a(self.Q, s, self.env_cfg, self.q_cfg)  # shape: s.shape[:-1]

        # sample noise
        noise = gen.standard_normal(mu.shape)  # shape: s.shape[:-1]

        # unclipped action
        a = mu + self.sigma * noise
        return a

    def log_prob(self, a: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Evaluate log pdf of Gaussian at unclipped actions.

        Input:
          a: [...], scalar per state.
          s: [..., 2] with same leading shape as a.

        Output:
          log_p: [...], scalar log-prob per state.

        Plan:
          mu = argmax_a(self.Q, s, self.env_cfg, self.q_cfg)  # shape s.shape[:-1]
          z = (a - mu) / self.sigma
          log_p = -0.5 * z**2 - self._log_sigma_const
          return log_p

        Standard univariate Gaussian log-pdf:
          log N(a | mu, sigma) = -0.5 * ((a - mu) / sigma)^2 - log(sigma) - 0.5*log(2*pi)
        """
        # runtime import to avoid circular dependency
        from src.utils.pendulum_q import argmax_a

        # compute mean action at each state
        mu = argmax_a(self.Q, s, self.env_cfg, self.q_cfg)  # shape: s.shape[:-1]

        # standardized residual
        z = (a - mu) / self.sigma

        # log pdf
        log_p = -0.5 * z**2 - self._log_sigma_const
        return log_p


@dataclass(frozen=False)
class MixPolicy:
    """
    2-component Gaussian mixture: pi_mix = (1-beta)*pi_O + beta*pi_E.

    Plan:
      - store two GaussPolicy objects (p_O "on-policy", p_E "expert").
      - sample: bernoulli gating between the two; sample both to keep RNG draws
        balanced (CRN).
      - log_prob: logsumexp over mixture components; special-case beta in {0, 1}.

    Vectorization: same as GaussPolicy (arbitrary leading dims).
    """

    p_O: GaussPolicy              # "on-policy" component
    p_E: GaussPolicy              # "expert" component
    beta: float                   # blend weight for p_E, in [0, 1]

    def sample(self, s: np.ndarray, gen: np.random.Generator) -> np.ndarray:
        """
        Sample from mixture by Bernoulli gating.

        Input:
          s: [..., 2].
          gen: np.random.Generator.

        Output:
          a: [...], scalar action per state.

        Plan:
          mask = gen.uniform(0, 1, size=s.shape[:-1]) < self.beta  # bool, shape s.shape[:-1]
          a_E = self.p_E.sample(s, gen)                            # always sample both
          a_O = self.p_O.sample(s, gen)
          a = np.where(mask, a_E, a_O)
          return a

        CRITICAL for CRN: Sample both a_E and a_O unconditionally to keep RNG
        call sequence deterministic across different beta values. Only the selection
        via np.where depends on beta.
        """
        # bernoulli gate: mask=True means sample from expert (p_E)
        mask = gen.uniform(0, 1, size=s.shape[:-1]) < self.beta  # shape: s.shape[:-1]

        # sample both components unconditionally for deterministic RNG sequence
        a_E = self.p_E.sample(s, gen)  # shape: s.shape[:-1]
        a_O = self.p_O.sample(s, gen)  # shape: s.shape[:-1]

        # select based on mask
        a = np.where(mask, a_E, a_O)
        return a

    def log_prob(self, a: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Evaluate log pdf of mixture at unclipped actions.

        Input:
          a: [...], scalar per state.
          s: [..., 2] with same leading shape as a.

        Output:
          log_p: [...], scalar log-prob per state.

        Plan:
          log_p_O = self.p_O.log_prob(a, s)  # shape a.shape
          log_p_E = self.p_E.log_prob(a, s)

          if self.beta == 0.0:
              return log_p_O
          elif self.beta == 1.0:
              return log_p_E
          else:
              log_w = np.array([np.log(1.0 - self.beta), np.log(self.beta)])
              log_w_plus_p = np.stack([
                  log_w[0] + log_p_O,
                  log_w[1] + log_p_E
              ], axis=-1)  # shape [..., 2]
              return logsumexp(log_w_plus_p, axis=-1)  # shape [...]

        Mixture log-pdf:
          log p_mix(a|s) = logsumexp([log(1-beta) + log_p_O, log(beta) + log_p_E])

        Special-case beta in {0, 1} to avoid log(0) and improve clarity.
        """
        # evaluate both components
        log_p_O = self.p_O.log_prob(a, s)  # shape: a.shape
        log_p_E = self.p_E.log_prob(a, s)  # shape: a.shape

        # short-circuit for degenerate cases
        if self.beta == 0.0:
            return log_p_O
        elif self.beta == 1.0:
            return log_p_E
        else:
            # logsumexp over 2-component mixture
            log_w = np.array([np.log(1.0 - self.beta), np.log(self.beta)])
            log_w_plus_p = np.stack([
                log_w[0] + log_p_O,
                log_w[1] + log_p_E
            ], axis=-1)  # shape: [..., 2]
            return logsumexp(log_w_plus_p, axis=-1)  # shape: [...]


def make_gauss(Q, sigma, env_cfg, q_cfg):
    """construct GaussPolicy."""
    return GaussPolicy(Q=Q, sigma=sigma, env_cfg=env_cfg, q_cfg=q_cfg)
