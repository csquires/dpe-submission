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


@dataclass(frozen=False)
class FlowPolicy:
    """
    Normalized-flow policy: bounded 1-D RQ-spline flow conditioned on state.
    Adapts src.models.flow.flow_policy_model.FlowPolicyModel (torch, RNG-free eval)
    to numpy interface matching GaussPolicy contract.

    Vectorization: leading dims arbitrary ([N,2], [N,T+1,2], etc.); last axis is 2.
    Output shapes match s.shape[:-1]. Actions are unclipped (env clips in F).

    CRN contract: sample() makes exactly one array-shaped gen.uniform() call
    per invocation, with size=leading. This preserves deterministic RNG sequencing across
    multiple policies sharing the same gen (monotone coupling). MixPolicy wraps FlowPolicy
    without modification.

    Plan:
      - store torch model and env config.
      - sample: draw u ~ Uniform [0,1] (single RNG call), flatten s/u, loop in chunks,
        call model.push_forward(u, s) with torch.no_grad(), reshape output, return float64.
      - log_prob: flatten a/s, loop in chunks, call model.log_prob(a, s) with torch.no_grad(),
        reshape output, return float64.
    """

    model: object                  # torch module, .eval() on its device
    env_cfg: 'PendulumCfg' = None  # interface compatibility only; unused
    chunk: int = 5000             # batch size for torch loops

    def sample(self, s: np.ndarray, gen: np.random.Generator) -> np.ndarray:
        """
        Draw action from normalized flow at state s.

        Input:
          s: [..., 2], arbitrary leading dims. last axis is (theta, theta_dot).
          gen: np.random.Generator.

        Output:
          a: [...], scalar action per state.

        Plan:
          leading = s.shape[:-1]
          u = gen.uniform(size=leading)  # exactly one array-shaped RNG call (CRN contract)
          if s.size == 0: return np.array([], dtype=np.float64)
          Flatten u, s to 1-D; loop in chunks of self.chunk:
            - np.ascontiguousarray on chunks, dtype conversions
            - torch.from_numpy, call self.model.push_forward(u, s) with torch.no_grad()
            - .detach().cpu().numpy().astype(np.float64)
            - Collect to a_flat
          Reshape a_flat to leading shape; return float64
        """
        import torch

        leading = s.shape[:-1]
        u = gen.uniform(size=leading)  # exactly one array-shaped RNG call for CRN

        # short-circuit empty inputs
        if s.size == 0:
            return np.array([], dtype=np.float64)

        # flatten to [B, 2] and [B]
        s_flat = s.reshape(-1, 2)  # [B, 2]
        u_flat = u.flatten()  # [B]

        B = s_flat.shape[0]
        a_flat = np.empty(B, dtype=np.float64)

        # loop in chunks
        for i in range(0, B, self.chunk):
            end = min(i + self.chunk, B)
            u_chunk = np.ascontiguousarray(u_flat[i:end], dtype=np.float32)
            s_chunk = np.ascontiguousarray(s_flat[i:end], dtype=np.float64)

            # torch computation
            u_t = torch.from_numpy(u_chunk)
            s_t = torch.from_numpy(s_chunk)

            with torch.no_grad():
                a_chunk = self.model.push_forward(u_t, s_t)  # [chunk] on device

            # back to numpy
            a_chunk = a_chunk.detach().cpu().numpy().astype(np.float64)
            a_flat[i:end] = a_chunk

        # reshape to leading dims
        a = a_flat.reshape(leading)
        return a

    def log_prob(self, a: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Evaluate log pdf of flow at unclipped actions.

        Input:
          a: [...], scalar per state.
          s: [..., 2] with same leading shape as a.

        Output:
          log_p: [...], scalar log-prob per state.

        Plan:
          leading = a.shape
          assert a.shape == s.shape[:-1] (for shape validation)
          if a.size == 0: return np.array([], dtype=np.float64)
          Flatten a, s to [B]; loop in chunks:
            - np.ascontiguousarray on chunks, dtype conversions
            - torch.from_numpy, call self.model.log_prob(a, s) with torch.no_grad()
            - .detach().cpu().numpy().astype(np.float64)
            - Collect to log_p_flat
          Reshape log_p_flat to leading shape; return float64
        """
        import torch

        leading = a.shape
        assert a.shape == s.shape[:-1], f"a.shape {a.shape} != s.shape[:-1] {s.shape[:-1]}"

        # short-circuit empty inputs
        if a.size == 0:
            return np.array([], dtype=np.float64)

        # flatten to [B]
        a_flat = a.flatten()  # [B]
        s_flat = s.reshape(-1, 2)  # [B, 2]

        B = a_flat.shape[0]
        log_p_flat = np.empty(B, dtype=np.float64)

        # loop in chunks
        for i in range(0, B, self.chunk):
            end = min(i + self.chunk, B)
            a_chunk = np.ascontiguousarray(a_flat[i:end], dtype=np.float64)
            s_chunk = np.ascontiguousarray(s_flat[i:end], dtype=np.float64)

            # torch computation
            a_t = torch.from_numpy(a_chunk)
            s_t = torch.from_numpy(s_chunk)

            with torch.no_grad():
                log_p_chunk = self.model.log_prob(a_t, s_t)  # [chunk] on device

            # back to numpy
            log_p_chunk = log_p_chunk.detach().cpu().numpy().astype(np.float64)
            log_p_flat[i:end] = log_p_chunk

        # reshape to leading dims
        log_p = log_p_flat.reshape(leading)
        return log_p


def make_gauss(Q, sigma, env_cfg, q_cfg):
    """construct GaussPolicy."""
    return GaussPolicy(Q=Q, sigma=sigma, env_cfg=env_cfg, q_cfg=q_cfg)


def make_policy(spec: dict, env_cfg, q_cfg=None, **ctx):
    """
    Construct policy from spec dict: gauss or flow.

    Input:
      spec: {"kind": "gauss", "sigma": <float>} or {"kind": "flow", "ckpt": <path>}
      env_cfg: PendulumCfg
      q_cfg: QGridCfg (ignored for flow; required for gauss)
      **ctx: unused; for forward-compat (Q passed here for gauss)

    Output:
      GaussPolicy or FlowPolicy instance

    Plan:
      kind = spec.get("kind")
      if kind == "gauss":
        Q = ctx.get("Q")
        sigma = spec["sigma"]
        return GaussPolicy(Q=Q, sigma=sigma, env_cfg=env_cfg, q_cfg=q_cfg)
      elif kind == "flow":
        from src.models.flow.train_flow_policy import load_flow
        ckpt_path = spec["ckpt"]
        model = load_flow(ckpt_path, device="cpu")
        return FlowPolicy(model=model, env_cfg=env_cfg)
      else:
        raise ValueError(f"Unknown policy kind: {kind}")
    """
    kind = spec.get("kind")

    if kind == "gauss":
        Q = ctx.get("Q")
        sigma = spec["sigma"]
        return GaussPolicy(Q=Q, sigma=sigma, env_cfg=env_cfg, q_cfg=q_cfg)

    elif kind == "flow":
        from src.models.flow.train_flow_policy import load_flow
        ckpt_path = spec["ckpt"]
        model = load_flow(ckpt_path, device="cpu")
        return FlowPolicy(model=model, env_cfg=env_cfg)

    else:
        raise ValueError(f"Unknown policy kind: {kind}")
