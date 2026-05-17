"""shared hyperparameter sampling for HPO config generators.

single source of truth for random-search sampling, consolidating four
previously-duplicated copies of `sample_param` + `gen_config` across the
ablation and semisynth experiments. all generators sample from the canonical
`METHOD_SPECS[method]["base_search_space"]`.

api:
  sample_param(spec)        -> one value from a (kind, ...args) spec tuple
  gen_config(method, trial_id) -> {"trial_id", "method", "hyperparams"} dict

callers add experiment-specific keys (kl_idx / pstar_idx / alpha_idx) to the
returned dict and handle output paths themselves.
"""

import math
import random

from ex.utils.hpo.method_specs import METHOD_SPECS


def sample_param(spec):
    """sample one hyperparameter value from a (kind, ...args) spec tuple.

    supported spec kinds (the five used by METHOD_SPECS search spaces):
      ("log_uniform", lo, hi)     -> continuous log-uniform in [lo, hi]
      ("log_uniform_int", lo, hi) -> log-uniform rounded to int
      ("uniform", lo, hi)         -> continuous uniform in [lo, hi]
      ("uniform_int", lo, hi)     -> uniform integer in [lo, hi]
      ("choice", options)         -> discrete uniform from list

    returns: sampled value (int, float, or object per spec kind).
    """
    kind = spec[0]

    if kind == "log_uniform":
        _, lo, hi = spec
        return math.exp(random.uniform(math.log(lo), math.log(hi)))

    if kind == "log_uniform_int":
        _, lo, hi = spec
        return int(round(math.exp(random.uniform(math.log(lo), math.log(hi)))))

    if kind == "uniform":
        _, lo, hi = spec
        return random.uniform(lo, hi)

    if kind == "uniform_int":
        _, lo, hi = spec
        return random.randint(lo, hi)

    if kind == "choice":
        _, options = spec
        return random.choice(options)

    raise ValueError(f"unknown distribution type: {kind!r}")


def gen_config(method: str, trial_id: int) -> dict:
    """sample a full hyperparameter config for one random-search trial.

    samples every parameter in METHOD_SPECS[method]["base_search_space"].
    raises KeyError if method is not a METHOD_SPECS key.

    returns: {"trial_id": int, "method": str, "hyperparams": dict}.
    """
    space = METHOD_SPECS[method]["base_search_space"]
    hyperparams = {name: sample_param(spec) for name, spec in space.items()}
    return {"trial_id": trial_id, "method": method, "hyperparams": hyperparams}
