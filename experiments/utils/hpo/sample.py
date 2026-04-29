"""
pure hyperparameter sampling functions for arbitrary HPO registries.

provides registry-agnostic samplers for hyperparameter search spaces.
no io, no global state, no external dependencies beyond math and random.

design: functions take registry as argument to remain independent of any
specific experiment or method configuration.
"""

import math
import random
from typing import Any, Dict, List, Tuple, Union


def sample_param(spec: Tuple) -> Union[int, float, Any]:
    """
    sample a hyperparameter value from a spec tuple.

    spec: tuple with first element defining distribution type and remaining
          elements as distribution parameters.
          - ("log_uniform", lo: float, hi: float)
                continuous log-uniform on [lo, hi]. raises ValueError if
                lo <= 0 or hi <= 0.
          - ("log_uniform_int", lo: float, hi: float)
                log-uniform sampled then rounded to nearest int. raises
                ValueError if lo <= 0 or hi <= 0.
          - ("uniform", lo: float, hi: float)
                continuous uniform on [lo, hi].
          - ("uniform_int", lo: int, hi: int)
                integer uniform on [lo, hi] inclusive.
          - ("choice", options: list)
                discrete uniform sample from options list. raises ValueError
                if options is empty.

    returns: sampled value. type depends on spec type: float for log_uniform/
             uniform, int for log_uniform_int/uniform_int, element of options
             for choice.

    raises:
      ValueError: if spec[0] is unknown distribution type, or if bounds
                  invalid (lo <= 0 for log variants, empty list for choice).

    pseudocode:
      1. extract distribution type from spec[0]
      2. dispatch on type:
         - "log_uniform":
             a. extract lo, hi from spec[1], spec[2]
             b. validate: lo > 0 and hi > 0, else raise ValueError
             c. sample u ~ Uniform(log(lo), log(hi))
             d. return exp(u)
         - "log_uniform_int":
             a. extract lo, hi from spec[1], spec[2]
             b. validate: lo > 0 and hi > 0, else raise ValueError
             c. sample u ~ Uniform(log(lo), log(hi))
             d. return round(exp(u))
         - "uniform":
             a. extract lo, hi from spec[1], spec[2]
             b. return Uniform(lo, hi)
         - "uniform_int":
             a. extract lo, hi from spec[1], spec[2]
             b. return randint(lo, hi) [inclusive on both ends]
         - "choice":
             a. extract options from spec[1]
             b. validate: len(options) > 0, else raise ValueError
             c. return choice(options)
         - else:
             raise ValueError describing unknown type
    """
    dist_type = spec[0]

    if dist_type == "log_uniform":
        lo, hi = spec[1], spec[2]
        if lo <= 0 or hi <= 0:
            raise ValueError(f"log_uniform bounds must be positive; got lo={lo}, hi={hi}")
        log_lo = math.log(lo)
        log_hi = math.log(hi)
        u = random.uniform(log_lo, log_hi)
        return math.exp(u)

    elif dist_type == "log_uniform_int":
        lo, hi = spec[1], spec[2]
        if lo <= 0 or hi <= 0:
            raise ValueError(f"log_uniform bounds must be positive; got lo={lo}, hi={hi}")
        log_lo = math.log(lo)
        log_hi = math.log(hi)
        u = random.uniform(log_lo, log_hi)
        return round(math.exp(u))

    elif dist_type == "uniform":
        lo, hi = spec[1], spec[2]
        return random.uniform(lo, hi)

    elif dist_type == "uniform_int":
        lo, hi = spec[1], spec[2]
        return random.randint(lo, hi)

    elif dist_type == "choice":
        options = spec[1]
        if len(options) == 0:
            raise ValueError("choice options list is empty")
        return random.choice(options)

    else:
        raise ValueError(f"unknown distribution type: {dist_type}")


def gen_config(registry: Dict[str, Dict[str, Any]], method: str,
               trial_id: int) -> Dict[str, Any]:
    """
    generate a complete hpo configuration for one trial.

    looks up method in registry, samples all hyperparameters from its search
    space, and returns structured config dict.

    registry: dict of method name -> method config dict.
              each method config must have key "search_space" mapping to a dict
              of {param_name: spec_tuple}.

    method: string key to look up in registry. must exist, else KeyError.

    trial_id: unique integer identifier for this trial (typically trial index).
              included in returned config for tracking.

    returns: dict with structure
             {
               "trial_id": int,
               "method": str,
               "hyperparams": {param_name: sampled_value, ...}
             }
             hyperparams dict has one entry per parameter in registry[method][
             "search_space"], with values sampled via sample_param.

    raises:
      KeyError: if method not in registry (dict will raise on key access).

    pseudocode:
      1. look up registry[method] -> raises KeyError if method not found
      2. extract registry[method]["search_space"] -> dict of param -> spec
      3. initialize hyperparams = {} (empty dict)
      4. for each (param_name, spec) in search_space.items():
         a. call value = sample_param(spec)
         b. set hyperparams[param_name] = value
      5. return dict with keys:
         - "trial_id": trial_id (int)
         - "method": method (str)
         - "hyperparams": hyperparams (dict)
    """
    method_config = registry[method]
    search_space = method_config["search_space"]

    hyperparams = {}
    for param_name, spec in search_space.items():
        value = sample_param(spec)
        hyperparams[param_name] = value

    return {
        "trial_id": trial_id,
        "method": method,
        "hyperparams": hyperparams,
    }
