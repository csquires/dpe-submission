"""
Generate random HPO trial configs for all methods and KL values.

Writes one JSON per trial to:
  hpo_configs/{method}/kl_{kl_idx}/trial_{i}.json
"""

import json
import math
import os
import random

import yaml

from ex.utils.hpo.method_specs import METHOD_SPECS as SEARCH_SPACES


def _sample_param(spec: tuple):
    """sample a hyperparameter value from a (kind, ...args) spec tuple.

    inlined from the deprecated ex.utils.hpo.sample module; only the
    five spec shapes used by METHOD_SPECS search-space declarations are
    supported.
    """
    kind = spec[0]
    if kind == "log_uniform":
        lo, hi = spec[1], spec[2]
        return math.exp(random.uniform(math.log(lo), math.log(hi)))
    if kind == "log_uniform_int":
        lo, hi = spec[1], spec[2]
        return int(round(math.exp(random.uniform(math.log(lo), math.log(hi)))))
    if kind == "uniform":
        return random.uniform(spec[1], spec[2])
    if kind == "uniform_int":
        return random.randint(spec[1], spec[2])
    if kind == "choice":
        return random.choice(spec[1])
    raise ValueError(f"unknown spec kind: {kind!r}")


def gen_config(registry: dict, method: str, trial_id: int) -> dict:
    """sample a full hyperparameter config for one random-search trial.

    inlined from the deprecated ex.utils.hpo.sample module. used by
    the random-search generator below; the new optuna stack does not call
    this function.
    """
    search_space = registry[method]["base_search_space"]
    hyperparams = {p: _sample_param(spec) for p, spec in search_space.items()}
    return {"trial_id": trial_id, "method": method, "hyperparams": hyperparams}


def main():
    config = yaml.safe_load(open("ex/plugin_dre/config.yaml"))
    hpo_configs_dir = config["hpo_configs_dir"]
    num_kls = len(config["kl_divergences"])
    num_trials = config["hpo_num_trials"]

    random.seed(config["seed"])

    total = 0
    for method in SEARCH_SPACES:
        for kl_idx in range(num_kls):
            out_dir = os.path.join(hpo_configs_dir, method, f"kl_{kl_idx}")
            os.makedirs(out_dir, exist_ok=True)

            for trial_id in range(num_trials):
                trial_cfg = gen_config(SEARCH_SPACES, method, trial_id)
                trial_cfg["kl_idx"] = kl_idx
                path = os.path.join(out_dir, f"trial_{trial_id}.json")
                with open(path, "w") as f:
                    json.dump(trial_cfg, f, indent=2)
                total += 1

    print(
        f"Generated {total} configs across "
        f"{len(SEARCH_SPACES)} methods x {num_kls} KLs x {num_trials} trials"
    )
    print(f"Output: {hpo_configs_dir}")


if __name__ == "__main__":
    main()
