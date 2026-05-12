"""
Generate random HPO trial configs for all methods and pstar_idx values.

For each (method, pstar_idx) pair, samples hpo_num_trials hyperparameter
configurations from the search space defined in hpo_search_spaces.py and
writes one JSON per trial:

  hpo_configs/{method}/pstar_{pstar_idx}/trial_{i}.json

Run once on a login node before submitting step2b array jobs.
"""

import json
import math
import os
import random

import yaml

from experiments.ablations.pstar_sample_complexity.hpo_search_spaces import SEARCH_SPACES


def sample_param(spec):
    """Sample one hyperparameter value from a search space spec tuple."""
    dist_type = spec[0]

    if dist_type == "log_uniform":
        _, lo, hi = spec
        return math.exp(random.uniform(math.log(lo), math.log(hi)))

    elif dist_type == "log_uniform_int":
        _, lo, hi = spec
        return round(math.exp(random.uniform(math.log(lo), math.log(hi))))

    elif dist_type == "uniform":
        _, lo, hi = spec
        return random.uniform(lo, hi)

    elif dist_type == "uniform_int":
        _, lo, hi = spec
        return random.randint(lo, hi)

    elif dist_type == "choice":
        _, options = spec
        return random.choice(options)

    else:
        raise ValueError(f"unknown distribution type: {dist_type}")


def gen_config(method: str, pstar_idx: int, trial_id: int) -> dict:
    """Sample all hyperparams for one trial and return as a config dict."""
    space = SEARCH_SPACES[method]["search_space"]
    hyperparams = {name: sample_param(spec) for name, spec in space.items()}
    return {
        "method": method,
        "pstar_idx": pstar_idx,
        "trial_id": trial_id,
        "hyperparams": hyperparams,
    }


def main():
    config = yaml.safe_load(open("experiments/pstar_sample_complexity/config.yaml"))
    hpo_configs_dir = config["hpo_configs_dir"]
    num_pstar = len(config["nsamples_pstar_values"])
    num_trials = config["hpo_num_trials"]

    # seed so configs are reproducible; step2b uses its own per-trial seeding
    random.seed(config["seed"])

    total = 0
    for method in SEARCH_SPACES:
        for pstar_idx in range(num_pstar):
            out_dir = os.path.join(hpo_configs_dir, method, f"pstar_{pstar_idx}")
            os.makedirs(out_dir, exist_ok=True)

            for trial_id in range(num_trials):
                cfg = gen_config(method, pstar_idx, trial_id)
                path = os.path.join(out_dir, f"trial_{trial_id}.json")
                with open(path, "w") as f:
                    json.dump(cfg, f, indent=2)
                total += 1

    print(f"Generated {total} configs across {len(SEARCH_SPACES)} methods x {num_pstar} pstar values x {num_trials} trials")
    print(f"Output: {hpo_configs_dir}")


if __name__ == "__main__":
    main()
