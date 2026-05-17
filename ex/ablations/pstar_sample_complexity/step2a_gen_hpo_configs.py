"""
Generate random HPO trial configs for all methods and pstar_idx values.

For each (method, pstar_idx) pair, samples hpo_num_trials hyperparameter
configurations from METHOD_SPECS and writes one JSON per trial:

  hpo_configs/{method}/pstar_{pstar_idx}/trial_{i}.json

Run once on a login node before submitting step2b array jobs.
"""

import json
import os
import random

import yaml

from ex.utils.hpo.method_specs import METHOD_SPECS
from ex.utils.hpo.sample import gen_config


def gen_trial_config(method: str, pstar_idx: int, trial_id: int) -> dict:
    """sample one trial config and tag it with pstar_idx."""
    cfg = gen_config(method, trial_id)
    cfg["pstar_idx"] = pstar_idx
    return cfg


def main():
    config = yaml.safe_load(open("ex/pstar_sample_complexity/config.yaml"))
    hpo_configs_dir = config["hpo_configs_dir"]
    num_pstar = len(config["nsamples_pstar_values"])
    num_trials = config["hpo_num_trials"]

    # seed so configs are reproducible; step2b uses its own per-trial seeding
    random.seed(config["seed"])

    total = 0
    for method in METHOD_SPECS:
        for pstar_idx in range(num_pstar):
            out_dir = os.path.join(hpo_configs_dir, method, f"pstar_{pstar_idx}")
            os.makedirs(out_dir, exist_ok=True)

            for trial_id in range(num_trials):
                cfg = gen_trial_config(method, pstar_idx, trial_id)
                path = os.path.join(out_dir, f"trial_{trial_id}.json")
                with open(path, "w") as f:
                    json.dump(cfg, f, indent=2)
                total += 1

    print(f"Generated {total} configs across {len(METHOD_SPECS)} methods x {num_pstar} pstar values x {num_trials} trials")
    print(f"Output: {hpo_configs_dir}")


if __name__ == "__main__":
    main()
