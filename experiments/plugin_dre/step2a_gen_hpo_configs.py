"""
Generate random HPO trial configs for all methods and KL values.

Writes one JSON per trial to:
  hpo_configs/{method}/kl_{kl_idx}/trial_{i}.json
"""

import json
import os
import random

import yaml

from experiments.plugin_dre.hpo_search_spaces import SEARCH_SPACES
from experiments.utils.hpo.sample import gen_config


def main():
    config = yaml.safe_load(open("experiments/plugin_dre/config.yaml"))
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
