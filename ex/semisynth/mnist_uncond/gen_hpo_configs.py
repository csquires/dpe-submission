"""
generate random hpo configs for tsm, ctsm, or vfm.

samples from METHOD_SPECS search spaces, writes one json per trial.
"""

import argparse
import json
import os
import random

from ex.utils.hpo.method_specs import METHOD_SPECS
from ex.utils.hpo.sample import gen_config


def main():
    """
    entry point: parse args, set seed, generate configs, write json files.

    args:
      --method: TSM, CTSM, VFM, FMDRE, FMDRE_S2, or MHTTDRE
      --num-trials: number of configs to generate
      --output-dir: directory to write json files
      --seed: random seed (default 42)

    output: one json file per trial named trial_{i}.json
    """
    parser = argparse.ArgumentParser(description="generate hpo configs")
    parser.add_argument("--method", type=str, required=True,
                       choices=list(METHOD_SPECS.keys()),
                       help="method name; must be a key in METHOD_SPECS")
    parser.add_argument("--num-trials", type=int, required=True,
                       help="number of trials to generate")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="directory to write json configs")
    parser.add_argument("--seed", type=int, default=42,
                       help="random seed for reproducibility")

    args = parser.parse_args()

    # set seed
    random.seed(args.seed)

    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # generate and write configs
    for i in range(args.num_trials):
        config = gen_config(args.method, trial_id=i)
        filepath = os.path.join(args.output_dir, f"trial_{i}.json")
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    # print summary
    print(f"Generated {args.num_trials} configs for {args.method}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
