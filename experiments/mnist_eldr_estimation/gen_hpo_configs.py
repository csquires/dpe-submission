"""
generate random hpo configs for tsm, ctsm, or vfm.

samples from predefined search spaces, writes one json per trial.
"""

import argparse
import json
import math
import os
import random


SEARCH_SPACES = {
    "TSM": {
        "n_epochs":   ("log_uniform_int", 750, 1500),
        "lr":         ("log_uniform", 3e-4, 3e-3),
        "batch_size": ("choice", [64, 128, 256]),
        "eps":        ("log_uniform", 1e-6, 1e-4),
    },
    "CTSM": {
        "n_epochs":   ("log_uniform_int", 750, 1500),
        "lr":         ("log_uniform", 3e-4, 3e-3),
        "batch_size": ("choice", [64, 128, 256]),
        "sigma":      ("log_uniform", 0.3, 3.0),
        "eps":        ("log_uniform", 3e-4, 3e-3),
    },
    "VFM": {
        "n_epochs":          ("log_uniform_int", 750, 1500),
        "lr":                ("log_uniform", 5e-4, 3e-3),
        "batch_size":        ("choice", [64, 128, 256]),
        "k":                 ("choice", [10, 20, 40]),
        "eps":               ("log_uniform", 1e-3, 5e-3),
        "integration_steps": ("uniform_int", 500, 5000),
    },
    "FMDRE": {
        "n_epochs":          ("log_uniform_int", 500, 1500),
        "lr":                ("log_uniform", 5e-4, 3e-3),
        "batch_size":        ("choice", [128, 256, 512]),
        "eps":               ("log_uniform", 1e-3, 5e-2),
        "integration_steps": ("uniform_int", 1000, 10000),
        "hidden_dim":        ("choice", [128, 256, 512]),
        "score_weight":      ("log_uniform", 0.1, 10.0),
    },
    "FMDRE_S2": {
        "n_epochs":          ("log_uniform_int", 500, 1500),
        "lr":                ("log_uniform", 5e-4, 3e-3),
        "batch_size":        ("choice", [128, 256, 512]),
        "eps":               ("log_uniform", 1e-3, 5e-2),
        "integration_steps": ("uniform_int", 1000, 10000),
        "hidden_dim":        ("choice", [128, 256, 512]),
        "score_weight":      ("log_uniform", 0.1, 10.0),
        "p_uncond":          ("uniform", 0.1, 0.9),
    },
    "MHTTDRE": {
        "learning_rate":     ("log_uniform", 1e-4, 1e-2),
        "num_epochs":        ("log_uniform_int", 100, 1000),
        "hidden_dim":        ("choice", [16, 32, 64, 128]),
        "head_dim":          ("choice", [10, 20, 40]),
        "num_shared_layers": ("choice", [1, 2, 3]),
    },
}


def sample_param(spec):
    """
    sample a hyperparameter value according to spec.

    spec: tuple with format matching SEARCH_SPACES entries
          - ("log_uniform", lo, hi): continuous log-uniform in [lo, hi]
          - ("log_uniform_int", lo, hi): log-uniform rounded to int
          - ("uniform", lo, hi): continuous uniform in [lo, hi]
          - ("uniform_int", lo, hi): uniform integer in [lo, hi]
          - ("choice", options): discrete uniform from list

    returns: sampled value (int, float, or object depending on spec type)
    """
    dist_type = spec[0]

    if dist_type == "log_uniform":
        _, lo, hi = spec
        log_val = random.uniform(math.log(lo), math.log(hi))
        return math.exp(log_val)

    elif dist_type == "log_uniform_int":
        _, lo, hi = spec
        log_val = random.uniform(math.log(lo), math.log(hi))
        return round(math.exp(log_val))

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


def gen_config(method, trial_id):
    """
    generate a complete hpo configuration for one trial.

    method: "TSM", "CTSM", "VFM", "FMDRE", "FMDRE_S2", or "MHTTDRE"
    trial_id: unique integer identifier for this trial

    returns: dict with structure {"trial_id": int, "method": str, "hyperparams": dict}
             hyperparams contains sampled values for all parameters in search space
    """
    space = SEARCH_SPACES[method]  # raises KeyError if invalid method

    hyperparams = {}
    for param_name, spec in space.items():
        hyperparams[param_name] = sample_param(spec)

    return {
        "trial_id": trial_id,
        "method": method,
        "hyperparams": hyperparams
    }


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
                       help="method: TSM, CTSM, VFM, FMDRE, FMDRE_S2, or MHTTDRE")
    parser.add_argument("--num-trials", type=int, required=True,
                       help="number of trials to generate")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="directory to write json configs")
    parser.add_argument("--seed", type=int, default=42,
                       help="random seed for reproducibility")

    args = parser.parse_args()

    # validate method
    if args.method not in ["TSM", "CTSM", "VFM", "FMDRE", "FMDRE_S2", "MHTTDRE"]:
        parser.error(f"invalid method: {args.method}. must be TSM, CTSM, VFM, FMDRE, FMDRE_S2, or MHTTDRE")

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
