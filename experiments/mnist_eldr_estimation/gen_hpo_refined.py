"""
generate refined hpo configs from top-5 per-alpha ranges.

reads round-1 hpo results, extracts top-5 hyperparams per (method, alpha),
samples densely within those ranges. writes per-alpha config directories.
"""

import argparse
import json
import math
import os
import random
import glob


ALPHA_PAIRS = {
    0: "0:0",
    1: "1:0",
    2: "2:0",
    3: "3:0",
}

# which params use log-uniform vs uniform vs choice
PARAM_TYPES = {
    "TSM": {
        "n_epochs": "uniform_int",
        "lr": "log_uniform",
        "batch_size": "choice",
        "eps": "log_uniform",
    },
    "CTSM": {
        "n_epochs": "uniform_int",
        "lr": "log_uniform",
        "batch_size": "choice",
        "sigma": "log_uniform",
        "eps": "log_uniform",
    },
    "VFM": {
        "n_epochs": "uniform_int",
        "lr": "log_uniform",
        "batch_size": "choice",
        "k": "choice",
        "eps": "log_uniform",
        "integration_steps": "uniform_int",
    },
}


def load_results(results_dir, method):
    """load all trial jsons for a method."""
    pattern = os.path.join(results_dir, method, "trial_*.json")
    trials = []
    for f in glob.glob(pattern):
        with open(f) as fh:
            trials.append(json.load(fh))
    return trials


def extract_ranges(trials, pair_key, top_k=5):
    """
    get top-k trials for a pair, return {param: [values]} dict.

    trials: list of trial dicts
    pair_key: e.g. "0:0"
    top_k: number of top trials to consider

    returns: dict mapping param name to list of values from top-k trials
    """
    ranked = sorted(trials, key=lambda t: t["per_pair_mae"][pair_key])[:top_k]
    params = list(ranked[0]["hyperparams"].keys())
    return {p: [t["hyperparams"][p] for t in ranked] for p in params}


def sample_from_range(values, param_type):
    """
    sample a value from the range defined by top-k values.

    continuous params: sample between min and max of values.
    discrete params: sample uniformly from unique values seen.
    """
    if param_type == "log_uniform":
        lo, hi = min(values), max(values)
        if lo == hi:
            return lo
        return math.exp(random.uniform(math.log(lo), math.log(hi)))

    elif param_type == "uniform_int":
        lo, hi = min(values), max(values)
        return random.randint(lo, hi)

    elif param_type == "choice":
        return random.choice(list(set(values)))

    else:
        raise ValueError(f"unknown param type: {param_type}")


def gen_refined_configs(method, trials, alpha_idx, num_trials, top_k=5):
    """
    generate num_trials configs for a (method, alpha) pair.

    extracts top-k ranges, samples densely within them.
    """
    pair_key = ALPHA_PAIRS[alpha_idx]
    ranges = extract_ranges(trials, pair_key, top_k)
    ptypes = PARAM_TYPES[method]

    configs = []
    for i in range(num_trials):
        hp = {}
        for param, values in ranges.items():
            hp[param] = sample_from_range(values, ptypes[param])
        configs.append({
            "trial_id": i,
            "method": method,
            "alpha_idx": alpha_idx,
            "hyperparams": hp,
        })
    return configs


def main():
    """
    parse args -> load round-1 results -> generate refined configs per alpha.

    output structure: {output_dir}/{method}/alpha_{idx}/trial_{i}.json
    """
    parser = argparse.ArgumentParser(description="generate refined hpo configs")
    parser.add_argument("--method", type=str, required=True,
                        choices=["TSM", "CTSM", "VFM"])
    parser.add_argument("--results-dir", type=str, required=True,
                        help="directory with round-1 results (contains TSM/, CTSM/, VFM/)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="directory to write refined configs")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5,
                        help="number of top trials to derive ranges from")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--alpha-idx", type=int, default=None,
                        help="single alpha index to generate for (default: all)")
    parser.add_argument("--num-eval-pairs", type=int, default=10,
                        help="number of pairs per alpha to evaluate on (sampled w/o replacement)")
    parser.add_argument("--total-pairs", type=int, default=40,
                        help="total available pairs per alpha")
    args = parser.parse_args()

    random.seed(args.seed)
    trials = load_results(args.results_dir, args.method)
    if not trials:
        print(f"no results found for {args.method} in {args.results_dir}")
        return

    alphas = [args.alpha_idx] if args.alpha_idx is not None else [0, 1, 2, 3]

    for aidx in alphas:
        out = os.path.join(args.output_dir, args.method, f"alpha_{aidx}")
        os.makedirs(out, exist_ok=True)

        configs = gen_refined_configs(
            args.method, trials, aidx, args.num_trials, args.top_k
        )
        for cfg in configs:
            path = os.path.join(out, f"trial_{cfg['trial_id']}.json")
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)

        # sample eval pairs without replacement, write eval_pairs.txt
        sampled = random.sample(range(args.total_pairs), args.num_eval_pairs)
        eval_str = ",".join(f"{aidx}:{p}" for p in sorted(sampled))
        ep_path = os.path.join(out, "eval_pairs.txt")
        with open(ep_path, "w") as f:
            f.write(eval_str)

        print(f"{args.method} alpha={aidx}: wrote {args.num_trials} configs, eval={eval_str}")


if __name__ == "__main__":
    main()
