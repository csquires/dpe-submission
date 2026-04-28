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

from experiments.mnist_eldr_estimation.hpo_search_spaces import SEARCH_SPACES
from experiments.mnist_eldr_estimation.gen_hpo_configs import sample_param


ALPHA_PAIRS = {
    0: "0:0",
    1: "1:0",
    2: "2:0",
    3: "3:0",
}


def load_results(results_dir, method):
    """load all trial jsons for a method."""
    pattern = os.path.join(results_dir, method, "trial_*.json")
    trials = []
    for f in glob.glob(pattern):
        with open(f) as fh:
            trials.append(json.load(fh))
    return trials


def narrow_spec(spec, top_k_values):
    """
    narrow a search-space spec tuple around observed top-K values.

    spec: tuple from SEARCH_SPACES (e.g., ("log_uniform", lo, hi))
    top_k_values: list of observed values from top-K trials

    returns: narrowed spec tuple of same kind, with bounds set to [min, max] of top_k_values.
             if min == max, collapses to ("choice", [value]).

    dispatch per spec[0]:
      "log_uniform": ("log_uniform", min(top_k_values), max(top_k_values))
                     → if min == max: ("choice", [min])

      "log_uniform_int": ("log_uniform_int", min(top_k_values), max(top_k_values))
                         → if min == max: ("choice", [int(min)])
                         NOTE: preserves log-uniform behavior; previous version downgraded to uniform_int.

      "uniform": ("uniform", min(top_k_values), max(top_k_values))
                 → if min == max: ("choice", [min])

      "uniform_int": ("uniform_int", min(top_k_values), max(top_k_values))
                     (no min==max collapse; discrete already, min/max are integers)

      "choice": ("choice", sorted(set(top_k_values)))
                (return unique values observed, sorted, as new choice set)

      else: raise ValueError(f"unknown spec type: {spec[0]}")
    """
    spec_type = spec[0]
    lo = min(top_k_values)
    hi = max(top_k_values)

    if spec_type == "log_uniform":
        if lo == hi:
            return ("choice", [lo])
        return ("log_uniform", lo, hi)

    elif spec_type == "log_uniform_int":
        if lo == hi:
            return ("choice", [int(lo)])
        return ("log_uniform_int", int(lo), int(hi))

    elif spec_type == "uniform":
        if lo == hi:
            return ("choice", [lo])
        return ("uniform", lo, hi)

    elif spec_type == "uniform_int":
        return ("uniform_int", int(lo), int(hi))

    elif spec_type == "choice":
        return ("choice", sorted(set(top_k_values)))

    else:
        raise ValueError(f"unknown spec type: {spec_type}")


def extract_ranges(trials, pair_key, top_k=5):
    """
    get top-k trials for a pair, return {param: [values]} dict.

    trials: list of trial dicts
    pair_key: e.g. "0:0"
    top_k: number of top trials to consider

    filtering: exclude trials where per_pair_mae[pair_key] is missing, NaN, or Inf.

    returns: dict mapping param name to list of values from top-k trials
    """
    # filter: only keep trials with valid (finite) MAE for this pair_key
    valid_trials = []
    for t in trials:
        if pair_key not in t.get("per_pair_mae", {}):
            continue
        mae = t["per_pair_mae"][pair_key]
        if math.isnan(mae) or math.isinf(mae):
            continue
        valid_trials.append(t)

    # sort and extract top-k
    ranked = sorted(valid_trials, key=lambda t: t["per_pair_mae"][pair_key])[:top_k]
    params = list(ranked[0]["hyperparams"].keys())
    return {p: [t["hyperparams"][p] for t in ranked] for p in params}


def gen_refined_configs(method, trials, alpha_idx, num_trials, top_k=5):
    """
    generate num_trials configs for a (method, alpha) pair.

    extracts top-k ranges, samples densely within them.
    """
    pair_key = ALPHA_PAIRS[alpha_idx]
    ranges = extract_ranges(trials, pair_key, top_k)
    broad_specs = SEARCH_SPACES[method]["search_space"]

    configs = []
    for i in range(num_trials):
        hp = {}
        for param, values in ranges.items():
            broad_spec = broad_specs[param]
            narrowed_spec = narrow_spec(broad_spec, values)
            hp[param] = sample_param(narrowed_spec)
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
                        choices=list(SEARCH_SPACES.keys()))
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
    parser.add_argument("--total-pairs", type=int, default=10,
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
