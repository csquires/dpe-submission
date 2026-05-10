"""CLI for picking HPO winners for mnist_uncond."""

import argparse
import os
from pathlib import Path

import yaml

from experiments.utils.winners import collect_winners, write_winners_yaml
from experiments.mnist_uncond.hpo_search_spaces import SEARCH_SPACES


def parse_args():
    """Parse CLI arguments for winner selection."""
    parser = argparse.ArgumentParser(
        description="Select top-K HPO trials per (method, alpha) and write to YAML"
    )
    default_results_dir = os.path.expandvars(
        "/data/user_data/$USER/dpe-submission/mnist_uncond/hpo_results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=default_results_dir,
        help="root HPO results directory (default: $DATA_ROOT/hpo_results)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/mnist_uncond/winners.yaml",
        help="output YAML file path",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0,1,2,3",
        help="comma-separated alpha indices",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="number of top trials to retain per (method, alpha) (default 3)",
    )
    return parser.parse_args()


def main():
    """Parse CLI args, auto-discover methods, collect winners, write YAML."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    alphas = [int(x.strip()) for x in args.alphas.split(",")]

    # auto-discover methods (pass empty list)
    methods = []

    winners = collect_winners(
        hpo_results_dir=results_dir,
        methods=methods,
        alphas=alphas,
        search_spaces=SEARCH_SPACES,
        top_k=args.top_k,
        bucket_axis="alpha_idx",
    )
    write_winners_yaml(winners, output_path)

    method_count = len(winners)
    bucket_count = sum(len(v) for v in winners.values())
    total_trials = sum(len(trials) for d in winners.values() for trials in d.values())
    print(f"wrote {method_count} methods, {bucket_count} total (method, alpha) pairs, {total_trials} total winners")
    print(f"output: {output_path}")


if __name__ == "__main__":
	main()
