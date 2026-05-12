"""
pick_winners.py (dbpedia cond-flow): aggregate HPO trial results and select best per
(method, alpha). thin shim around mnist_eldr_estimation.pick_winners — reuses
all the library functions (load_trials, best_trial_for_alpha, collect_winners,
write_winners_yaml) and only overrides the CLI defaults to point at dbpedia cond-flow
paths.
"""

import argparse
import os
from pathlib import Path

from experiments.mnist_eldr_estimation.pick_winners import (
    collect_winners,
    write_winners_yaml,
)


def main():
    """parse CLI args, auto-discover methods from results_dir, collect winners,
    write YAML, print summary. defaults point at dbpedia cond-flow paths."""
    parser = argparse.ArgumentParser(
        description="Select best HPO trials per (method, alpha) and write to YAML (dbpedia cond-flow)"
    )
    default_results_dir = os.path.expandvars(
        "$DPE_DATA_ROOT/dbpedia_eldr_cond_flow/hpo_results"
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
        default="experiments/dbpedia_eldr_cond_flow/winners.yaml",
        help="output YAML file path",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0,1,2,3",
        help="comma-separated alpha indices",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    alphas = [int(x.strip()) for x in args.alphas.split(",")]

    # auto-discover methods (filtered against SEARCH_SPACES inside collect_winners)
    methods: list[str] = []

    winners = collect_winners(results_dir, methods, alphas)
    write_winners_yaml(winners, output_path)

    method_count = len(winners)
    alpha_count = sum(len(v) for v in winners.values())
    print(f"wrote {method_count} methods, {alpha_count} total (method, alpha) pairs")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
