"""
pick_winners.py (dbpedia eldr): aggregate HPO trial results and select best per
(method, alpha). thin shim around experiments.utils.winners — reuses all the
library functions (load_trials, best_trial_for_alpha, collect_winners,
write_winners_yaml) and only overrides the CLI defaults to point at dbpedia eldr
paths.
"""

import argparse
import os
from pathlib import Path

from experiments.utils.winners import (
    collect_winners,
    write_winners_yaml,
)
from experiments.dbpedia.hpo_search_spaces import SEARCH_SPACES


def main():
    """parse CLI args, auto-discover methods from results_dir, collect winners,
    write YAML, print summary. defaults point at dbpedia eldr paths."""
    parser = argparse.ArgumentParser(
        description="Select best HPO trials per (method, alpha) and write to YAML (dbpedia eldr)"
    )
    default_results_dir = os.path.expandvars(
        "/data/user_data/$USER/dpe-submission/dbpedia_eldr/hpo_results"
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
        default="experiments/dbpedia/winners.yaml",
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

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    alphas = [int(x.strip()) for x in args.alphas.split(",")]

    # auto-discover methods
    methods = []

    winners = collect_winners(
        results_dir,
        methods,
        alphas,
        search_spaces=SEARCH_SPACES,
        top_k=args.top_k,
        bucket_axis="alpha_idx",
    )
    write_winners_yaml(winners, output_path)

    method_count = len(winners)
    alpha_count = sum(len(v) for v in winners.values())
    print(f"wrote {method_count} methods, {alpha_count} total (method, alpha) pairs")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
