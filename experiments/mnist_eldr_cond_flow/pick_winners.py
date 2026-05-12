"""
pick_winners.py (cond-flow): aggregate HPO trial results and select top-K per
(method, alpha). thin shim around mnist_eldr_estimation.pick_winners — reuses
all the library functions (load_trials, top_trials_for_alpha, collect_winners,
write_winners_yaml) and only overrides the CLI defaults to point at cond-flow
paths. reads cond_flow config to infer alpha indices.
"""

import argparse
import os
from pathlib import Path

import yaml

from experiments.mnist_eldr_estimation.pick_winners import (
	collect_winners,
	write_winners_yaml,
)
from experiments.mnist_eldr_cond_flow.hpo_search_spaces import SEARCH_SPACES


def main():
	"""parse CLI args, read cond_flow config, auto-discover methods from
	results_dir, collect top-K winners, write YAML. defaults point at cond-flow paths."""
	parser = argparse.ArgumentParser(
		description="Select top-K HPO trials per (method, alpha) and write to YAML (cond-flow)"
	)
	parser.add_argument(
		"--config",
		type=str,
		default="experiments/mnist_eldr_cond_flow/config.yaml",
		help="cond_flow config YAML",
	)
	default_results_dir = os.path.expandvars(
		"$DPE_DATA_ROOT/mnist_eldr_cond_flow/hpo_results"
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
		default="experiments/mnist_eldr_cond_flow/winners.yaml",
		help="output YAML file path",
	)
	parser.add_argument(
		"--alphas",
		type=str,
		default=None,
		help="comma-separated alpha indices (default: inferred from config)",
	)
	parser.add_argument(
		"--top-k",
		type=int,
		default=3,
		help="number of top trials to retain per (method, alpha) (default 3)",
	)

	args = parser.parse_args()

	# read config to infer alpha indices if not overridden
	if args.alphas is None:
		config_path = Path(args.config)
		with open(config_path) as f:
			config = yaml.safe_load(f)
		num_alphas = len(config.get("alphas", [0.1, 0.3, 0.9, 2.7]))
		alphas = list(range(num_alphas))
	else:
		alphas = [int(x.strip()) for x in args.alphas.split(",")]

	results_dir = Path(args.results_dir)
	output_path = Path(args.output)

	# auto-discover methods (filtered against SEARCH_SPACES inside collect_winners)
	methods: list[str] = []

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
	bucket_count = sum(len(v) for v in winners.values())
	total_trials = sum(len(trials) for d in winners.values() for trials in d.values())
	print(f"wrote {method_count} methods, {bucket_count} total (method, alpha) pairs, {total_trials} total winners")
	print(f"output: {output_path}")


if __name__ == "__main__":
	main()
