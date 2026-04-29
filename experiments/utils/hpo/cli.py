"""
generic registry-agnostic CLI for generating HPO trial configs.

generates random hyperparameter configurations for any experiment by importing
a registry module (hpo_search_spaces.py equivalent) at runtime. outputs one json
per trial. registry module must expose SEARCH_SPACES dict with shape:
  {method_name: {"search_space": {...}, "builder": ..., "requires_pstar": ...}}
"""

import argparse
import importlib
import json
import os
import random

from experiments.utils.hpo.sample import gen_config


def main():
	"""
	entry point: parse args, import registry, validate method, generate configs.

	exit codes:
	  0: success
	  1 or 2: error (see argparse convention for 2)
	"""
	parser = argparse.ArgumentParser(
		description="generate hpo configs from registry module"
	)
	parser.add_argument(
		"--registry-module",
		type=str,
		required=True,
		help="dotted module path (e.g., experiments.eig_estimation.hpo_search_spaces)",
	)
	parser.add_argument(
		"--method",
		type=str,
		required=True,
		help="method name; must be key in SEARCH_SPACES of registry module",
	)
	parser.add_argument(
		"--num-trials",
		type=int,
		required=True,
		help="number of hpo configs to generate",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		required=True,
		help="directory to write trial_{i}.json files",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="random seed for reproducibility (default: 42)",
	)

	args = parser.parse_args()

	# import registry module dynamically
	try:
		registry_module = importlib.import_module(args.registry_module)
	except ModuleNotFoundError:
		parser.error(f"registry module not found: {args.registry_module}")
		return 1

	# extract SEARCH_SPACES from module
	try:
		search_spaces = getattr(registry_module, "SEARCH_SPACES")
	except AttributeError:
		parser.error(
			f"registry module {args.registry_module} missing SEARCH_SPACES dict"
		)
		return 1

	# validate method is in SEARCH_SPACES
	if args.method not in search_spaces:
		valid_methods = ", ".join(sorted(search_spaces.keys()))
		parser.error(
			f"method '{args.method}' not in registry. valid methods: {valid_methods}"
		)
		return 1

	# create output directory
	try:
		os.makedirs(args.output_dir, exist_ok=True)
	except IOError as e:
		parser.error(f"cannot create output directory: {e}")
		return 1

	# set random seed for reproducibility
	random.seed(args.seed)

	# generate trial configs
	for i in range(args.num_trials):
		config = gen_config(search_spaces, args.method, trial_id=i)
		filepath = os.path.join(args.output_dir, f"trial_{i}.json")
		with open(filepath, "w") as f:
			json.dump(config, f, indent=2)

	# print summary
	print(f"Generated {args.num_trials} configs for method: {args.method}")
	print(f"Seed: {args.seed}")
	print(f"Output directory: {args.output_dir}")
	print(f"Registry module: {args.registry_module}")


if __name__ == "__main__":
	main()
