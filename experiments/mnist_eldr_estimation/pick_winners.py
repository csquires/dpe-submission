"""
pick_winners.py: aggregate HPO trial results and select best per (method, alpha).

main loop: discover methods from hpo_results/ subdirs → for each (method, alpha):
check refined dir first, fallback to broad → load all trials → compute median MAE
over per-pair entries for that alpha → select trial with min median (tie-break by
trial_id) → write hierarchical winners.yaml with trial info and hyperparams.
"""

import argparse
import json
import math
import os
from pathlib import Path

import yaml


def load_trials(results_dir: Path, method: str, source: str, alpha: int) -> list[dict]:
	"""
	load all trial JSONs for (method, source, alpha).
	source in {'refined', 'broad'}. refined → hpo_results/refined/<method>/alpha_<alpha>/.
	broad → hpo_results/<method>/ (ignores alpha, returns all broad trials).
	returns trials sorted by trial_id (ascending, lexicographic).
	returns empty list if dir does not exist.
	"""
	if source == "refined":
		trial_dir = results_dir / "refined" / method / f"alpha_{alpha}"
	elif source == "broad":
		trial_dir = results_dir / method
	else:
		raise ValueError(f"source must be 'refined' or 'broad', got {source}")

	if not trial_dir.exists():
		return []

	trials = []
	for trial_json in sorted(trial_dir.glob("trial_*.json")):
		try:
			with open(trial_json) as f:
				trial = json.load(f)
		except (json.JSONDecodeError, OSError) as e:
			print(f"warning: skipping corrupt trial {trial_json}: {e}")
			continue
		trials.append(trial)

	# sort by trial_id (numeric: trial_id is stored as int by gen_hpo_configs)
	trials.sort(key=lambda t: int(t["trial_id"]))
	return trials


def best_trial_for_alpha(trials: list[dict], alpha: int) -> dict | None:
	"""
	select best trial for given alpha.
	filter per-pair MAEs: remove NaN and Inf before computing median.
	compute median MAE for each trial over keys matching f"{alpha}:{pair_idx}".
	return trial with min median MAE (tie-break: smallest trial_id).
	return None if all trials have zero usable MAE entries for alpha.
	"""
	if not trials:
		return None

	# compute median MAE for each trial
	trial_scores = []
	for trial in trials:
		mae_values = _alpha_maes(trial, alpha)
		if not mae_values:
			continue
		trial_scores.append((_median(mae_values), int(trial["trial_id"]), trial))

	if not trial_scores:
		return None

	# min by median MAE, then by trial_id (numeric)
	trial_scores.sort(key=lambda x: (x[0], x[1]))
	_, _, best = trial_scores[0]
	return best


def _alpha_maes(trial: dict, alpha: int) -> list:
	"""extract finite per_pair_mae values for the given alpha by parsing keys.

	the per_pair_mae dict is keyed by "<alpha>:<pair>" strings. discover all
	such pairs dynamically (no hardcoded upper bound).
	"""
	per_pair_mae = trial.get("per_pair_mae", {})
	prefix = f"{alpha}:"
	out = []
	for key, val in per_pair_mae.items():
		if not key.startswith(prefix):
			continue
		if isinstance(val, (int, float)) and math.isfinite(val):
			out.append(val)
	return out


def _median(values: list) -> float:
	"""standard median over a non-empty list of floats."""
	s = sorted(values)
	n = len(s)
	mid = n // 2
	if n % 2 == 1:
		return s[mid]
	return (s[mid] + s[mid - 1]) / 2


def collect_winners(
	results_dir: Path, methods: list[str], alphas: list[int]
) -> dict:
	"""
	aggregate winners across all (method, alpha) pairs.
	auto-discover methods if methods list is empty: list immediate subdirs of results_dir.
	returns hierarchical dict {method: {alpha: {trial_id, mae_median, source, hyperparams}}}.
	methods/alphas absent from disk are absent from output.
	"""
	if not methods:
		# auto-discover: list immediate subdirs of results_dir, exclude 'refined',
		# and intersect with registered methods so stray dirs do not pollute winners.
		from experiments.mnist_eldr_estimation.hpo_search_spaces import SEARCH_SPACES
		registered = set(SEARCH_SPACES.keys())
		discovered = [
			d.name
			for d in results_dir.iterdir()
			if d.is_dir() and d.name != "refined"
		]
		methods = [m for m in discovered if m in registered]
		stray = [m for m in discovered if m not in registered]
		if stray:
			print(f"warning: ignoring unregistered method dirs: {sorted(stray)}")

	winners = {}

	for method in sorted(methods):
		method_winners = {}

		for alpha in alphas:
			# try refined first, then broad
			trials = load_trials(results_dir, method, "refined", alpha)
			source = "refined"

			if not trials:
				trials = load_trials(results_dir, method, "broad", alpha)
				source = "broad"

			if not trials:
				continue

			best = best_trial_for_alpha(trials, alpha)
			if best is None:
				continue

			# compute best median MAE for output
			mae_values = _alpha_maes(best, alpha)
			mae_median = _median(mae_values) if mae_values else float("nan")

			method_winners[alpha] = {
				"trial_id": int(best["trial_id"]),
				"mae_median": float(mae_median),
				"source": source,
				"hyperparams": best.get("hyperparams", {}),
			}

		if method_winners:
			winners[method] = method_winners

	return winners


def write_winners_yaml(winners: dict, output_path: Path) -> None:
	"""
	atomic temp-file write + os.replace.
	write to <output>.tmp, fsync, then os.replace.
	use yaml.dump with default_flow_style=False, sort_keys=True.
	"""
	tmp = output_path.with_suffix(output_path.suffix + ".tmp")
	with tmp.open("w") as f:
		yaml.dump(winners, f, default_flow_style=False, sort_keys=True)
		f.flush()
		os.fsync(f.fileno())
	os.replace(tmp, output_path)


def main():
	"""
	parse CLI args, auto-discover methods from results_dir, collect winners,
	write YAML, print summary.
	"""
	parser = argparse.ArgumentParser(
		description="Select best HPO trials per (method, alpha) and write to YAML"
	)
	default_results_dir = os.path.expandvars(
		"/data/user_data/$USER/dpe-submission/mnist_eldr_estimation/hpo_results"
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
		default="experiments/mnist_eldr_estimation/winners.yaml",
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

	# auto-discover methods
	methods = []

	winners = collect_winners(results_dir, methods, alphas)
	write_winners_yaml(winners, output_path)

	method_count = len(winners)
	alpha_count = sum(len(v) for v in winners.values())
	print(f"wrote {method_count} methods, {alpha_count} total (method, alpha) pairs")
	print(f"output: {output_path}")


if __name__ == "__main__":
	main()
