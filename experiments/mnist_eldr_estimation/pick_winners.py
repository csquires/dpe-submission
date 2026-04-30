"""
pick_winners.py: aggregate HPO trial results and select top-K per (method, bucket).

main loop: discover methods from hpo_results/ subdirs → apply legacy alias resolution
→ for each (method, bucket): check refined dir first, fallback to broad → load all trials
→ compute median MAE over per-bucket entries → select top-k by mae (sorted ascending) →
write hierarchical winners.yaml as lists of trial dicts with trial_id, mae, hyperparams.

shared by mnist_eldr_estimation, mnist_eldr_cond_flow, dbpedia_eldr_cond_flow via
import from experiments.mnist_eldr_estimation.pick_winners.
"""

import argparse
import json
import math
import os
from pathlib import Path

import yaml

from experiments.utils.hpo.registry import LEGACY_ALIASES


def load_trials(results_dir: Path, method: str, source: str, bucket_idx: int, bucket_axis: str) -> list[dict]:
	"""
	load all trial JSONs for (method, source, bucket_idx, bucket_axis).

	source in {'refined', 'broad'}.
	refined → hpo_results/refined/<method>/<bucket_axis>_<bucket_idx>/.
	broad → hpo_results/<method>/ (ignores bucket, returns all broad trials).
	bucket_axis in {'alpha_idx', 'kl_idx', ...} (used to construct path).

	returns trials sorted by trial_id (ascending, numeric).
	returns empty list if dir does not exist.
	"""
	if source == "refined":
		trial_dir = results_dir / "refined" / method / f"{bucket_axis}_{bucket_idx}"
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

	trials.sort(key=lambda t: int(t["trial_id"]))
	return trials


def top_trials_for_alpha(trials: list[dict], bucket_idx: int, k: int, bucket_axis: str = "alpha_idx") -> list[dict]:
	"""
	select top-k trials for given bucket, sorted ascending by mae (best first).

	filter per-pair MAEs: remove NaN and Inf before computing median.
	compute median MAE for each trial over keys matching f"{bucket_idx}:<pair_idx>".
	return list of k trial dicts sorted by (mae, trial_id), ascending (best first).
	return all trials if fewer than k have finite MAE entries.
	return empty list if all trials have zero usable MAE entries for bucket.

	bucket_axis argument is kept for compatibility but not used in key parsing
	(keys are always f"{bucket_idx}:<pair_idx>" regardless of bucket_axis name).
	"""
	if not trials:
		return []

	trial_scores = []
	for trial in trials:
		mae_values = _alpha_maes(trial, bucket_idx)
		if not mae_values:
			continue
		trial_scores.append((_median(mae_values), int(trial["trial_id"]), trial))

	if not trial_scores:
		return []

	trial_scores.sort(key=lambda x: (x[0], x[1]))
	return [t for _, _, t in trial_scores[:k]]


def _alpha_maes(trial: dict, bucket_idx: int) -> list:
	"""
	extract finite per_pair_mae values for the given bucket by parsing keys.

	the per_pair_mae dict is keyed by "<bucket_idx>:<pair>" strings.
	discover all such pairs dynamically (no hardcoded upper bound).
	"""
	per_pair_mae = trial.get("per_pair_mae", {})
	prefix = f"{bucket_idx}:"
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
	hpo_results_dir: Path,
	methods: list[str],
	alphas: list[int],
	search_spaces: dict,
	top_k: int = 3,
	bucket_axis: str = "alpha_idx",
	legacy_aliases: dict[str, str] | None = None,
) -> dict:
	"""
	aggregate winners across all (method, bucket) pairs.

	auto-discover methods if methods list is empty: list immediate subdirs of hpo_results_dir,
	apply legacy alias resolution, and intersect with search_spaces keys.

	args:
		hpo_results_dir: root HPO results directory.
		methods: list of method names to process. if empty, auto-discover.
		alphas: list of bucket indices (alpha_idx, kl_idx, etc).
		search_spaces: dict of canonical method names (used for intersection in auto-discovery).
		top_k: number of top trials to retain per (method, bucket). default 3.
		bucket_axis: name of bucket axis for path construction (e.g., "alpha_idx", "kl_idx").
		legacy_aliases: dict mapping old dir names to canonical names (e.g., MHTTDRE → MultiHeadTriangularTDRE).
			if None, uses LEGACY_ALIASES from registry. this parameter allows tests to override.

	returns:
		hierarchical dict {method: {bucket: [{trial_id, mae, source, hyperparams}, ...]}}
		where each bucket maps to a sorted list of top-k trial dicts (ascending by mae).
		methods/buckets absent from disk are absent from output.
	"""
	if legacy_aliases is None:
		legacy_aliases = LEGACY_ALIASES

	auto_discover = not methods
	alias_dirs = {}  # alias_name -> canonical_name for this run

	if auto_discover:
		# auto-discover: list immediate subdirs of hpo_results_dir, exclude 'refined',
		# apply legacy alias resolution, and intersect with search_spaces.
		# search_spaces contains both canonical and alias keys (alias auto-injection
		# in registry.py); subtract aliases so dir-name matching prefers canonical.
		canonical = set(search_spaces.keys()) - set(legacy_aliases.keys())
		discovered = [
			d.name
			for d in hpo_results_dir.iterdir()
			if d.is_dir() and d.name != "refined"
		]

		# re-key alias dirs to canonical; collect all canonical dirs to process
		dirs_by_canonical = {}
		for d in discovered:
			if d in canonical:
				# already canonical
				canonical_name = d
			elif d in legacy_aliases:
				# alias: map to canonical
				canonical_name = legacy_aliases[d]
				alias_dirs[d] = canonical_name
				if canonical_name in canonical:
					print(f"info: treating legacy alias {d}/ as {canonical_name}")
				else:
					# canonical not in search_spaces; skip
					print(f"warning: legacy alias {d} maps to {canonical_name}, not in search_spaces; skipping")
					continue
			else:
				# stray dir, not canonical and not aliased
				continue

			if canonical_name not in dirs_by_canonical:
				dirs_by_canonical[canonical_name] = []
			dirs_by_canonical[canonical_name].append(d)

		methods = sorted(dirs_by_canonical.keys())

	winners = {}

	for method in sorted(methods):
		method_winners = {}

		for bucket_idx in alphas:
			# try refined first, then broad
			trials = load_trials(hpo_results_dir, method, "refined", bucket_idx, bucket_axis)
			source = "refined"

			if not trials:
				trials = load_trials(hpo_results_dir, method, "broad", bucket_idx, bucket_axis)
				source = "broad"

			# merge legacy alias dirs unconditionally (auto-discover mode only).
			# do this BEFORE the "no trials" check so a method that exists only
			# under an alias dir is still discovered.
			if auto_discover:
				existing_ids = {int(t["trial_id"]) for t in trials}
				for old_name, canonical_name in alias_dirs.items():
					if canonical_name == method:
						alias_trials = load_trials(hpo_results_dir, old_name, "broad", bucket_idx, bucket_axis)
						for t in alias_trials:
							tid = int(t["trial_id"])
							if tid not in existing_ids:
								trials.append(t)
								existing_ids.add(tid)
				if trials and source == "broad" and not (hpo_results_dir / method).exists():
					source = "broad"

			if not trials:
				continue

			top = top_trials_for_alpha(trials, bucket_idx, top_k, bucket_axis)
			if not top:
				continue

			method_winners[bucket_idx] = [
				{
					"trial_id": int(t["trial_id"]),
					"mae": float(_median(_alpha_maes(t, bucket_idx))),
					"source": source,
					"hyperparams": t.get("hyperparams", {}),
				}
				for t in top
			]

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
	from experiments.mnist_eldr_estimation.hpo_search_spaces import SEARCH_SPACES

	parser = argparse.ArgumentParser(
		description="Select top-K HPO trials per (method, alpha) and write to YAML"
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
	bucket_count = sum(len(v) for v in winners.values())
	total_trials = sum(len(trials) for d in winners.values() for trials in d.values())
	print(f"wrote {method_count} methods, {bucket_count} total (method, alpha) pairs, {total_trials} total winners")
	print(f"output: {output_path}")


if __name__ == "__main__":
	main()
