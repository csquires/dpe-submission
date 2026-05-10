"""
shared HPO winners aggregation utility.

provides generic functions to load trial results, compute median MAE per bucket,
select top-k trials, and write hierarchical winners YAML. no experiment-specific logic.
"""

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
