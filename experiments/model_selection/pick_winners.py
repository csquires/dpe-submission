"""aggregate HPO trial results; rank by median MAE per KL bucket; emit winners.yaml."""
import argparse
import json
import logging
import os
from pathlib import Path
from statistics import median

import yaml

from experiments.model_selection.hpo_search_spaces import SEARCH_SPACES
from experiments.utils.hpo.registry import LEGACY_ALIASES

CONFIG_PATH = "experiments/model_selection/config.yaml"

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """load config from YAML. expand ${DPE_DATA_ROOT} env var.

    returns dict with keys: hpo_results_dir, num_instances_per_kl, top_k, kl_distances.
    top_k defaults to 3 if missing.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # expand env vars
    config["hpo_results_dir"] = os.path.expandvars(config["hpo_results_dir"])
    if "winners_path" in config:
        config["winners_path"] = os.path.expandvars(config["winners_path"])

    # validate
    if not os.path.isabs(config["hpo_results_dir"]):
        raise ValueError(f"hpo_results_dir must be absolute: {config['hpo_results_dir']}")

    if not isinstance(config["num_instances_per_kl"], int):
        raise ValueError(f"num_instances_per_kl must be int, got {type(config['num_instances_per_kl'])}")

    if not isinstance(config["kl_distances"], list):
        raise ValueError(f"kl_distances must be list, got {type(config['kl_distances'])}")

    # set default top_k
    if "top_k" not in config:
        config["top_k"] = 3

    return config


def resolve_methods(hpo_results_dir: Path) -> list[str]:
    """discover canonical method names from disk + registry.

    iterate SEARCH_SPACES.keys() (includes 12 canonical + 5 aliases).
    deduplicate by entry identity using id(): only canonical entry refs.
    filter to methods whose dir exists under hpo_results_dir.
    return sorted list of canonical method names (no aliases).
    """
    seen_ids = set()
    canonical_methods = []

    for method_name, entry in SEARCH_SPACES.items():
        entry_id = id(entry)
        if entry_id not in seen_ids:
            seen_ids.add(entry_id)
            canonical_methods.append(method_name)

    # filter to methods with dirs on disk
    existing = []
    for method in canonical_methods:
        method_dir = hpo_results_dir / method
        if method_dir.exists() and method_dir.is_dir():
            existing.append(method)
            logger.info(f"discovered method: {method}")
        # silently skip if dir doesn't exist

    return sorted(existing)


def load_trials(method_dir: Path) -> list[dict]:
    """load all trial_*.json files from method_dir, sorted by trial_id (numeric).

    skip corrupt JSONs with warning. skip trials missing per_row_ldr_mean_ae key
    (old format) with warning. return empty list if dir does not exist.
    """
    if not method_dir.exists():
        return []

    trials = []
    trial_files = sorted(method_dir.glob("trial_*.json"))

    for fpath in trial_files:
        try:
            with open(fpath) as f:
                trial = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"skipping corrupt JSON {fpath}: {e}")
            continue

        if "per_row_ldr_mean_ae" not in trial:
            logger.warning(f"skipping trial without per_row_ldr_mean_ae: {fpath}")
            continue

        trials.append(trial)

    # sort by numeric trial_id
    trials.sort(key=lambda t: int(t["trial_id"]))
    return trials


def bucket_rows(per_row_ldr_mean_ae: dict, num_instances_per_kl: int) -> dict:
    """group per-row MAEs by kl_idx = int(row_str) // num_instances_per_kl.

    per_row_ldr_mean_ae is dict with string keys (row indices) and float values (MAEs).
    return dict {kl_idx: [mae_values]} sorted by kl_idx.
    """
    buckets = {}

    for row_str, mae_value in per_row_ldr_mean_ae.items():
        row_idx = int(row_str)
        kl_idx = row_idx // num_instances_per_kl
        if kl_idx not in buckets:
            buckets[kl_idx] = []
        buckets[kl_idx].append(mae_value)

    return buckets


def trial_bucket_scores(trial: dict, num_instances_per_kl: int) -> dict:
    """compute median MAE per bucket for a single trial.

    per-trial-per-bucket: median of that bucket's row metrics.
    return dict {kl_idx: median_mae}.
    """
    if "per_row_ldr_mean_ae" not in trial:
        return {}

    per_row = trial["per_row_ldr_mean_ae"]
    buckets = bucket_rows(per_row, num_instances_per_kl)

    # compute median per bucket
    scores = {}
    for kl_idx, mae_values in buckets.items():
        scores[kl_idx] = median(mae_values)

    return scores


def pick_winners(
    trials: list[dict],
    num_instances_per_kl: int,
    top_k: int,
) -> dict:
    """rank trials per kl_bucket by median MAE; retain top_k.

    per (method, kl_idx): sort trials ascending by bucket-median-MAE.
    return dict {kl_idx: [{trial_id, mae, hyperparams}, ...]}.
    """
    # group trials by (kl_idx, median_mae)
    kl_candidates = {}  # kl_idx -> [(median_mae, trial_id, hyperparams)]

    for trial in trials:
        bucket_scores = trial_bucket_scores(trial, num_instances_per_kl)
        if not bucket_scores:
            continue

        trial_id = int(trial["trial_id"])
        hyperparams = trial.get("hyperparams", {})

        for kl_idx, median_mae in bucket_scores.items():
            if kl_idx not in kl_candidates:
                kl_candidates[kl_idx] = []
            kl_candidates[kl_idx].append((median_mae, trial_id, hyperparams))

    # rank per kl_idx and retain top_k
    winners = {}
    for kl_idx, candidates in kl_candidates.items():
        # sort by (median_mae, trial_id)
        candidates.sort(key=lambda x: (x[0], x[1]))

        # retain top_k
        top_candidates = candidates[:top_k]
        winners[kl_idx] = [
            {
                "trial_id": trial_id,
                "mae": median_mae,
                "hyperparams": hyperparams,
            }
            for median_mae, trial_id, hyperparams in top_candidates
        ]

    return winners


def write_winners(winners: dict, output_path: Path) -> None:
    """atomic write: temp file + os.replace.

    output structure:
    <method>:
      <kl_idx>:
        - {trial_id, mae, hyperparams}  # rank 0
        - {trial_id, mae, hyperparams}  # rank 1
        ...
    """
    tmp_path = output_path.with_suffix(".tmp")

    # write to temp file
    with open(tmp_path, "w") as f:
        yaml.dump(winners, f, default_flow_style=False, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    # atomic rename
    os.replace(tmp_path, output_path)


def main():
    """parse CLI (no args); load config; iterate methods; collect winners; write YAML.

    CLI: python -m experiments.model_selection.pick_winners (no arguments).
    """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="aggregate HPO trial results")
    parser.parse_args()

    config = load_config(CONFIG_PATH)
    hpo_results_dir = Path(config["hpo_results_dir"])
    winners_path = Path(config["winners_path"])

    methods = resolve_methods(hpo_results_dir)
    logger.info(f"discovered {len(methods)} methods")

    winners = {}
    num_method_kl_pairs = 0

    for method in methods:
        method_dir = hpo_results_dir / method
        trials = load_trials(method_dir)

        kl_bucket_winners = pick_winners(
            trials,
            config["num_instances_per_kl"],
            config["top_k"],
        )

        if kl_bucket_winners:
            winners[method] = kl_bucket_winners
            num_method_kl_pairs += len(kl_bucket_winners)

    write_winners(winners, winners_path)
    logger.info(f"aggregated {len(winners)} methods, {num_method_kl_pairs} (method, kl_bucket) pairs")
    logger.info(f"wrote winners to {winners_path}")


if __name__ == "__main__":
    main()
