"""
Aggregate HPO trial results and select best hyperparams per (method, pstar_idx).

For each (method, pstar_idx), loads all trial JSONs from hpo_results/, selects
the trial with the lowest mean_mae (finite values only), and writes:

  hpo_summary/winners.json          flat lookup used by step2_run_algorithms.py
  hpo_summary/{method}.json         full record with all trials, for building tables

Run after all step2b array jobs have completed.
"""

import json
import math
import os

import yaml

from experiments.pstar_sample_complexity.hpo_search_spaces import SEARCH_SPACES


def load_trials(hpo_results_dir: str, method: str, pstar_idx: int) -> list[dict]:
    """Load all completed trial JSONs for (method, pstar_idx), sorted by trial_id."""
    trial_dir = os.path.join(hpo_results_dir, method, f"pstar_{pstar_idx}")
    if not os.path.isdir(trial_dir):
        return []

    trials = []
    for fname in sorted(os.listdir(trial_dir)):
        if not fname.startswith("trial_") or not fname.endswith(".json"):
            continue
        path = os.path.join(trial_dir, fname)
        try:
            with open(path) as f:
                trials.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  warning: skipping corrupt file {path}: {e}")

    trials.sort(key=lambda t: t["trial_id"])
    return trials


def pick_best(trials: list[dict]) -> dict | None:
    """Return the trial with the lowest finite mean_mae, None if no valid trials."""
    valid = [t for t in trials if math.isfinite(t.get("mean_mae", float("nan")))]
    if not valid:
        return None
    return min(valid, key=lambda t: t["mean_mae"])


def main():
    config = yaml.safe_load(open("experiments/pstar_sample_complexity/config.yaml"))
    hpo_results_dir = config["hpo_results_dir"]
    hpo_summary_dir = config["hpo_summary_dir"]
    nsamples_pstar_values = config["nsamples_pstar_values"]
    os.makedirs(hpo_summary_dir, exist_ok=True)

    # winners[method][pstar_key] = hyperparams dict  (pstar_key = "pstar_0", "pstar_1", ...)
    winners = {}

    for method in SEARCH_SPACES:
        method_pstar_results = {}

        for pstar_idx, nsamples_pstar in enumerate(nsamples_pstar_values):
            pstar_key = f"pstar_{pstar_idx}"
            trials = load_trials(hpo_results_dir, method, pstar_idx)

            if not trials:
                print(f"  {method} pstar_{pstar_idx}: no trials found, skipping")
                continue

            best = pick_best(trials)
            if best is None:
                print(f"  {method} pstar_{pstar_idx}: all trials non-finite, skipping")
                continue

            print(f"  {method} pstar_{pstar_idx} (n={nsamples_pstar}): "
                  f"winner trial_id={best['trial_id']}, MAE={best['mean_mae']:.4f}")

            method_pstar_results[pstar_key] = {
                "nsamples_pstar": nsamples_pstar,
                "winner": {
                    "trial_id": best["trial_id"],
                    "hyperparams": best["hyperparams"],
                    "mae": best["mean_mae"],
                },
                "all_trials": [
                    {
                        "trial_id": t["trial_id"],
                        "hyperparams": t["hyperparams"],
                        "mae": t["mean_mae"],
                        "mae_per_instance": t.get("mae_per_instance", {}),
                    }
                    for t in trials
                ],
            }

            if method not in winners:
                winners[method] = {}
            winners[method][pstar_key] = best["hyperparams"]

        # write per-method summary JSON
        method_summary = {
            "method": method,
            "hpo_num_trials": config["hpo_num_trials"],
            "hpo_num_eval_instances": config["hpo_num_eval_instances"],
            "pstar_results": method_pstar_results,
        }
        method_path = os.path.join(hpo_summary_dir, f"{method}.json")
        with open(method_path, "w") as f:
            json.dump(method_summary, f, indent=2)
        print(f"  wrote {method_path}")

    # write flat winners.json — this is what step2_run_algorithms.py reads
    winners_path = os.path.join(hpo_summary_dir, "winners.json")
    with open(winners_path, "w") as f:
        json.dump(winners, f, indent=2)

    print(f"\nWinners written to {winners_path}")
    print(f"Methods with winners: {list(winners.keys())}")


if __name__ == "__main__":
    main()
