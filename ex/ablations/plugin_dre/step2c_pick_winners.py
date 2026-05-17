"""
Aggregate plugin_dre HPO trial results and select best hyperparams per KL.
"""

import json
import math
import os

import yaml

from ex.utils.hpo.method_specs import METHOD_SPECS


def load_trials(hpo_results_dir: str, method: str, kl_idx: int) -> list[dict]:
    trial_dir = os.path.join(hpo_results_dir, method, f"kl_{kl_idx}")
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
    valid = [t for t in trials if math.isfinite(t.get("mean_mae", float("nan")))]
    if not valid:
        return None
    return min(valid, key=lambda t: t["mean_mae"])


def main():
    config = yaml.safe_load(open("ex/ablations/plugin_dre/config.yaml"))
    hpo_results_dir = config["hpo_results_dir"]
    hpo_summary_dir = config["hpo_summary_dir"]
    kl_divergences = config["kl_divergences"]
    os.makedirs(hpo_summary_dir, exist_ok=True)

    winners = {}

    for method in METHOD_SPECS:
        method_kl_results = {}

        for kl_idx, kl_value in enumerate(kl_divergences):
            kl_key = f"kl_{kl_idx}"
            trials = load_trials(hpo_results_dir, method, kl_idx)

            if not trials:
                print(f"  {method} kl_{kl_idx}: no trials found, skipping")
                continue

            best = pick_best(trials)
            if best is None:
                print(f"  {method} kl_{kl_idx}: all trials non-finite, skipping")
                continue

            print(
                f"  {method} kl_{kl_idx} (KL={kl_value}): "
                f"winner trial_id={best['trial_id']}, MAE={best['mean_mae']:.4f}"
            )

            method_kl_results[kl_key] = {
                "kl_value": kl_value,
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

            winners.setdefault(method, {})[kl_key] = best["hyperparams"]

        method_summary = {
            "method": method,
            "nsamples_train": config["nsamples_train"],
            "hpo_num_trials": config["hpo_num_trials"],
            "hpo_num_eval_instances": config["hpo_num_eval_instances"],
            "kl_results": method_kl_results,
        }
        method_path = os.path.join(hpo_summary_dir, f"{method}.json")
        with open(method_path, "w") as f:
            json.dump(method_summary, f, indent=2)
        print(f"  wrote {method_path}")

    winners_path = os.path.join(hpo_summary_dir, "winners.json")
    with open(winners_path, "w") as f:
        json.dump(winners, f, indent=2)

    print(f"\nWinners written to {winners_path}")
    print(f"Methods with winners: {list(winners.keys())}")


if __name__ == "__main__":
    main()
