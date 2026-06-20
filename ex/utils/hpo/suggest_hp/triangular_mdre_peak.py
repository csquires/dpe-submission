"""peak-extraction variant of TriangularMDRE. N_STEPS bumped to 24300 to match
peak campaign max_resource. batch_size widened to [64, 128, 256, 512, 1024].

consider_pruned defaults to False: TriangularMDRE uses a triangular annealing
path; rung-pruned trials may fail due to schedule mismatch rather than bad
hyperparams, introducing unmeasured bias. conservative default: no pruning.

to keep base and peak in sync, see test_peak_parity.py — it asserts every
non-bumped suggest_* call matches the base module byte-for-byte.
"""

from typing import Any

import optuna


N_STEPS = 24300


METADATA = {
    "uses_pruning": True,
    "requires_pstar": True,
    "builder": "build_TriangularMDRE",
    "consider_pruned": False,
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters for TriangularMDRE.

    emits n_steps as the fixed constant N_STEPS, plus 7 tuned params:
    - learning_rate: log-uniform [1e-4, 3e-2]
    - latent_dim: categorical [64, 128, 256]
    - batch_size: categorical [64, 128, 256, 512, 1024]
    - num_waypoints: categorical [3, 5, 9, 15]
    - midpoint_oversample: categorical [0, 3, 5, 7]
    - gamma_power: uniform [0.3, 5.0]
    - vertex: uniform [0.1, 0.9]

    not searched -- pinned per-experiment via StudyConfig.fixed_hp:
    - n_hidden_layers (mirrors VFM's convention)

    args:
        trial: optuna trial object

    returns:
        flat dict; builder pops num_waypoints / midpoint_oversample /
        gamma_power / vertex (waypoint-builder knobs) and forwards the rest
        to DefaultMulticlassClassifier via make_multiclass_classifier.
    """
    hp = {}

    hp["n_steps"] = N_STEPS

    hp["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 3e-2, log=True
    )
    hp["latent_dim"] = trial.suggest_categorical(
        "latent_dim", [64, 128, 256]
    )
    hp["batch_size"] = trial.suggest_categorical(
        "batch_size", [64, 128, 256, 512, 1024]
    )
    hp["num_waypoints"] = trial.suggest_categorical(
        "num_waypoints", [3, 5, 9, 15]
    )

    hp["midpoint_oversample"] = trial.suggest_categorical(
        "midpoint_oversample", [0, 3, 5, 7]
    )
    hp["gamma_power"] = trial.suggest_float("gamma_power", 0.3, 5.0)
    hp["vertex"] = trial.suggest_float("vertex", 0.1, 0.9)
    hp["weight_decay"] = trial.suggest_categorical(
        "weight_decay", [0.0, 1e-5, 1e-4, 1e-3]
    )

    return hp
