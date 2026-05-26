"""define-by-run optuna suggest_hp for TriangularMDRE.

translates tuple-format search space from method_specs.py to trial.suggest_*
calls. flat parameter space; the behavioral inertness probe
(scratch/bdre_mdre_inertness_probe.py + bdre_mdre_inertness_results.txt)
confirmed all parameters active in every training context (including the
waypoint-builder knobs midpoint_oversample / gamma_power / vertex, which
flow through the training data via TriangularWaypointBuilder1D). no
conditional branching. fixes num_epochs at N_EPOCHS = 4000 (HPO decision:
uniform multi-fidelity resource axis, mirroring VFM / MultiHeadTDRE).
"""

from typing import Any

import optuna


N_EPOCHS = 4000


METADATA = {
    "uses_pruning": True,
    "requires_pstar": True,
    "builder": "build_TriangularMDRE",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters for TriangularMDRE.

    emits num_epochs as the fixed constant N_EPOCHS, plus 7 tuned params:
    - learning_rate: log-uniform [1e-4, 1e-2]
    - latent_dim: categorical [64, 128, 256]
    - batch_size: categorical [None, 128, 256]
    - num_waypoints: categorical [3, 5, 7, 9]
    - midpoint_oversample: categorical [0, 3, 5, 7]
    - gamma_power: uniform [1.0, 5.0]
    - vertex: uniform [0.2, 0.8]

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

    hp["num_epochs"] = N_EPOCHS

    hp["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 1e-2, log=True
    )
    hp["latent_dim"] = trial.suggest_categorical(
        "latent_dim", [64, 128, 256]
    )
    hp["batch_size"] = trial.suggest_categorical(
        "batch_size", [None, 128, 256, 512]
    )
    hp["num_waypoints"] = trial.suggest_categorical(
        "num_waypoints", [3, 5, 7, 9]
    )

    hp["midpoint_oversample"] = trial.suggest_categorical(
        "midpoint_oversample", [0, 3, 5, 7]
    )
    hp["gamma_power"] = trial.suggest_float("gamma_power", 1.0, 5.0)
    hp["vertex"] = trial.suggest_float("vertex", 0.2, 0.8)
    hp["weight_decay"] = trial.suggest_categorical(
        "weight_decay", [0.0, 1e-5, 1e-4, 1e-3]
    )

    return hp
