"""define-by-run optuna suggest_hp for MDRE.

translates tuple-format search space from method_specs.py to trial.suggest_*
calls. flat parameter space; the behavioral inertness probe
(scratch/bdre_mdre_inertness_probe.py + bdre_mdre_inertness_results.txt)
confirmed all parameters active in every training context, so no conditional
branching. fixes n_steps at N_STEPS = 6400 (HPO decision: uniform
multi-fidelity resource axis, mirroring VFM / MultiHeadTDRE).
"""

from typing import Any

import optuna


N_STEPS = 6400


METADATA = {
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_MDRE",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters for MDRE.

    emits n_steps as the fixed constant N_STEPS, plus 4 tuned params:
    - learning_rate: log-uniform [1e-4, 1e-2]
    - latent_dim: categorical [64, 128, 256]
    - batch_size: categorical [64, 128, 256, 512]
    - num_waypoints: categorical [3, 5, 7, 9] (= classifier num_classes)

    not searched -- pinned per-experiment via StudyConfig.fixed_hp:
    - n_hidden_layers (mirrors VFM's convention)

    args:
        trial: optuna trial object

    returns:
        flat dict; builder pops num_waypoints (= num_classes) and forwards
        the rest to DefaultMulticlassClassifier via
        make_multiclass_classifier.
    """
    hp = {}

    hp["n_steps"] = N_STEPS

    hp["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 1e-2, log=True
    )
    hp["latent_dim"] = trial.suggest_categorical(
        "latent_dim", [64, 128, 256]
    )
    hp["batch_size"] = trial.suggest_categorical(
        "batch_size", [64, 128, 256, 512]
    )
    hp["num_waypoints"] = trial.suggest_categorical(
        "num_waypoints", [3, 5, 7, 9]
    )
    hp["weight_decay"] = trial.suggest_categorical(
        "weight_decay", [0.0, 1e-5, 1e-4, 1e-3]
    )

    return hp
