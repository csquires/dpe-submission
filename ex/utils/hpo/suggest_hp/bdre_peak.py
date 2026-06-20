"""peak-extraction variant of BDRE. N_STEPS bumped to 24300 to match
peak campaign max_resource. batch_size widened to [64, 128, 256, 512, 1024].

consider_pruned defaults to True: BDRE is a monotone-learning binary classifier;
intermediate cross-entropy strongly predicts final-epoch metric, safe for pruning.

to keep base and peak in sync, see test_peak_parity.py — it asserts every
non-bumped suggest_* call matches the base module byte-for-byte.
"""

from typing import Any

import optuna


N_STEPS = 24300


METADATA = {
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_BDRE",
    "consider_pruned": True,
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters for BDRE.

    emits n_steps as the fixed constant N_STEPS, plus 3 tuned params:
    - learning_rate: log-uniform [1e-4, 3e-2]
    - latent_dim: categorical [64, 128, 256]
    - batch_size: categorical [64, 128, 256, 512, 1024]

    not searched -- pinned per-experiment via StudyConfig.fixed_hp:
    - n_hidden_layers (mirrors VFM's convention)

    args:
        trial: optuna trial object

    returns:
        flat dict; builder forwards every key to DefaultBinaryClassifier
        via make_binary_classifier.
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
    hp["weight_decay"] = trial.suggest_categorical(
        "weight_decay", [0.0, 1e-5, 1e-4, 1e-3]
    )

    return hp
