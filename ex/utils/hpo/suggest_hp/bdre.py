"""define-by-run optuna suggest_hp for BDRE.

translates tuple-format search space from method_specs.py to trial.suggest_*
calls. flat parameter space; the behavioral inertness probe
(scratch/bdre_mdre_inertness_probe.py + bdre_mdre_inertness_results.txt)
confirmed all parameters active in every training context, so no conditional
branching. fixes num_epochs at N_EPOCHS = 2000 (HPO decision: uniform
multi-fidelity resource axis, mirroring VFM / MultiHeadTDRE).
"""

from typing import Any

import optuna


N_EPOCHS = 4000


METADATA = {
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_BDRE",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters for BDRE.

    emits num_epochs as the fixed constant N_EPOCHS, plus 3 tuned params:
    - learning_rate: log-uniform [1e-4, 1e-2]
    - latent_dim: categorical [64, 128, 256]
    - batch_size: categorical [None, 128, 256]

    not searched -- pinned per-experiment via StudyConfig.fixed_hp:
    - n_hidden_layers (mirrors VFM's convention)

    args:
        trial: optuna trial object

    returns:
        flat dict; builder forwards every key to DefaultBinaryClassifier
        via make_binary_classifier.
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
    hp["weight_decay"] = trial.suggest_categorical(
        "weight_decay", [0.0, 1e-5, 1e-4, 1e-3]
    )

    return hp
