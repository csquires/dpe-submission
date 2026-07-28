"""define-by-run optuna suggest_hp for BDRE_DV.

translates hyperparameter search space to trial.suggest_* calls. flat
parameter space; all parameters active in every training context, so no
conditional branching. fixes n_steps at N_STEPS = 6400 per HPO decision.
learning rate key renamed to "lr" to match StepBinaryCritic constructor.
"""

from typing import Any

import optuna


N_STEPS = 6400


METADATA = {
    "uses_pruning": True,
    "requires_pstar": True,
    "builder": "build_BDRE",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters for BDRE_DV.

    emits n_steps as the fixed constant N_STEPS, plus 4 tuned params:
    - lr: log-uniform [1e-4, 3e-2]
    - latent_dim: categorical [64, 128, 256]
    - batch_size: categorical [64, 128, 256, 512]
    - weight_decay: categorical [0.0, 1e-5, 1e-4, 1e-3]

    not searched -- pinned per-experiment via StudyConfig.fixed_hp:
    - n_hidden_layers

    args:
        trial: optuna trial object

    returns:
        flat dict; builder forwards every key to StepBinaryCritic.
    """
    hp = {}

    hp["n_steps"] = N_STEPS

    hp["lr"] = trial.suggest_float(
        "lr", 1e-4, 3e-2, log=True
    )
    hp["latent_dim"] = trial.suggest_categorical(
        "latent_dim", [64, 128, 256]
    )
    hp["batch_size"] = trial.suggest_categorical(
        "batch_size", [64, 128, 256, 512]
    )
    hp["weight_decay"] = trial.suggest_categorical(
        "weight_decay", [0.0, 1e-5, 1e-4, 1e-3]
    )

    return hp
