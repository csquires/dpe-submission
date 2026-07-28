"""suggest hyperparameters for MHT_NWJ via optuna.

maps hyperparameter search space to optuna trial calls. flat parameter
space; all parameters active in every training context, so no conditional
branching. fixes n_steps at N_STEPS = 6400 per HPO decision. single
latent_dim replaces hidden_dim/head_dim/num_shared_layers; learning rate
key renamed to "lr" to match StepMultiHeadCritic constructor.
"""

from typing import Any
import optuna


METADATA = {
    "uses_pruning": True,
    "requires_pstar": True,
    "builder": "build_MHTDRE",
}

N_STEPS = 6400


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """suggest hyperparameters for MHT_NWJ.

    translate hyperparameter search space to optuna calls:
    - lr: log-uniform [1e-4, 3e-2]
    - latent_dim: categorical [64, 128, 256]
    - num_waypoints: categorical [5, 10, 15]
    - batch_size: categorical [64, 128, 256, 512]
    - weight_decay: categorical [0.0, 1e-5, 1e-4, 1e-3]

    n_steps fixed at N_STEPS per shared HPO decision: uniform resource
    axis for Hyperband. builder (build_MHTDRE) reads flat_hp["n_steps"]
    mandatorily.

    returns flat dict passed to builder; no branching—all parameters active.
    """
    hp = {}

    # fixed training budget (Hyperband resource axis)
    hp["n_steps"] = N_STEPS

    # log-uniform continuous
    hp["lr"] = trial.suggest_float("lr", 1e-4, 3e-2, log=True)

    # categorical discrete
    hp["latent_dim"] = trial.suggest_categorical("latent_dim", [64, 128, 256])
    hp["num_waypoints"] = trial.suggest_categorical("num_waypoints", [5, 10, 15])
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3])

    return hp
