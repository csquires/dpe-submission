"""suggest hyperparameters for MultiHeadTDRE via optuna.

maps base_search_space from method_specs.py to optuna trial calls.
flat parameter space; behavioral inertness probe confirmed all parameters
active in every training context, so no conditional branching.
"""

from typing import Any
import optuna


METADATA = {
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_MHTDRE",
}

N_STEPS = 6400


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """suggest hyperparameters for MultiHeadTDRE.

    translate method_specs.py base_search_space to optuna calls:
    - learning_rate: log-uniform [1e-4, 1e-2]
    - hidden_dim: categorical [64, 128, 256]
    - head_dim: categorical [10, 20, 40]
    - num_shared_layers: categorical [1, 2, 3]
    - num_waypoints: categorical [5, 10, 15]
    - batch_size: categorical [None, 128, 256, 512]
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
    hp["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # categorical discrete
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    hp["head_dim"] = trial.suggest_categorical("head_dim", [10, 20, 40])
    hp["num_shared_layers"] = trial.suggest_categorical("num_shared_layers", [1, 2, 3])
    hp["num_waypoints"] = trial.suggest_categorical("num_waypoints", [5, 10, 15])
    hp["batch_size"] = trial.suggest_categorical("batch_size", [None, 128, 256, 512])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3])

    return hp
