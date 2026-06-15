"""suggest hyperparameters for MultiHeadTriangularTDRE via optuna.

maps base_search_space from method_specs.py to optuna trial calls.
triangular methods require intermediate distribution p* from adapter.
"""

from typing import Any
import optuna


N_STEPS = 6400


METADATA = {
    "uses_pruning": True,
    "requires_pstar": True,
    "builder": "build_MHTTDRE",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """suggest hyperparameters for MultiHeadTriangularTDRE.

    translate method_specs.py base_search_space to optuna calls:
    - learning_rate: log-uniform [1e-4, 3e-2]
    - hidden_dim: categorical [64, 128, 256]
    - head_dim: categorical [10, 20, 40]
    - num_shared_layers: categorical [1, 2, 3]
    - num_waypoints: categorical [3, 5, 9, 15]
    - vertex: uniform [0.1, 0.9]
    - batch_size: categorical [64, 128, 256, 512]
    - weight_decay: categorical [0.0, 1e-5, 1e-4, 1e-3]
    - midpoint_oversample: categorical [0, 3, 5, 7]
    - gamma_power: log-uniform [0.3, 5.0]
    - n_steps: constant N_STEPS

    returns flat dict passed to builder; builder validates shape constraints.
    """
    hp = {}

    # log-uniform continuous
    hp["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True)

    # categorical discrete
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    hp["head_dim"] = trial.suggest_categorical("head_dim", [10, 20, 40])
    hp["num_shared_layers"] = trial.suggest_categorical("num_shared_layers", [1, 2, 3])
    hp["num_waypoints"] = trial.suggest_categorical("num_waypoints", [3, 5, 9, 15])
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3])
    hp["midpoint_oversample"] = trial.suggest_categorical("midpoint_oversample", [0, 3, 5, 7])

    # uniform continuous (position on interpolation path)
    hp["vertex"] = trial.suggest_float("vertex", 0.1, 0.9)

    # log-uniform continuous (preconditioning power)
    hp["gamma_power"] = trial.suggest_float("gamma_power", 0.3, 5.0, log=True)

    # constant (training iterations)
    hp["n_steps"] = N_STEPS

    return hp
