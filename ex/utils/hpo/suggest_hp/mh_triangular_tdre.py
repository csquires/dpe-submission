"""suggest hyperparameters for MultiHeadTriangularTDRE via optuna.

maps base_search_space from method_specs.py to optuna trial calls.
triangular methods require intermediate distribution p* from adapter.
"""

from typing import Any
import optuna


METADATA = {
    "cores_per_trial": 4,
    "uses_pruning": True,
    "requires_pstar": True,
    "builder": "build_MHTTDRE",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """suggest hyperparameters for MultiHeadTriangularTDRE.

    translate method_specs.py base_search_space to optuna calls:
    - learning_rate: log-uniform [1e-4, 1e-2]
    - hidden_dim: categorical [16, 32, 64, 128]
    - head_dim: categorical [10, 20, 40]
    - num_shared_layers: categorical [1, 2, 3]
    - num_waypoints: categorical [5, 10, 15]
    - vertex: uniform [0.2, 0.8]

    num_epochs removed per shared_context.md §13 (computed internally).

    returns flat dict passed to builder; builder validates shape constraints.
    """
    hp = {}

    # log-uniform continuous
    hp["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    # categorical discrete
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
    hp["head_dim"] = trial.suggest_categorical("head_dim", [10, 20, 40])
    hp["num_shared_layers"] = trial.suggest_categorical("num_shared_layers", [1, 2, 3])
    hp["num_waypoints"] = trial.suggest_categorical("num_waypoints", [5, 10, 15])

    # uniform continuous (position on interpolation path)
    hp["vertex"] = trial.suggest_float("vertex", 0.2, 0.8)

    return hp
