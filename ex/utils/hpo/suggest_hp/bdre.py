"""define-by-run optuna suggest_hp for BDRE.

translates tuple-format search space from method_specs.py to trial.suggest_*
calls. excludes num_epochs; Hyperband owns the budget axis.
"""

from typing import Any
import optuna


METADATA = {
    "cores_per_trial": 2,
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_BDRE",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """translate BDRE base_search_space to trial.suggest_* calls.

    takes an optuna Trial and returns a dict of hyperparameters for BDRE.
    - excludes num_epochs (Hyperband computes budget internally from
      max_resource_steps)
    - returns deterministic dict given trial; no side effects beyond optuna
      state

    args:
        trial: optuna trial object

    returns:
        dict with keys {latent_dim, learning_rate}
    """
    hyperparams = {}

    # latent_dim: choice from {64, 128, 256}
    hyperparams["latent_dim"] = trial.suggest_categorical(
        "latent_dim", [64, 128, 256]
    )

    # learning_rate: log_uniform in [1e-4, 1e-2]
    hyperparams["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 1e-2, log=True
    )

    # num_epochs is NOT included; builder (build_BDRE) computes it
    # internally from max_resource_steps and batch_size.

    return hyperparams
