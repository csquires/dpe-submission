"""suggest_hp for TriangularFMDRE: flow matching with triangular p0 -> p* -> p1 path."""

from typing import Any

import optuna


METADATA = {
    "uses_pruning": True,
    "requires_pstar": True,
    "builder": "build_TriangularFMDRE",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from triangular FMDRE search space.

    implements 7-dimensional log/linear/categorical search space:
    n_epochs (log), lr (log), batch_size (cat), eps (log),
    integration_steps (linear), hidden_dim (cat), score_weight (log).

    args:
        trial: optuna trial object

    returns:
        flat dict with all 7 hyperparameters as hashable values
    """
    return {
        "n_epochs": trial.suggest_int("n_epochs", 500, 1500, log=True),
        "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "eps": trial.suggest_float("eps", 1e-3, 5e-2, log=True),
        "integration_steps": trial.suggest_int("integration_steps", 1000, 3000, log=False),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "score_weight": trial.suggest_float("score_weight", 0.1, 10.0, log=True),
    }
