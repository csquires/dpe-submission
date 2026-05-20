"""suggest_hp for TabularPluginDRE: oracle tabular method with empty search space."""

from typing import Any

import optuna


METADATA = {
    "uses_pruning": False,
    "requires_pstar": False,
    "builder": "build_TabularPluginDRE",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """suggest hyperparameters for TabularPluginDRE.

    TabularPluginDRE is a tabular oracle method with empty search space.
    no hyperparameters are tunable; method is deterministic given data.
    returns empty dict. trial argument is accepted for API uniformity but unused.

    args:
        trial: optuna trial object (unused; API contract only)

    returns:
        empty dict (no tunable hyperparameters)
    """
    return {}
