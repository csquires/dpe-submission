"""peak-extraction variant of TSM. N_STEPS bumped to 24300 to match
peak campaign max_resource. batch_size widened to [64, 128, 256, 512, 1024].

consider_pruned defaults to True: TSM is a monotone-learning score-matching
method with no curriculum or annealing; intermediate rung metrics reliably
predict final-epoch outcomes, safe for pruning.

to keep base and peak in sync, see test_peak_parity.py — it asserts every
non-bumped suggest_* call matches the base module byte-for-byte.
"""

from typing import Any

import optuna
from src.methods.reg.common._time_samplers import TIME_DISTS


N_STEPS = 24300


METADATA = {
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_TSM",
    "consider_pruned": True,
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from the TSM search space.

    emits n_steps as the fixed constant N_STEPS, plus 1 switch (time_dist),
    1 conditional (apply_iw -- suggested only when time_dist != "uniform"),
    and 11 unconditional.

    args:
        trial: optuna trial object

    returns:
        flat dict of hyperparameters; conditionally-omitted apply_iw is absent
        when time_dist == "uniform" and the builder supplies its default via .get().
    """
    hp = {}

    # fixed constant + mandatory builder keys
    hp["n_steps"] = N_STEPS
    hp["lr"] = trial.suggest_float("lr", 3e-5, 3e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])

    # switch param (suggest before its dependent branch)
    time_dist = trial.suggest_categorical("time_dist", list(TIME_DISTS))
    hp["time_dist"] = time_dist

    # conditional: importance weighting is no-op under uniform sampling
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    # unconditional always-active params. eps NOT widened (not FMDRE family);
    # activation kept searchable.
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
    hp["integration_steps"] = trial.suggest_categorical("integration_steps", [400, 1200, 2600])
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    hp["activation"] = trial.suggest_categorical("activation", ["elu", "gelu", "silu"])
    hp["reweight"] = trial.suggest_categorical("reweight", [False, True])
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])

    # pinned per holdout boundary analysis (cosine_min_factor=0 won 5/6; OOR within-noise).
    hp["cosine_min_factor"] = 0.0

    return hp
