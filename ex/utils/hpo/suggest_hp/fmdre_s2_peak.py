"""peak-extraction variant of FMDRE_S2. N_STEPS bumped to 24300 to match
peak campaign max_resource. batch_size widened to [64, 128, 256, 512, 1024].

consider_pruned defaults to False: FMDRE_S2 uses staged gamma schedule;
annealing bias in intermediate metrics means rung pruning introduces risk.
conservative default: no pruning.

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
    "builder": "build_FMDRE_S2",
    "consider_pruned": False,
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from the FMDRE_S2 search space.

    emits n_steps as the fixed constant N_STEPS, plus 2 switch (precond,
    time_dist), 2 conditional (reweight, apply_iw), and 13 unconditional
    (p_uncond included).

    args:
        trial: optuna trial object

    returns:
        flat dict of hyperparameters; conditionally-omitted params are absent
        and the builder supplies their defaults via .get().
    """
    hp = {}

    # fixed constant + mandatory builder keys
    hp["n_steps"] = N_STEPS
    hp["lr"] = trial.suggest_float("lr", 3e-5, 3e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])

    # switch params (suggest before any branch that reads them)
    precond = trial.suggest_categorical("precond", [False, True])
    hp["precond"] = precond
    time_dist = trial.suggest_categorical("time_dist", list(TIME_DISTS))
    hp["time_dist"] = time_dist

    # conditional params
    if not precond:
        hp["reweight"] = trial.suggest_categorical("reweight", [False, True])
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    # unconditional always-active params. eps upper extended to 2e-1 (FMDRE-family,
    # per OOR); score_weight lower to 1e-3 (OOR -27% at sw=0.01). p_uncond kept.
    hp["eps"] = trial.suggest_float("eps", 1e-4, 2e-1, log=True)
    hp["integration_steps"] = trial.suggest_categorical("integration_steps", [400, 1200, 2600])
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    # n_shared_layers in [1, n_hidden_layers]; 5 == fully shared (pre-split FMDRE_S2).
    hp["n_shared_layers"] = trial.suggest_categorical("n_shared_layers", [1, 2, 3, 4, 5])
    hp["score_weight"] = trial.suggest_float("score_weight", 1e-3, 3.0, log=True)
    hp["p_uncond"] = trial.suggest_float("p_uncond", 0.1, 0.9)
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])

    # pinned per holdout boundary analysis (cosine_min_factor=0 won 5/6; OOR within-noise).
    hp["cosine_min_factor"] = 0.0

    return hp
