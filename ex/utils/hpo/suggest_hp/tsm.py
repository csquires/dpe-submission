"""define-by-run optuna suggest_hp for TSM (time score matching).

translates tuple-format search space from method_specs.py to trial.suggest_*
calls. implements conditional suggestion of inert params per
notes/hpo_search_space_finalization.md. fixes n_epochs at 2000 (HPO decision:
uniform multi-fidelity resource axis).

inertness edges (from static + scratch/tsm_inertness_probe.py):
  - apply_iw inert when time_dist == "uniform" (UniformSampler returns iw=1;
    only behavioral edge -- TSM has no path schedule, no precond, no inner_eps)

not searched -- pinned:
  - n_hidden_layers: per-experiment via StudyConfig.fixed_hp
"""

from typing import Any

import optuna
from src.methods.reg.common._time_samplers import TIME_DISTS


N_EPOCHS = 6400


METADATA = {
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_TSM",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from the TSM search space.

    emits n_epochs as the fixed constant N_EPOCHS, plus 1 switch (time_dist),
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
    hp["n_epochs"] = N_EPOCHS
    hp["lr"] = trial.suggest_float("lr", 3e-5, 1e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    # switch param (suggest before its dependent branch)
    time_dist = trial.suggest_categorical("time_dist", list(TIME_DISTS))
    hp["time_dist"] = time_dist

    # conditional: importance weighting is no-op under uniform sampling
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    # unconditional always-active params. eps NOT widened (not FMDRE family);
    # activation kept searchable.
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
    hp["integration_steps"] = trial.suggest_int("integration_steps", 100, 2600)
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    hp["activation"] = trial.suggest_categorical("activation", ["elu", "gelu", "silu"])
    hp["reweight"] = trial.suggest_categorical("reweight", [False, True])
    hp["ema_decay"] = 0.999
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])

    # pinned per holdout boundary analysis (cosine_min_factor=0 won 5/6; OOR within-noise).
    hp["cosine_min_factor"] = 0.0

    return hp
