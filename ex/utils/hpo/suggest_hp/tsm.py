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


N_EPOCHS = 2000


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
    hp["lr"] = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])

    # switch param (suggest before its dependent branch)
    time_dist = trial.suggest_categorical("time_dist", ["uniform", "beta_2_2", "beta_5_5"])
    hp["time_dist"] = time_dist

    # conditional: importance weighting is no-op under uniform sampling
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    # unconditional always-active params
    hp["eps"] = trial.suggest_float("eps", 1e-6, 1e-4, log=True)
    hp["integration_steps"] = trial.suggest_int("integration_steps", 300, 2600)
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    hp["activation"] = trial.suggest_categorical("activation", ["elu", "gelu", "silu"])
    hp["reweight"] = trial.suggest_categorical("reweight", [False, True])
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3])
    hp["cosine_min_factor"] = trial.suggest_categorical("cosine_min_factor", [0.0, 0.01, 0.1])

    return hp
