"""define-by-run optuna suggest_hp for VFM.

translates tuple-format search space from method_specs.py to trial.suggest_*
calls. implements conditional suggestion of inert params per
notes/vfm_vfmorthros_conditionality.md. fixes n_epochs at 1500 (HPO decision:
uniform Hyperband resource axis).
"""

from typing import Any

import optuna


N_EPOCHS = 1500


METADATA = {
    "cores_per_trial": 4,
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_VFM",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from VFM search space.

    implements 29 total params: 1 fixed constant (n_epochs) + 28 tuned
    (6 switch, 6 conditional, 16 unconditional). conditionally-omitted params
    are not emitted; builder supplies defaults via .get().

    args:
        trial: optuna trial object

    returns:
        flat dict with hyperparameters. n_epochs is the constant N_EPOCHS;
        switch params are always present; conditional params only if their
        condition holds; 16 unconditional params always present.
    """
    hp = {}

    # fixed constant + mandatory builder keys
    hp["n_epochs"] = N_EPOCHS
    hp["lr"] = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])

    # switch params (suggest before any branch that reads them)
    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    test_sched = trial.suggest_categorical("test_sched", ["stiff", "bridge"])
    hp["test_sched"] = test_sched
    inner_eps = trial.suggest_categorical("inner_eps", [0.0, 0.05, 0.1])
    hp["inner_eps"] = inner_eps
    test_inner_eps = trial.suggest_categorical("test_inner_eps", [0.0, 0.05, 0.1])
    hp["test_inner_eps"] = test_inner_eps
    div_method = trial.suggest_categorical("div_method", ["hutchinson", "exact"])
    hp["div_method"] = div_method
    precond = trial.suggest_categorical("precond", [False, True])
    hp["precond"] = precond

    # conditional params (6 inertness edges)
    if sched == "stiff" or test_sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40])
    if inner_eps == 0.0:
        hp["gamma_min"] = trial.suggest_float("gamma_min", 1e-2, 2e-1, log=True)
    if test_inner_eps == 0.0:
        hp["test_gamma_min"] = trial.suggest_float("test_gamma_min", 1e-2, 2e-1, log=True)
    if div_method == "hutchinson":
        hp["div_noise"] = trial.suggest_categorical("div_noise", ["rademacher", "gaussian"])
        hp["n_hutch_samples"] = trial.suggest_categorical("n_hutch_samples", [1, 4, 16])
    if not precond:
        hp["reweight"] = trial.suggest_categorical("reweight", [False, True])

    # unconditional always-active params (16)
    hp["eps"] = trial.suggest_float("eps", 1e-3, 5e-3, log=True)
    hp["integration_steps"] = trial.suggest_int("integration_steps", 300, 2600)
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["activation"] = trial.suggest_categorical("activation", ["gelu", "elu", "silu"])
    hp["sigma"] = trial.suggest_float("sigma", 0.3, 3.0, log=True)
    hp["test_eps"] = trial.suggest_float("test_eps", 1e-3, 1e-1, log=True)
    hp["test_sigma"] = trial.suggest_float("test_sigma", 0.3, 3.0, log=True)
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    hp["n_hidden_layers"] = trial.suggest_categorical("n_hidden_layers", [2, 3, 4])
    hp["layernorm"] = trial.suggest_categorical("layernorm", ["off", "pre", "post"])
    hp["antithetic"] = trial.suggest_categorical("antithetic", [False, True])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3])
    hp["cosine_min_factor"] = trial.suggest_categorical("cosine_min_factor", [0.0, 0.01, 0.1])

    return hp
