"""define-by-run optuna suggest_hp for VFMOrthros.

velocity flow matching with orthogonal shared backbone; conditional suggestion
of inert params. see notes/vfm_vfmorthros_conditionality.md for conditionality
rationale.
"""

from typing import Any

import optuna


N_EPOCHS = 2000


METADATA = {
    "cores_per_trial": 4,
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_VFMOrthros",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from the VFMOrthros search space.

    emits n_epochs as the fixed constant N_EPOCHS, plus 26 tuned params:
    5 switch (sched, inner_eps, div_method, precond, time_dist), 6 conditional
    (k, gamma_min, div_noise, n_hutch_samples, reweight, apply_iw -- each
    suggested only when its switch condition holds), and 15 unconditional
    (incl. the VFMOrthros-specific n_shared_layers). the depth knob
    n_hidden_layers is not searched: it is pinned per-experiment via
    StudyConfig.fixed_hp. the test-path params
    (test_sched, test_sigma, test_inner_eps, test_gamma_min, test_k) are
    derived equal to their train counterparts; test_eps is the only
    independent test-path knob.

    args:
        trial: optuna trial object

    returns:
        flat dict of hyperparameters; conditionally-omitted params are absent
        and the builder supplies their defaults via .get().
    """
    hp = {}

    # fixed constant + mandatory builder keys
    hp["n_epochs"] = N_EPOCHS
    hp["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
    hp["integration_steps"] = trial.suggest_int("integration_steps", 300, 2600)

    # switch params (suggest before any branch)
    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    inner_eps = trial.suggest_categorical("inner_eps", [0.0, 0.05, 0.1])
    hp["inner_eps"] = inner_eps
    div_method = trial.suggest_categorical("div_method", ["hutchinson", "exact"])
    hp["div_method"] = div_method
    precond = trial.suggest_categorical("precond", [False, True])
    hp["precond"] = precond

    # conditional params (6 inertness edges)
    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40])
    if inner_eps == 0.0:
        hp["gamma_min"] = trial.suggest_float("gamma_min", 1e-2, 2e-1, log=True)
    if div_method == "hutchinson":
        hp["div_noise"] = trial.suggest_categorical("div_noise", ["rademacher", "gaussian"])
        hp["n_hutch_samples"] = trial.suggest_categorical("n_hutch_samples", [1, 4, 16])
    if not precond:
        hp["reweight"] = trial.suggest_categorical("reweight", [False, True])

    # unconditional always-active params (15 total)
    hp["n_shared_layers"] = trial.suggest_categorical("n_shared_layers", [1, 2])
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["activation"] = trial.suggest_categorical("activation", ["gelu", "elu", "silu"])
    hp["sigma"] = trial.suggest_float("sigma", 0.3, 3.0, log=True)
    hp["test_eps"] = trial.suggest_float("test_eps", 1e-3, 1e-1, log=True)
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    hp["layernorm"] = trial.suggest_categorical("layernorm", ["off", "pre", "post"])
    hp["antithetic"] = trial.suggest_categorical("antithetic", [False, True])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3])
    hp["cosine_min_factor"] = trial.suggest_categorical("cosine_min_factor", [0.0, 0.01, 0.1])
    hp["time_dist"] = trial.suggest_categorical("time_dist", ["uniform", "beta_2_2", "beta_5_5"])

    # derive test params from train params
    hp["test_sched"] = hp["sched"]
    hp["test_sigma"] = hp["sigma"]
    hp["test_inner_eps"] = hp["inner_eps"]
    if "gamma_min" in hp:
        hp["test_gamma_min"] = hp["gamma_min"]
    if "k" in hp:
        hp["test_k"] = hp["k"]

    # conditional apply_iw only when time_dist != uniform
    time_dist = hp["time_dist"]
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    return hp
