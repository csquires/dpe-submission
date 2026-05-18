"""define-by-run optuna suggest_hp for VFM.

translates tuple-format search space from method_specs.py to trial.suggest_*
calls. implements conditional suggestion of inert params per
notes/vfm_vfmorthros_conditionality.md. fixes n_epochs at 2000 (HPO decision:
uniform multi-fidelity resource axis).
"""

from typing import Any

import optuna


N_EPOCHS = 2000


METADATA = {
    "cores_per_trial": 4,
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_VFM",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from the VFM search space.

    emits n_epochs as the fixed constant N_EPOCHS, plus 21 tuned params:
    4 switch (sched, inner_eps, precond, time_dist), 4 conditional
    (k, gamma_min, reweight, apply_iw -- each suggested only when its switch
    condition holds), and 13 unconditional.

    not searched -- pinned: n_hidden_layers (per-experiment via
    StudyConfig.fixed_hp), div_method/div_noise/n_hutch_samples (provisionally
    hutchinson/rademacher/4 samples), activation (the VFM class default). the
    test-path params (test_sched, test_sigma, test_inner_eps, test_gamma_min,
    test_k) are derived equal to their train counterparts; test_eps is the
    only independent test-path knob.

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

    # switch params (suggest before any branch that reads them)
    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    inner_eps = trial.suggest_categorical("inner_eps", [0.0, 0.05, 0.1])
    hp["inner_eps"] = inner_eps
    precond = trial.suggest_categorical("precond", [False, True])
    hp["precond"] = precond

    # provisionally pinned, not searched (a dedicated study is deferred):
    # divergence estimator -> hutchinson/rademacher/4; activation -> VFM default.
    hp["div_method"] = "hutchinson"
    hp["div_noise"] = "rademacher"
    hp["n_hutch_samples"] = 4
    hp["activation"] = "silu"

    # conditional params
    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40])
    if inner_eps == 0.0:
        hp["gamma_min"] = trial.suggest_float("gamma_min", 1e-2, 2e-1, log=True)
    if not precond:
        hp["reweight"] = trial.suggest_categorical("reweight", [False, True])

    # unconditional always-active params (17)
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
    hp["integration_steps"] = trial.suggest_int("integration_steps", 300, 2600)
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["sigma"] = trial.suggest_float("sigma", 0.3, 3.0, log=True)
    hp["test_eps"] = trial.suggest_float("test_eps", 1e-3, 1e-1, log=True)
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    hp["layernorm"] = trial.suggest_categorical("layernorm", ["off", "pre", "post"])
    hp["antithetic"] = trial.suggest_categorical("antithetic", [False, True])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3])
    hp["cosine_min_factor"] = trial.suggest_categorical("cosine_min_factor", [0.0, 0.01, 0.1])
    hp["time_dist"] = trial.suggest_categorical("time_dist", ["uniform", "beta_2_2", "beta_5_5"])

    # conditional: importance weighting (only meaningful under non-uniform time sampling)
    time_dist = hp["time_dist"]
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    # derive test path from train path (mirrors train except test_eps, which is
    # the only independent test-path knob). placed last so every train param it
    # mirrors (sched, sigma, inner_eps, gamma_min, k) is already in hp.
    hp["test_sched"] = hp["sched"]
    hp["test_sigma"] = hp["sigma"]
    hp["test_inner_eps"] = hp["inner_eps"]
    if "gamma_min" in hp:
        hp["test_gamma_min"] = hp["gamma_min"]
    if "k" in hp:
        hp["test_k"] = hp["k"]

    return hp
