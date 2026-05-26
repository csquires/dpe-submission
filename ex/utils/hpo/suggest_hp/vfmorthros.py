"""define-by-run optuna suggest_hp for VFMOrthros.

velocity flow matching with orthogonal shared backbone; conditional suggestion
of inert params. see notes/vfm_vfmorthros_conditionality.md for conditionality
rationale.
"""

from typing import Any

import optuna
from src.methods.reg.common._time_samplers import TIME_DISTS


N_EPOCHS = 6400


METADATA = {
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_VFMOrthros",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from the VFMOrthros search space.

    emits n_epochs as the fixed constant N_EPOCHS, plus 22 tuned params:
    4 switch (sched, inner_eps, precond, time_dist), 4 conditional
    (k, gamma_min, reweight, apply_iw -- each suggested only when its switch
    condition holds), and 14 unconditional (incl. the VFMOrthros-specific
    n_shared_layers).

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
    hp["lr"] = trial.suggest_float("lr", 3e-5, 1e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
    hp["integration_steps"] = trial.suggest_int("integration_steps", 100, 2600)

    # switch params (suggest before any branch). precond pinned True per
    # holdout boundary analysis -- masks the reweight branch entirely.
    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    inner_eps = trial.suggest_categorical("inner_eps", [0.0, 0.05, 0.1])
    hp["inner_eps"] = inner_eps
    hp["precond"] = True

    # divergence estimator pinned to exact; div_noise / n_hutch_samples inert
    # under method=="exact" but kept set for downstream validation. activation
    # -> VFM default.
    hp["div_method"] = "exact"
    hp["div_noise"] = "rademacher"
    hp["n_hutch_samples"] = 4
    hp["activation"] = "silu"

    # conditional params (reweight branch removed: precond=True masks it).
    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40, 80])
    if inner_eps == 0.0:
        hp["gamma_min"] = trial.suggest_float("gamma_min", 1e-2, 2e-1, log=True)

    # unconditional always-active params
    hp["n_shared_layers"] = trial.suggest_categorical("n_shared_layers", [1, 2, 3])
    hp["ema_decay"] = 0.999
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["sigma"] = trial.suggest_float("sigma", 0.1, 5.0, log=True)
    hp["test_eps"] = trial.suggest_float("test_eps", 1e-3, 3e-1, log=True)
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])
    hp["time_dist"] = trial.suggest_categorical("time_dist", list(TIME_DISTS))

    # pinned per holdout boundary analysis (won 5/6 winners; OOR within-noise):
    # layernorm=off, antithetic=True, cosine_min_factor=0.0.
    hp["layernorm"] = "off"
    hp["antithetic"] = True
    hp["cosine_min_factor"] = 0.0

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
