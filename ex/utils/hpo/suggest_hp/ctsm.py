"""define-by-run optuna suggest_hp for CTSM (conditional time-score matching).

translates tuple-format search space from method_specs.py to trial.suggest_*
calls. implements conditional suggestion of inert params per
notes/hpo_search_space_finalization.md. fixes n_epochs at 2000 (HPO decision:
uniform multi-fidelity resource axis).

inertness edges (from static + scratch/ctsm_inertness_probe.py):
  - k inert when sched == "bridge" (bridge_noise doesn't read k)
  - gamma_min inert when inner_eps > 0 (non-zero band clamps gamma above the
    gamma_min floor and masks it -- mirrors the VFM/VFMOrthros edge)
  - apply_iw inert when time_dist == "uniform" (UniformSampler returns iw=1)

test-path knobs (test_sched, test_sigma, test_inner_eps, test_gamma_min,
test_k) are derived from the train-side counterparts; only test_eps remains
independent (mirrors VFM/VFMOrthros convention).

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
    "builder": "build_CTSM",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from the CTSM search space.

    emits n_epochs as the fixed constant N_EPOCHS, plus 3 switch (sched,
    inner_eps, time_dist), 3 conditional (k, gamma_min, apply_iw -- each
    suggested only when its switch condition holds), and 15 unconditional.

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

    # switch params (suggest before any branch that reads them)
    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    inner_eps = trial.suggest_categorical("inner_eps", [0.0, 0.05, 0.1])
    hp["inner_eps"] = inner_eps
    time_dist = trial.suggest_categorical("time_dist", list(TIME_DISTS))
    hp["time_dist"] = time_dist

    # conditional params
    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40, 80])
    if inner_eps == 0.0:
        hp["gamma_min"] = trial.suggest_float("gamma_min", 1e-2, 2e-1, log=True)
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    # unconditional always-active params. eps NOT widened (path method; the
    # (1e-4, 2e-1) eps widening is FMDRE-family-only). activation kept searchable.
    hp["sigma"] = trial.suggest_float("sigma", 0.1, 5.0, log=True)
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
    hp["integration_steps"] = trial.suggest_int("integration_steps", 100, 2600)
    hp["test_eps"] = trial.suggest_float("test_eps", 1e-3, 3e-1, log=True)
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    hp["activation"] = trial.suggest_categorical("activation", ["elu", "gelu", "silu"])
    hp["reweight"] = trial.suggest_categorical("reweight", [False, True])
    hp["ema_decay"] = 0.999
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])

    # pinned per holdout boundary analysis (cosine_min_factor=0 won 5/6; OOR within-noise).
    hp["cosine_min_factor"] = 0.0

    # derive test-path schedule from train-side counterparts (mirrors VFM).
    hp["test_sched"] = hp["sched"]
    hp["test_sigma"] = hp["sigma"]
    hp["test_inner_eps"] = hp["inner_eps"]
    if "gamma_min" in hp:
        hp["test_gamma_min"] = hp["gamma_min"]
    if "k" in hp:
        hp["test_k"] = hp["k"]

    return hp
