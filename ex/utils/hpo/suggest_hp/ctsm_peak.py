"""peak-extraction variant of CTSM. N_STEPS bumped to 24300 to match
peak campaign max_resource. batch_size widened to [64, 128, 256, 512, 1024].

consider_pruned defaults to True: CTSM is a monotone-learning conditional
score-matching method; no curriculum or annealing schedule, so intermediate
rung metrics reliably predict final-epoch quality.

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
    "builder": "build_CTSM",
    "consider_pruned": True,
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from the CTSM search space.

    emits n_steps as the fixed constant N_STEPS, plus 2 switch (sched,
    time_dist), 2 conditional (k, apply_iw -- each suggested only when its
    switch condition holds), and 16 unconditional. inner_eps is pinned to 0.

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
    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    hp["inner_eps"] = 0.0
    time_dist = trial.suggest_categorical("time_dist", list(TIME_DISTS))
    hp["time_dist"] = time_dist

    # conditional params
    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40, 80])
    # eig winner sits at 0.095 (well inside this range), and the active-band
    # check at scratch/gamma_min_audit shows no useful config below 1e-2 for
    # bridge-scheduled CTSM. lower bound stays at 1e-2.
    hp["gamma_min"] = trial.suggest_float("gamma_min", 1e-2, 2e-1, log=True)
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    # unconditional always-active params. eps NOT widened (path method; the
    # (1e-4, 2e-1) eps widening is FMDRE-family-only). activation kept searchable.
    hp["sigma"] = trial.suggest_float("sigma", 0.02, 10.0, log=True)
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
    hp["integration_steps"] = trial.suggest_categorical("integration_steps", [400, 1200, 2600])
    # inference margin sealed to the trained tau domain: test_eps = eps * factor,
    # factor log-uniform [0.8, 10]. factor >= 1 keeps [test_eps, 1-test_eps] a
    # strict subset of the trained [eps, 1-eps] (never integrate where the score
    # was unsupervised); the sub-1 tail [0.8, 1) gives a small ood slack in case
    # endpoint extrapolation is genuinely free (cf. eig).
    hp["test_eps"] = max(hp["eps"], hp["inner_eps"]) * trial.suggest_float(
        "test_eps_factor", 0.8, 10.0, log=True)
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    hp["activation"] = trial.suggest_categorical("activation", ["elu", "gelu", "silu"])
    hp["reweight"] = trial.suggest_categorical("reweight", [False, True])
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])

    # pinned per holdout boundary analysis (cosine_min_factor=0 won 5/6; OOR within-noise).
    hp["cosine_min_factor"] = 0.0

    # derive test-path schedule from train-side counterparts (mirrors VFM).
    hp["test_sched"] = hp["sched"]
    hp["test_sigma"] = hp["sigma"]
    hp["test_inner_eps"] = hp["inner_eps"]
    hp["test_gamma_min"] = hp["gamma_min"]
    if "k" in hp:
        hp["test_k"] = hp["k"]

    return hp
