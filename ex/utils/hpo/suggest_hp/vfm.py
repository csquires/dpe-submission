"""define-by-run optuna suggest_hp for VFM.

translates tuple-format search space from method_specs.py to trial.suggest_*
calls. implements conditional suggestion of inert params per
notes/vfm_vfmorthros_conditionality.md. fixes n_steps at 2000 (HPO decision:
uniform multi-fidelity resource axis).
"""

from typing import Any

import optuna
from src.methods.reg.common._time_samplers import TIME_DISTS


N_STEPS = 6400


METADATA = {
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_VFM",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from the VFM search space.

    emits n_steps as the fixed constant N_STEPS, plus tuned params:
    3 switch (sched, precond, time_dist), 3 conditional (k, reweight, apply_iw
    -- each suggested only when its switch condition holds), and the rest
    unconditional. inner_eps is pinned to 0 (matches triangular VFM convention;
    the clamp-mode alternative shadowed gamma_min in compose order).

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
    hp["n_steps"] = N_STEPS
    hp["lr"] = trial.suggest_float("lr", 3e-5, 3e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    # switch params (suggest before any branch that reads them)
    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    hp["inner_eps"] = 0.0
    precond = trial.suggest_categorical("precond", [False, True])
    hp["precond"] = precond

    # divergence estimator pinned to exact (closed-form trace); div_noise /
    # n_hutch_samples are inert under method=="exact" but kept set so downstream
    # build_div_fn / estimator validation stays happy. activation -> VFM default.
    hp["div_method"] = "exact"
    hp["div_noise"] = "rademacher"
    hp["n_hutch_samples"] = 4
    hp["activation"] = "silu"

    # conditional params
    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40, 80])
    # gamma_min pinned 0 (eig winner sat at 0.036, well below the natural
    # floor g(eps)=0.31; HPO never used the clamp).
    hp["gamma_min"] = 0.0
    if not precond:
        hp["reweight"] = trial.suggest_categorical("reweight", [False, True])

    # unconditional always-active params. eps NOT widened (VFM winners mid-range);
    # the (1e-4, 2e-1) eps widening is FMDRE-family-only.
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
    hp["integration_steps"] = trial.suggest_categorical("integration_steps", [400, 1200, 2600])
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["sigma"] = trial.suggest_float("sigma", 0.02, 10.0, log=True)
    # inference margin sealed to the trained tau domain: test_eps = eps * factor,
    # factor log-uniform [0.8, 10] (see ctsm.py for rationale).
    hp["test_eps"] = max(hp["eps"], hp["inner_eps"]) * trial.suggest_float(
        "test_eps_factor", 0.8, 10.0, log=True)
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])
    hp["time_dist"] = trial.suggest_categorical("time_dist", list(TIME_DISTS))

    # pinned per holdout boundary analysis (won 5/6 winners; OOR within-noise):
    # layernorm=off, antithetic=True, cosine_min_factor=0.0.
    hp["layernorm"] = "off"
    hp["antithetic"] = True
    hp["cosine_min_factor"] = 0.0

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
    hp["test_gamma_min"] = hp["gamma_min"]
    if "k" in hp:
        hp["test_k"] = hp["k"]

    return hp
