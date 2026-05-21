"""define-by-run optuna suggest_hp for FMDRE_S2 (CFG flow-matching DRE).

same conditionality structure as fmdre.py with one extra always-active knob
(`p_uncond`, the cfg dropout probability). search space mirrors FMDRE_S2's
entry in method_specs.py.

inertness edges (from static + scratch/fmdre_inertness_probe.py):
  - reweight inert when precond == True (EDM lambda bypasses reweight)
  - apply_iw inert when time_dist == "uniform"

`p_uncond` is always active across the searched range [0.1, 0.9]; both bounds
trigger the bernoulli-dropout path in make_fm_loss.

not searched -- pinned:
  - n_hidden_layers: per-experiment via StudyConfig.fixed_hp
  - div_method/n_hutch_samples: pinned "hutch_rademacher" / 4 inside
    build_FMDRE_S2 (mirrors the VFM/VFMOrthros pin)
  - sentinel_cond: FMDRE_S2 internal CFG sentinel (not an HPO knob)
  - activation/layernorm: CondVelScoreMLP has hardcoded GELU, no layernorm
"""

from typing import Any

import optuna


N_EPOCHS = 4000


METADATA = {
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_FMDRE_S2",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from the FMDRE_S2 search space.

    emits n_epochs as the fixed constant N_EPOCHS, plus 2 switch (precond,
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
    hp["n_epochs"] = N_EPOCHS
    hp["lr"] = trial.suggest_float("lr", 3e-5, 1e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    # switch params (suggest before any branch that reads them)
    precond = trial.suggest_categorical("precond", [False, True])
    hp["precond"] = precond
    time_dist = trial.suggest_categorical("time_dist", ["uniform", "beta_2_2", "beta_5_5"])
    hp["time_dist"] = time_dist

    # conditional params
    if not precond:
        hp["reweight"] = trial.suggest_categorical("reweight", [False, True])
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    # unconditional always-active params. eps upper extended to 2e-1 (FMDRE-family,
    # per OOR); score_weight lower to 1e-3 (OOR -27% at sw=0.01). p_uncond kept.
    hp["eps"] = trial.suggest_float("eps", 1e-4, 2e-1, log=True)
    hp["integration_steps"] = trial.suggest_int("integration_steps", 100, 2600)
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    hp["score_weight"] = trial.suggest_float("score_weight", 1e-3, 10.0, log=True)
    hp["p_uncond"] = trial.suggest_float("p_uncond", 0.1, 0.9)
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])

    # pinned per holdout boundary analysis (cosine_min_factor=0 won 5/6; OOR within-noise).
    hp["cosine_min_factor"] = 0.0

    return hp
