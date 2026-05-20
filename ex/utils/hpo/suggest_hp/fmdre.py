"""define-by-run optuna suggest_hp for FMDRE (S1, two-source flow-matching DRE).

translates tuple-format search space from method_specs.py to trial.suggest_*
calls. implements conditional suggestion of inert params per
notes/hpo_search_space_finalization.md. fixes n_epochs at 2000 (HPO decision:
uniform multi-fidelity resource axis).

inertness edges (from static + scratch/fmdre_inertness_probe.py):
  - reweight inert when precond == True (EDM lambda outer_weight bypasses
    reweight inside make_fm_loss; warning emitted by FMDRE.fit when both set)
  - apply_iw inert when time_dist == "uniform" (UniformSampler returns iw=1)

not searched -- pinned:
  - n_hidden_layers: per-experiment via StudyConfig.fixed_hp
  - div_method: deliberately pinned "exact" inside build_FMDRE
  - activation/layernorm: CondVelScoreMLP has hardcoded GELU, no layernorm
"""

from typing import Any

import optuna


N_EPOCHS = 2000


METADATA = {
    "cores_per_trial": 4,
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_FMDRE",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from the FMDRE (S1) search space.

    emits n_epochs as the fixed constant N_EPOCHS, plus 2 switch (precond,
    time_dist), 2 conditional (reweight, apply_iw -- each suggested only when
    its switch condition holds), and 12 unconditional.

    args:
        trial: optuna trial object

    returns:
        flat dict of hyperparameters; conditionally-omitted params are absent
        and the builder supplies their defaults via .get().
    """
    hp = {}

    # fixed constant + mandatory builder keys
    hp["n_epochs"] = N_EPOCHS
    hp["lr"] = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512])

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

    # unconditional always-active params
    hp["eps"] = trial.suggest_float("eps", 1e-3, 5e-2, log=True)
    hp["integration_steps"] = trial.suggest_int("integration_steps", 1000, 3000)
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    hp["score_weight"] = trial.suggest_float("score_weight", 0.1, 10.0, log=True)
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3])
    hp["cosine_min_factor"] = trial.suggest_categorical("cosine_min_factor", [0.0, 0.01, 0.1])

    return hp
