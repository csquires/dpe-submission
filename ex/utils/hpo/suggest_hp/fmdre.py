"""define-by-run optuna suggest_hp for FMDRE (S1, two-source flow-matching DRE).

translates tuple-format search space from method_specs.py to trial.suggest_*
calls. implements conditional suggestion of inert params per
notes/hpo_search_space_finalization.md. fixes n_steps at 2000 (HPO decision:
uniform multi-fidelity resource axis).

inertness edges (from static + scratch/fmdre_inertness_probe.py):
  - reweight inert when precond == True (EDM lambda outer_weight bypasses
    reweight inside make_fm_loss; warning emitted by FMDRE.fit when both set)
  - apply_iw inert when time_dist == "uniform" (UniformSampler returns iw=1)

not searched -- pinned:
  - n_hidden_layers: per-experiment via StudyConfig.fixed_hp
  - div_method/n_hutch_samples: pinned "hutch_rademacher" / 4 inside build_FMDRE
    (mirrors the VFM/VFMOrthros pin)
  - activation/layernorm: CondVelScoreMLP has hardcoded GELU, no layernorm
"""

from typing import Any

import optuna
from src.methods.reg.common._time_samplers import TIME_DISTS


N_STEPS = 6400


METADATA = {
    "uses_pruning": True,
    "requires_pstar": False,
    "builder": "build_FMDRE",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from the FMDRE (S1) search space.

    emits n_steps as the fixed constant N_STEPS, plus 2 switch (precond,
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
    hp["n_steps"] = N_STEPS
    hp["lr"] = trial.suggest_float("lr", 3e-5, 3e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    # switch params (suggest before any branch that reads them)
    precond = trial.suggest_categorical("precond", [False, True])
    hp["precond"] = precond
    time_dist = trial.suggest_categorical("time_dist", list(TIME_DISTS))
    hp["time_dist"] = time_dist

    # conditional params
    if not precond:
        hp["reweight"] = trial.suggest_categorical("reweight", [False, True])
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    # unconditional always-active params. eps upper extended to 2e-1 (FMDRE-family,
    # per OOR TriangularFMDRE eps=0.15 optimum); score_weight lower to 1e-3 (OOR -27%).
    hp["eps"] = trial.suggest_float("eps", 1e-4, 2e-1, log=True)
    hp["integration_steps"] = trial.suggest_categorical("integration_steps", [400, 1200, 2600])
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    # n_shared_layers in [1, n_hidden_layers]; 5 == fully shared (pre-split FMDRE).
    hp["n_shared_layers"] = trial.suggest_categorical("n_shared_layers", [1, 2, 3, 4, 5])
    hp["score_weight"] = trial.suggest_float("score_weight", 1e-3, 3.0, log=True)
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])

    # pinned per holdout boundary analysis (cosine_min_factor=0 won 5/6; OOR within-noise).
    hp["cosine_min_factor"] = 0.0

    return hp
