"""define-by-run optuna suggest_hp for TriangularTSM (time-score matching on a
piecewise-quadratic bell path).

translates the tuple-format search space from method_specs.py to trial.suggest_*
calls. mirrors the structure of `tsm.py` for the optimizer/regulariser block
(TriangularTSM shares the score-matching core and the same TSM-family knobs)
and adopts the vertex range from `triangular_ctsm.py`/`triangular_vfm.py`
(0.2, 0.8). pins n_steps at N_STEPS (uniform multi-fidelity resource axis).

unique to TriangularTSM:
  - vertex in (0, 1): bell peak location; mirrored to (0.2, 0.8) per the
    TriangularCTSM/VFM convention to avoid degenerate endpoint kinks.
  - peak_max in (0, 1]: bell peak height. lower bound 0.05 inherited from the
    legacy method_specs ablation (D=14 gaussian, 2 seeds): peak_max=0.05
    matches plain TSM's MAE while 0.5/1.0 was materially worse. the
    constructor enforces peak_max <= 1.0, so 1.0 is the open upper bound.

absent vs TSM: integration_steps -- TriangularTSM.predict_ldr uses a hardcoded
100-point trapezoid grid, so the key would be inert.

inertness edges (carried over from TSM):
  - apply_iw inert when time_dist == "uniform" (UniformSampler returns iw=1).
    no path schedule, no precond, no inner_eps in this method, so this is the
    only conditional branch.

not searched -- pinned:
  - n_hidden_layers: per-experiment via StudyConfig.fixed_hp.
  - cosine_min_factor: 0.0 (mirrors TSM's pin).
"""

from typing import Any

import optuna
from src.methods.reg.common._time_samplers import TIME_DISTS


N_STEPS = 6400


METADATA = {
    "uses_pruning": True,
    "requires_pstar": True,
    "builder": "build_TriangularTSM",
}


def suggest_hp(trial: optuna.Trial) -> dict[str, Any]:
    """sample hyperparameters from the TriangularTSM search space.

    emits n_steps as the fixed constant N_STEPS, plus 1 switch (time_dist),
    1 conditional (apply_iw -- suggested only when time_dist != "uniform"),
    and 13 unconditional knobs (incl. the bell-path scalars vertex/peak_max).

    args:
        trial: optuna trial object

    returns:
        flat dict of hyperparameters; conditionally-omitted apply_iw is absent
        when time_dist == "uniform" and the builder supplies its default via .get().
    """
    hp = {}

    # fixed constant + mandatory builder keys (mirrors tsm.py)
    hp["n_steps"] = N_STEPS
    hp["lr"] = trial.suggest_float("lr", 3e-5, 1e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    # switch param (suggest before its dependent branch)
    time_dist = trial.suggest_categorical("time_dist", list(TIME_DISTS))
    hp["time_dist"] = time_dist

    # conditional: importance weighting is no-op under uniform sampling
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    # unconditional shared-with-TSM knobs.
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    hp["activation"] = trial.suggest_categorical("activation", ["elu", "gelu", "silu"])
    hp["reweight"] = trial.suggest_categorical("reweight", [False, True])
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])

    # bell-path knobs unique to TriangularTSM.
    hp["vertex"] = trial.suggest_float("vertex", 0.2, 0.8)
    hp["peak_max"] = trial.suggest_float("peak_max", 0.05, 1.0)

    # pinned (mirrors tsm.py: cosine_min_factor=0 won 5/6 holdout boundary).
    hp["cosine_min_factor"] = 0.0

    return hp
