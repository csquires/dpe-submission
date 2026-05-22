"""define-by-run optuna suggest_hp for TriangularVFM V1/V2/V3.

translates the tuple-format search spaces from method_specs.py to trial.suggest_*
calls. conditional suggestion of inert params per the inertness edges in
notes/hpo_search_space_finalization.md (static scan + seeded double-build probe
in scratch/triangular_vfm_inertness_probe.py). pins n_epochs at N_EPOCHS.

three versions differ in the path family (psb_1d / bary_1d / rect_2d), mirroring
TriangularCTSM. like stock VFM, the divergence estimator is pinned
(hutchinson/rademacher/4) and activation is pinned -- these are inference-side
knobs the train-side probe cannot rank, so a dedicated study is deferred.

inertness edges (probe + static):
  - k inert when sched == "bridge".
  - gamma_min inert when inner_eps > 0 (V1/V2). V3 searches gamma_min
    unconditionally (probe gates it on sched, a fragile edge we decline).
  - V1 always samples time per-leg via a width-proportional two-leg mixture
    sampler (any inner_eps; every TIME_DISTS value applied per leg), so time_dist is
    unconditional -- same treatment as V2, only the sampler differs
    (see notes/triangular_v1_time_dist_coupling.md).
  - apply_iw inert when time_dist == "uniform".
  - reweight gated on precond == False (mirrors stock VFM: the precond=True
    EDM-lambda path masks it). NB the probe found reweight active even under
    precond == True for the triangular paths, but per the user decision we
    mirror stock VFM's structure exactly. V3 has no precond, so reweight is
    always active there.
  - V3 (TriangularVFM2D) has no precond support and no time-sampling knobs;
    path_height is inference-only (curve used only in predict_ldr).

test-path knobs are derived from the train-side counterparts; only test_eps is
independent. not searched -- pinned: n_hidden_layers (via StudyConfig.fixed_hp),
div_method/div_noise/n_hutch_samples, activation.
"""

from typing import Any

import optuna
from src.methods.reg.common._time_samplers import TIME_DISTS


N_EPOCHS = 4000


METADATA_V1 = {"uses_pruning": True, "requires_pstar": True, "builder": "build_TriangularVFM_V1"}
METADATA_V2 = {"uses_pruning": True, "requires_pstar": True, "builder": "build_TriangularVFM_V2"}
METADATA_V3 = {"uses_pruning": True, "requires_pstar": True, "builder": "build_TriangularVFM_V3"}


def _common(trial: optuna.Trial, hp: dict) -> None:
    """suggest the knobs shared by all three versions, including the pinned
    divergence/activation knobs (mirrors stock VFM)."""
    hp["n_epochs"] = N_EPOCHS
    hp["lr"] = trial.suggest_float("lr", 3e-5, 1e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    hp["sigma"] = trial.suggest_float("sigma", 0.1, 5.0, log=True)
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
    hp["integration_steps"] = trial.suggest_int("integration_steps", 100, 2600)
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    hp["layernorm"] = trial.suggest_categorical("layernorm", ["off", "pre", "post"])
    hp["antithetic"] = trial.suggest_categorical("antithetic", [False, True])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])
    hp["cosine_min_factor"] = trial.suggest_categorical("cosine_min_factor", [0.0, 0.01, 0.1])
    hp["test_eps"] = trial.suggest_float("test_eps", 1e-3, 3e-1, log=True)

    # provisionally pinned, not searched (a dedicated study is deferred):
    # divergence estimator -> hutchinson/rademacher/4; activation -> VFM default.
    hp["div_method"] = "hutchinson"
    hp["div_noise"] = "rademacher"
    hp["n_hutch_samples"] = 4
    hp["activation"] = "silu"


def _derive_test(hp: dict) -> None:
    """derive the test-path schedule from the train-side knobs (test_eps stays
    independent)."""
    hp["test_sched"] = hp["sched"]
    hp["test_sigma"] = hp["sigma"]
    hp["test_inner_eps"] = hp["inner_eps"]
    if "gamma_min" in hp:
        hp["test_gamma_min"] = hp["gamma_min"]
    if "k" in hp:
        hp["test_k"] = hp["k"]


def _suggest_1d(trial: optuna.Trial, *, inner_eps_grid: list, psb: bool) -> dict[str, Any]:
    """shared body for the two 1d-time versions (V1 psb, V2 bary)."""
    hp: dict[str, Any] = {}
    _common(trial, hp)

    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    inner_eps = trial.suggest_categorical("inner_eps", inner_eps_grid)
    hp["inner_eps"] = inner_eps
    hp["vertex"] = trial.suggest_float("vertex", 0.2, 0.8)
    precond = trial.suggest_categorical("precond", [False, True])
    hp["precond"] = precond

    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40, 80])
    if inner_eps == 0.0:
        hp["gamma_min"] = trial.suggest_float("gamma_min", 1e-2, 2e-1, log=True)
    # reweight gated on not-precond (mirrors stock VFM: the precond=True path
    # uses an EDM lambda outer-weight that masks reweight).
    if not precond:
        hp["reweight"] = trial.suggest_categorical("reweight", [False, True])

    # time_dist/apply_iw always active for V1 and V2. V1 samples time per-leg
    # via the two-leg mixture sampler (any inner_eps; every TIME_DISTS value per
    # leg); V2 uses the global sampler. same suggester treatment.
    time_dist = trial.suggest_categorical("time_dist", list(TIME_DISTS))
    hp["time_dist"] = time_dist
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    _derive_test(hp)
    return hp


def suggest_hp_v1(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularVFM V1 (piecewise-SB path, searched vertex, searched precond).

    note: the stiff-schedule `k` IS live for V1 (it parameterises stiff_noise);
    the old "k vestigial" spec comment confused it with the barycentric
    constructor scalar. conditional: k (stiff), gamma_min (inner_eps==0),
    apply_iw (time_dist != uniform). time_dist always active.
    """
    return _suggest_1d(trial, inner_eps_grid=[0.0, 0.05, 0.1], psb=True)


def suggest_hp_v2(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularVFM V2 (barycentric path, searched vertex, searched precond).

    conditional: k (stiff), gamma_min (inner_eps==0), apply_iw (time_dist !=
    uniform). time_dist always active.
    """
    return _suggest_1d(trial, inner_eps_grid=[0.0, 0.05, 0.1], psb=False)


def suggest_hp_v3(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularVFM V3 (2D stacked path + LowArcCurve2D).

    no precond (TriangularVFM2D has no precond support) and no time-sampling
    knobs (the builder hardcodes a product of uniforms). adds t2_max +
    path_height (inference-only). conditional: k (stiff). gamma_min searched
    unconditionally. test_gamma_min derived from gamma_min.
    """
    hp: dict[str, Any] = {}
    _common(trial, hp)

    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    hp["inner_eps"] = trial.suggest_categorical("inner_eps", [0.0, 0.05, 0.1])
    hp["t2_max"] = trial.suggest_float("t2_max", 0.6, 0.9)
    hp["path_height"] = trial.suggest_float("path_height", 1.0, 2.0)
    hp["gamma_min"] = trial.suggest_float("gamma_min", 1e-2, 2e-1, log=True)
    # no precond support, so reweight is always active (the gate `not precond`
    # is vacuously true here).
    hp["reweight"] = trial.suggest_categorical("reweight", [False, True])

    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40, 80])

    _derive_test(hp)
    return hp
