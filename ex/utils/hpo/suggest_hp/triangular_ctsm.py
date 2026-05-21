"""define-by-run optuna suggest_hp for TriangularCTSM V1/V2/V3.

translates the tuple-format search spaces from method_specs.py to trial.suggest_*
calls. conditional suggestion of inert params per the inertness edges in
notes/hpo_search_space_finalization.md (static scan + seeded double-build probe
in scratch/triangular_ctsm_inertness_probe.py). pins n_epochs at N_EPOCHS
(uniform multi-fidelity resource axis).

three versions share most of the space; they differ in the path family:
  - V1 psb_1d (piecewise-Schroedinger-bridge), vertex searched.
  - V2 bary_1d (barycentric), vertex pinned 0.5 in the builder.
  - V3 rect_2d (2D stacked) + LowArcCurve2D; no time-sampling knobs (the
    builder hardcodes a product of uniforms); adds t2_max + path_height.

inertness edges (probe + static):
  - k inert when sched == "bridge" (bridge_noise ignores k).
  - gamma_min inert when inner_eps > 0 (non-zero band clamps gamma above the
    floor and masks it) -- V1/V2 only.
  - V1 only: time_dist AND apply_iw inert when inner_eps > 0, because the
    builder swaps to make_piecewise_sb_sampler, which reads neither.
  - apply_iw inert when time_dist == "uniform" (iw == 1).
  - V3 path_height is inference-only (curve used only in predict_ldr); searched
    unconditionally, probe blind. V3 gamma_min searched unconditionally (probe
    gates it on sched, a fragile edge we decline to encode).

test-path knobs (test_sched, test_sigma, test_inner_eps, test_gamma_min,
test_k) are derived from the train-side counterparts; only test_eps remains
independent (mirrors the VFM/CTSM convention).

not searched -- pinned: n_hidden_layers (per-experiment via StudyConfig.fixed_hp).
"""

from typing import Any

import optuna
from src.methods.reg.common._time_samplers import TIME_DISTS


N_EPOCHS = 4000


METADATA_V1 = {"uses_pruning": True, "requires_pstar": True, "builder": "build_TriangularCTSM_V1"}
METADATA_V2 = {"uses_pruning": True, "requires_pstar": True, "builder": "build_TriangularCTSM_V2"}
METADATA_V3 = {"uses_pruning": True, "requires_pstar": True, "builder": "build_TriangularCTSM_V3"}


def _common_optim(trial: optuna.Trial, hp: dict) -> None:
    """suggest the optimizer + regulariser knobs shared by all three versions."""
    hp["n_epochs"] = N_EPOCHS
    hp["lr"] = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])
    hp["sigma"] = trial.suggest_float("sigma", 0.3, 3.0, log=True)
    hp["integration_steps"] = trial.suggest_int("integration_steps", 300, 2600)
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["activation"] = trial.suggest_categorical("activation", ["elu", "gelu", "silu"])
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    hp["reweight"] = trial.suggest_categorical("reweight", [False, True])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3])
    hp["cosine_min_factor"] = trial.suggest_categorical("cosine_min_factor", [0.0, 0.01, 0.1])
    hp["test_eps"] = trial.suggest_float("test_eps", 1e-3, 1e-1, log=True)


def _derive_test(hp: dict) -> None:
    """derive the test-path schedule from the train-side knobs (test_eps stays
    independent). placed last so every mirrored train key is already present."""
    hp["test_sched"] = hp["sched"]
    hp["test_sigma"] = hp["sigma"]
    hp["test_inner_eps"] = hp["inner_eps"]
    if "gamma_min" in hp:
        hp["test_gamma_min"] = hp["gamma_min"]
    if "k" in hp:
        hp["test_k"] = hp["k"]


def _suggest_1d(trial: optuna.Trial, *, inner_eps_grid: list, psb: bool) -> dict[str, Any]:
    """shared body for the two 1d-time versions (V1 psb, V2 bary).

    psb=True selects the V1 piecewise-SB branch where inner_eps > 0 swaps the
    sampler and masks both time_dist and apply_iw.
    """
    hp: dict[str, Any] = {}
    _common_optim(trial, hp)
    hp["eps"] = trial.suggest_float("eps", 1e-3, 3e-3, log=True)

    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    inner_eps = trial.suggest_categorical("inner_eps", inner_eps_grid)
    hp["inner_eps"] = inner_eps

    if psb:
        hp["vertex"] = trial.suggest_float("vertex", 0.2, 0.8)

    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40])
    if inner_eps == 0.0:
        hp["gamma_min"] = trial.suggest_float("gamma_min", 1e-2, 2e-1, log=True)

    # time sampling. V1 (psb): builder ignores time_dist/apply_iw when inner_eps
    # > 0, so only expose them in the inner_eps == 0 branch. V2 (bary): always
    # active.
    expose_time = (not psb) or inner_eps == 0.0
    if expose_time:
        time_dist = trial.suggest_categorical("time_dist", list(TIME_DISTS))
        hp["time_dist"] = time_dist
        if time_dist != "uniform":
            hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    _derive_test(hp)
    return hp


def suggest_hp_v1(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularCTSM V1 (piecewise-SB path, searched vertex).

    switches: sched, inner_eps, time_dist (the last gated by inner_eps == 0).
    conditional: k (stiff), gamma_min (inner_eps==0), time_dist (inner_eps==0),
    apply_iw (inner_eps==0 and time_dist != uniform).
    """
    return _suggest_1d(trial, inner_eps_grid=[0.0, 0.02, 0.05], psb=True)


def suggest_hp_v2(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularCTSM V2 (barycentric path, vertex pinned 0.5 in the builder).

    switches: sched, inner_eps, time_dist. conditional: k (stiff), gamma_min
    (inner_eps==0), apply_iw (time_dist != uniform). time_dist always active.
    """
    return _suggest_1d(trial, inner_eps_grid=[0.0, 0.05, 0.1], psb=False)


def suggest_hp_v3(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularCTSM V3 (2D stacked path + LowArcCurve2D).

    no time-sampling knobs (the builder hardcodes a product of uniforms). adds
    t2_max + path_height (path_height is inference-only: it parameterises the
    curve used only in predict_ldr). conditional: k (stiff). gamma_min searched
    unconditionally (the probe gates it on sched, a fragile edge we decline to
    encode). test_gamma_min derived from gamma_min.
    """
    hp: dict[str, Any] = {}
    _common_optim(trial, hp)
    hp["eps"] = trial.suggest_float("eps", 1e-3, 3e-3, log=True)

    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    hp["inner_eps"] = trial.suggest_categorical("inner_eps", [0.0, 0.05, 0.1])
    hp["t2_max"] = trial.suggest_float("t2_max", 0.6, 0.9)
    hp["path_height"] = trial.suggest_float("path_height", 1.0, 2.0)
    hp["gamma_min"] = trial.suggest_float("gamma_min", 1e-2, 2e-1, log=True)

    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [12, 24, 48])

    _derive_test(hp)
    return hp
