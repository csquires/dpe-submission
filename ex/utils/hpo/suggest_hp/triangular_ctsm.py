"""define-by-run optuna suggest_hp for TriangularCTSM V1/V2/V3.

translates the tuple-format search spaces from method_specs.py to trial.suggest_*
calls. conditional suggestion of inert params per the inertness edges in
notes/hpo_search_space_finalization.md (static scan + seeded double-build probe
in scratch/triangular_ctsm_inertness_probe.py). pins n_steps at N_STEPS
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
  - apply_iw inert when time_dist == "uniform" (iw == 1).
  - V1 always samples time per-leg via a width-proportional two-leg mixture
    sampler (any inner_eps; every TIME_DISTS value applied per leg), so time_dist
    is unconditional -- same treatment as V2, only the sampler differs
    (see notes/triangular_v1_time_dist_coupling.md).
  - V3 path_height is inference-only (curve used only in predict_ldr); searched
    unconditionally, probe blind. V3 gamma_min searched unconditionally (probe
    gates it on sched, a fragile edge we decline to encode).

test-path knobs (test_sched, test_sigma, test_inner_eps, test_gamma_min,
test_k) are derived from the train-side counterparts. test_eps is no longer
independent: we search a small test_eps_offset and derive test_eps = eps +
offset, which forces the inference linspace boundary strictly inside the
training support.

not searched -- pinned: n_hidden_layers (per-experiment via StudyConfig.fixed_hp).
"""

from typing import Any

import optuna
from src.methods.reg.common._time_samplers import TIME_DISTS


N_STEPS = 6400


METADATA_V1 = {"uses_pruning": True, "requires_pstar": True, "builder": "build_TriangularCTSM_V1"}
METADATA_V2 = {"uses_pruning": True, "requires_pstar": True, "builder": "build_TriangularCTSM_V2"}
METADATA_V3 = {"uses_pruning": True, "requires_pstar": True, "builder": "build_TriangularCTSM_V3"}


def _common_optim(trial: optuna.Trial, hp: dict) -> None:
    """suggest the optimizer + regulariser knobs shared by all three versions."""
    hp["n_steps"] = N_STEPS
    hp["lr"] = trial.suggest_float("lr", 3e-5, 1e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    hp["sigma"] = trial.suggest_float("sigma", 0.1, 5.0, log=True)
    hp["integration_steps"] = trial.suggest_int("integration_steps", 100, 2600)
    hp["ema_decay"] = 0.999
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["activation"] = trial.suggest_categorical("activation", ["elu", "gelu", "silu"])
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    hp["reweight"] = trial.suggest_categorical("reweight", [False, True])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])
    hp["cosine_min_factor"] = 0.0
    hp["test_eps_offset"] = trial.suggest_float(
        "test_eps_offset", 0, 1e-3, log=True,
    )


def _derive_test(hp: dict) -> None:
    """derive the test-path schedule from the train-side knobs.

    test_eps is derived as max(eps, inner_eps) + test_eps_offset; this forces
    the inference linspace strictly inside the training time support AND
    strictly outside the path's gamma-clamp band, killing both the boundary
    off-path failure mode and the clamp-band off-path failure mode (the latter
    only kicks in when inner_eps > eps, which is V1's regime). placed last so
    every mirrored train key is already present.
    """
    hp["test_eps"] = max(hp["eps"], hp["inner_eps"]) + hp["test_eps_offset"]
    hp["test_sched"] = hp["sched"]
    hp["test_sigma"] = hp["sigma"]
    hp["test_inner_eps"] = hp["inner_eps"]
    if "gamma_min" in hp:
        hp["test_gamma_min"] = hp["gamma_min"]
    if "k" in hp:
        hp["test_k"] = hp["k"]


def _suggest_1d(trial: optuna.Trial, *, inner_eps_grid: list, psb: bool) -> dict[str, Any]:
    """shared body for the two 1d-time versions (V1 psb, V2 bary).

    psb=True selects the V1 piecewise-SB branch (vertex searched); psb=False
    selects V2 barycentric (vertex pinned in builder). time_dist and apply_iw
    are always live for both. (V1 differs only in the builder: it samples time
    per-leg via the two-leg mixture sampler; V2 uses the global sampler.)
    """
    hp: dict[str, Any] = {}
    _common_optim(trial, hp)
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)

    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    inner_eps = trial.suggest_categorical("inner_eps", inner_eps_grid)
    hp["inner_eps"] = inner_eps

    if psb:
        hp["vertex"] = trial.suggest_float("vertex", 0.25, 0.75)

    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40, 80])
    if inner_eps == 0.0:
        hp["gamma_min"] = trial.suggest_float("gamma_min", 1e-2, 2e-1, log=True)

    # time sampling. time_dist/apply_iw are always live for V1 and V2 -- V1
    # samples time per-leg via the two-leg mixture sampler (any inner_eps; every
    # TIME_DISTS value per leg), V2 uses the global sampler. same suggester
    # treatment either way.
    time_dist = trial.suggest_categorical("time_dist", list(TIME_DISTS))
    hp["time_dist"] = time_dist
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    _derive_test(hp)
    return hp


def suggest_hp_v1(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularCTSM V1 (piecewise-SB path, searched vertex).

    switches: sched, inner_eps, time_dist. conditional: k (stiff), gamma_min
    (inner_eps==0), apply_iw (time_dist != uniform).
    """
    return _suggest_1d(trial, inner_eps_grid=[0.0, 0.05, 0.1], psb=True)


def suggest_hp_v2(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularCTSM V2 (barycentric path, vertex pinned 0.5 in the builder).

    switches: sched, time_dist. conditional: k (stiff), apply_iw (time_dist !=
    uniform). inner_eps pinned to 0: for bary_1d it clamps the same axis as
    eps (the outer tau), so the two knobs guard the same endpoint zero and
    inner_eps > eps merely produces a band of biased zero-gradient training
    mass (audit sec 2-3, notes/triangular_nogozone_audit.md). gamma_min is now
    unconditionally searched as the sole endpoint protection.
    """
    return _suggest_1d(trial, inner_eps_grid=[0.0], psb=False)


def suggest_hp_v3(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularCTSM V3 (2D stacked path + LowArcCurve2D).

    no time-sampling knobs (the builder hardcodes a product of uniforms). adds
    t2_max + path_height (path_height is inference-only: it parameterises the
    curve used only in predict_ldr). conditional: k (stiff). gamma_min searched
    unconditionally (the probe gates it on sched, a fragile edge we decline to
    encode). inner_eps pinned to 0: for rect_2d it clamps the same t1 axis as
    eps, so the two knobs guard the same endpoint zero (audit sec 2-3,
    notes/triangular_nogozone_audit.md). test_gamma_min derived from gamma_min.
    """
    hp: dict[str, Any] = {}
    _common_optim(trial, hp)
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)

    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    hp["inner_eps"] = 0.0
    hp["t2_max"] = trial.suggest_float("t2_max", 0.6, 0.9)
    hp["path_height"] = trial.suggest_float("path_height", 1.0, 2.0)
    hp["gamma_min"] = trial.suggest_float("gamma_min", 1e-2, 2e-1, log=True)

    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40, 80])

    _derive_test(hp)
    return hp
