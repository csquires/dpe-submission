"""define-by-run optuna suggest_hp for TriangularCTSM V1/V2/V3.

translates the tuple-format search spaces from method_specs.py to trial.suggest_*
calls. conditional suggestion of inert params per the inertness edges in
notes/hpo_search_space_finalization.md (static scan + seeded double-build probe
in scratch/triangular_ctsm_inertness_probe.py). pins n_steps at N_STEPS
(uniform multi-fidelity resource axis).

three versions share most of the space; they differ in the path family:
  - V1 psb_1d (piecewise-Schroedinger-bridge), vertex + vertex_band searched.
  - V2 bary_1d (barycentric), vertex searched.
  - V3 rect_2d (2D stacked) + LowArcCurve2D; no time-sampling knobs (the
    builder hardcodes a product of uniforms); adds t2_max + path_height.

boundary protection. the search is pinned to sampler-side endpoint protection:
eps > 0 and inner_eps = 0. The path's local_tau clamp (inner_eps) remains
plumbed through the builders/method classes but is not currently searched (per
the 2026-05-31 audit, the clamp-mode alternative -- eps = 0, inner_eps > 0 --
introduces train/inference inconsistencies we have not finished untangling).

V1-only `vertex_band` is an independent sampler-side excision of the vertex
neighbourhood, always > 0 in V1. It is decoupled from `inner_eps`. The
inference linspace mirrors the same vertex_band excision via test_vertex_band
= vertex_band + test_vertex_band_offset, which guarantees the inference
excision is strictly wider than the training one.

inertness edges (probe + static):
  - k inert when sched == "bridge" (bridge_noise ignores k).
  - gamma_min searched for V1/V2 (the 2026-06 audit showed the gamma floor is
    the mechanism that protects the inference integrand near the path's
    gamma-zeros; the earlier 0.0 pin was based on circular winner evidence).
    V3 keeps the pin: its low-arc curve never approaches a gamma-zero.
  - apply_iw inert when time_dist == "uniform" (iw == 1).
  - V1 always samples time per-leg via a width-proportional two-leg mixture
    sampler (every TIME_DISTS value applied per leg), so time_dist is
    unconditional -- same treatment as V2, only the sampler differs
    (see notes/triangular_v1_time_dist_coupling.md).
  - V3 path_height is inference-only (curve used only in predict_ldr); searched
    unconditionally, probe blind.

test-path knobs (test_sched, test_sigma, test_inner_eps, test_gamma_min,
test_k, test_vertex_band) are derived from the train-side counterparts.
test_eps = max(eps, inner_eps) * test_eps_factor with test_eps_factor
log-uniform [0.8, 10]: a factor (not independent) seals the inference domain
to a subset of the trained tau range; the bound is widened from the old 2x cap
to 10x so methods can trim the lambda->0 endpoints as aggressively as stock.
test_vertex_band = vertex_band * (1 + test_vertex_band_frac) (V1 only).

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
    hp["lr"] = trial.suggest_float("lr", 3e-5, 3e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    hp["sigma"] = trial.suggest_float("sigma", 0.02, 10.0, log=True)
    hp["integration_steps"] = trial.suggest_categorical("integration_steps", [400, 1200, 2600])
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["activation"] = trial.suggest_categorical("activation", ["elu", "gelu", "silu"])
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    hp["reweight"] = trial.suggest_categorical("reweight", [False, True])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])
    hp["cosine_min_factor"] = 0.0
    hp["test_eps_factor"] = trial.suggest_float("test_eps_factor", 0.8, 10.0, log=True)


def _derive_test(hp: dict) -> None:
    """derive the test-path schedule from the train-side knobs.

    test_eps is a multiplicative factor on the train boundary, NOT independent:
    this seals the inference domain [test_eps, 1-test_eps] to a subset of the
    trained [eps, 1-eps] (factor >= 1) so we never integrate where the score was
    unsupervised. factor log-uniform [0.8, 10]: up to 10x trim for methods that
    want to avoid the lambda->0 endpoints, plus a small sub-1 ood slack.

    test_eps         = max(eps, inner_eps) * test_eps_factor
    test_vertex_band = vertex_band * (1 + test_vertex_band_frac)        [V1 only]
    """
    boundary = max(hp["eps"], hp["inner_eps"])
    hp["test_eps"] = boundary * hp["test_eps_factor"]
    hp["test_sched"] = hp["sched"]
    hp["test_sigma"] = hp["sigma"]
    hp["test_inner_eps"] = hp["inner_eps"]
    if "vertex_band" in hp:
        hp["test_vertex_band"] = hp["vertex_band"] * (1.0 + hp["test_vertex_band_frac"])
    else:
        hp["test_vertex_band"] = 0.0
    if "gamma_min" in hp:
        hp["test_gamma_min"] = hp["gamma_min"]
    if "k" in hp:
        hp["test_k"] = hp["k"]


def _suggest_1d(trial: optuna.Trial, *, psb: bool) -> dict[str, Any]:
    """shared body for the two 1d-time versions (V1 psb, V2 bary).

    psb=True selects the V1 piecewise-SB branch (adds vertex_band knob);
    psb=False selects V2 barycentric. Both versions search vertex. time_dist
    and apply_iw are always live for both.
    """
    hp: dict[str, Any] = {}
    _common_optim(trial, hp)
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
    hp["inner_eps"] = 0.0

    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    hp["vertex"] = trial.suggest_float("vertex", 0.1, 0.9)
    # gamma floor as a FRACTION of sigma, not absolute: the bridge noise peaks at
    # sigma*sqrt(2)*0.5 ~= 0.707*sigma, so an absolute gamma_min can exceed the
    # natural noise when sigma is small and flatten the whole path (degenerate,
    # broke occ TriCTSM_V2 2026-06-14). frac <= 0.5 keeps gamma_min < the peak so
    # it only floors the gamma-zeros (vertex + endpoints), never the bulk.
    hp["gamma_min"] = trial.suggest_float("gamma_min_frac", 1e-3, 0.5, log=True) * hp["sigma"]

    if psb:
        # V1-only: vertex_band controls sampler-side vertex excision and
        # the train half of inference-side excise band. always > 0.
        hp["vertex_band"] = trial.suggest_float("vertex_band", 1e-4, 1e-1, log=True)
        hp["test_vertex_band_frac"] = trial.suggest_float(
            "test_vertex_band_frac", 1e-4, 1.0, log=True,
        )

    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40, 80])

    time_dist = trial.suggest_categorical("time_dist", list(TIME_DISTS))
    hp["time_dist"] = time_dist
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    _derive_test(hp)
    return hp


def suggest_hp_v1(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularCTSM V1 (piecewise-SB path, searched vertex + vertex_band).

    switches: sched, time_dist. always: vertex_band, test_vertex_band_offset,
    gamma_min. conditional: k (stiff), apply_iw (time_dist != uniform).
    """
    return _suggest_1d(trial, psb=True)


def suggest_hp_v2(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularCTSM V2 (barycentric path, searched vertex).

    switches: sched, time_dist. always: vertex, gamma_min. conditional: k
    (stiff), apply_iw (time_dist != uniform).
    """
    return _suggest_1d(trial, psb=False)


def suggest_hp_v3(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularCTSM V3 (2D stacked path + LowArcCurve2D).

    reweight pinned to True (1/gamma^2 is the analytic answer for CTSM time-score
    regression; no longer a search dimension). No precond (CTSM has no precond
    plumbing). no time-sampling knobs (the builder hardcodes a product of uniforms).
    adds t2_max + path_height (path_height is inference-only). conditional: k (stiff).
    gamma_min pinned 0 (the low-arc inference curve never approaches a gamma-zero).
    """
    hp: dict[str, Any] = {}
    _common_optim(trial, hp)
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
    hp["inner_eps"] = 0.0

    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    hp["t2_max"] = trial.suggest_float("t2_max", 0.6, 0.9)
    hp["path_height"] = trial.suggest_float("path_height", 1.0, 2.0)
    # gamma_min pinned 0: eig V3 winner sat at 0.05 * g(eps), fully inert.
    hp["gamma_min"] = 0.0
    hp["reweight"] = True  # pinned True for V3 CTSM (per user direction 2026-05-31):
    # 1/gamma^2 is the analytically-correct outer weight for time-score regression
    # targets. don't search.

    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40, 80])

    _derive_test(hp)
    return hp
