"""define-by-run optuna suggest_hp for TriangularVFM V1/V2/V3.

translates the tuple-format search spaces from method_specs.py to trial.suggest_*
calls. conditional suggestion of inert params per the inertness edges in
notes/hpo_search_space_finalization.md (static scan + seeded double-build probe
in scratch/triangular_vfm_inertness_probe.py). pins n_steps at N_STEPS.

three versions differ in the path family (psb_1d / bary_1d / rect_2d), mirroring
TriangularCTSM. like stock VFM, the divergence estimator is pinned
(hutchinson/rademacher/4) and activation is pinned -- these are inference-side
knobs the train-side probe cannot rank, so a dedicated study is deferred.

boundary protection is pinned to sampler-side: eps > 0 and inner_eps = 0. The
path's local_tau clamp is plumbed through the builders but not searched.

V1-only vertex_band is an always-on knob (sampler-side excision of the vertex);
the inference excise band uses test_vertex_band = vertex_band +
test_vertex_band_offset, guaranteeing the inference excision is strictly wider
than the training one.

inertness edges (probe + static):
  - k inert when sched == "bridge".
  - gamma_min searched for V1/V2 (load-bearing inference floor; see 2026-06
    audit). V3 keeps the pin (low-arc curve avoids gamma-zeros).
  - V1 always samples time per-leg via a width-proportional two-leg mixture
    sampler (every TIME_DISTS value applied per leg), so time_dist is
    unconditional (see notes/triangular_v1_time_dist_coupling.md).
  - apply_iw inert when time_dist == "uniform".
  - reweight pinned to False for all variants; precond is the principled
    outer-weighting mechanism. all three versions search precond.
  - V3 (TriangularVFM2D) has no precond support and no time-sampling knobs;
    path_height is inference-only (curve used only in predict_ldr).

test-path knobs are derived from train-side counterparts. test_eps =
max(eps, inner_eps) * test_eps_factor, factor log-uniform [0.8, 10] -- a factor
(not independent) seals the inference domain to a subset of the trained tau
range; widened from the old 2x cap to 10x. test_vertex_band = vertex_band *
(1 + test_vertex_band_frac) (V1 only).
"""

from typing import Any

import optuna
from src.methods.reg.common._time_samplers import TIME_DISTS


N_STEPS = 6400


METADATA_V1 = {"uses_pruning": True, "requires_pstar": True, "builder": "build_TriangularVFM_V1"}
METADATA_V2 = {"uses_pruning": True, "requires_pstar": True, "builder": "build_TriangularVFM_V2"}
METADATA_V3 = {"uses_pruning": True, "requires_pstar": True, "builder": "build_TriangularVFM_V3"}


def _common(trial: optuna.Trial, hp: dict) -> None:
    """suggest the knobs shared by all three versions, including the pinned
    divergence/activation knobs (mirrors stock VFM)."""
    hp["n_steps"] = N_STEPS
    hp["lr"] = trial.suggest_float("lr", 3e-5, 3e-2, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    hp["sigma"] = trial.suggest_float("sigma", 0.02, 10.0, log=True)
    hp["eps"] = trial.suggest_float("eps", 1e-4, 1e-2, log=True)
    hp["inner_eps"] = 0.0
    hp["integration_steps"] = trial.suggest_categorical("integration_steps", [400, 1200, 2600])
    hp["ema_decay"] = trial.suggest_categorical("ema_decay", [None, 0.999, 0.9999])
    hp["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [None, 1.0, 5.0])
    hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    hp["layernorm"] = "off"
    hp["antithetic"] = trial.suggest_categorical("antithetic", [False, True])
    hp["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])
    hp["cosine_min_factor"] = 0.0
    hp["test_eps_factor"] = trial.suggest_float("test_eps_factor", 0.8, 10.0, log=True)

    # divergence estimator pinned to exact; div_noise / n_hutch_samples inert
    # under method=="exact" but kept set for downstream validation. activation
    # -> VFM default.
    hp["div_method"] = "exact"
    hp["div_noise"] = "rademacher"
    hp["n_hutch_samples"] = 4
    hp["activation"] = "silu"


def _derive_test(hp: dict) -> None:
    """derive the test-path schedule from the train-side knobs.

    test_eps is a multiplicative factor on the train boundary (NOT independent),
    sealing the inference domain to a subset of the trained [eps, 1-eps]:

    test_eps         = max(eps, inner_eps) * test_eps_factor   (factor [0.8, 10])
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

    psb=True adds vertex_band + test_vertex_band_offset (V1 only). Both
    versions search vertex.
    """
    hp: dict[str, Any] = {}
    _common(trial, hp)

    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    hp["vertex"] = trial.suggest_float("vertex", 0.1, 0.9)
    # gamma floor as a FRACTION of sigma (frac <= 0.5 -> below the 0.707*sigma
    # bridge-noise peak), so it floors only the gamma-zeros and can't flatten the
    # whole path when sigma is small (the degeneracy that broke occ TriCTSM_V2).
    hp["gamma_min"] = trial.suggest_float("gamma_min_frac", 1e-3, 0.5, log=True) * hp["sigma"]
    precond = trial.suggest_categorical("precond", [False, True])
    hp["precond"] = precond

    if psb:
        hp["vertex_band"] = trial.suggest_float("vertex_band", 1e-4, 1e-1, log=True)
        hp["test_vertex_band_frac"] = trial.suggest_float(
            "test_vertex_band_frac", 1e-4, 1.0, log=True,
        )

    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40, 80])

    # reweight pinned to False for all TriangularVFM variants (user direction
    # 2026-05-31); precond is the principled outer-weight mechanism.
    hp["reweight"] = False

    time_dist = trial.suggest_categorical("time_dist", list(TIME_DISTS))
    hp["time_dist"] = time_dist
    if time_dist != "uniform":
        hp["apply_iw"] = trial.suggest_categorical("apply_iw", [True, False])

    _derive_test(hp)
    return hp


def suggest_hp_v1(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularVFM V1 (piecewise-SB path, searched vertex + vertex_band).

    switches: sched, time_dist, precond. always: vertex_band,
    test_vertex_band_frac, gamma_min. reweight pinned to False. conditional:
    k (stiff), apply_iw (time_dist != uniform).
    """
    return _suggest_1d(trial, psb=True)


def suggest_hp_v2(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularVFM V2 (barycentric path, searched vertex, searched precond).

    switches: sched, time_dist, precond. always: vertex, gamma_min. reweight
    pinned to False. conditional: k (stiff), apply_iw (time_dist != uniform).
    """
    return _suggest_1d(trial, psb=False)


def suggest_hp_v3(trial: optuna.Trial) -> dict[str, Any]:
    """TriangularVFM V3 (2D stacked path + LowArcCurve2D).

    precond searched (True/False); reweight pinned to False (no longer a
    search dimension); gamma_min pinned to 0 (per user direction 2026-05-31:
    V3 VFM doesn't need the floor since precond + outer_path_var_v3's own
    gamma_eps cover the 1/gamma^2 stability concern).
    no time-sampling knobs (the builder hardcodes a product of uniforms).
    adds t2_max + path_height (inference-only). conditional: k (stiff).
    """
    hp: dict[str, Any] = {}
    _common(trial, hp)

    sched = trial.suggest_categorical("sched", ["stiff", "bridge"])
    hp["sched"] = sched
    hp["t2_max"] = trial.suggest_float("t2_max", 0.6, 0.9)
    hp["path_height"] = trial.suggest_float("path_height", 1.0, 2.0)
    hp["gamma_min"] = 0.0
    precond = trial.suggest_categorical("precond", [False, True])
    hp["precond"] = precond
    hp["reweight"] = False

    if sched == "stiff":
        hp["k"] = trial.suggest_categorical("k", [10, 20, 40, 80])

    _derive_test(hp)
    return hp
