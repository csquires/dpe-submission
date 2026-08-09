"""step3: reduce elbo_boed gathered.h5 to per-(method,geometry) metrics.

analytic-design sibling of eig_elbo_boed: the design rule (top eigenvector
of the approx belief) is method-independent, so there are no design trials
to reduce and no regret_A or rank_tau metrics (regret_A is trivially 0 by
construction, and rank_tau has nothing to correlate without DRE-EIG design
search).

headline: shortfall (eig-oracle info deficit under the shared design rule,
PRIMARY; running-true-posterior oracle, provably >= 0). secondary: post_kl
LOCAL per-round tempering gap + post_kl_global (vs the running true posterior)
+ their anytime curves (post_kl_curves_for_study, reused from eig_elbo_boed),
alpha_bias / alpha_abs_err (tempering miscalibration), fallback_rate, and the
per-round mislocation diagnostic (mislocation_angle, mislocation_eiggap)
comparing the logged design against the exact-optimal design given the same
history. output: processed_results.h5 with flat <metric>_<method>_<geometry>
[+_lo/_hi] datasets, matching eig_elbo_boed's schema conventions.
"""
import argparse
import os

import h5py
import numpy as np
import torch
import yaml

from ex.ablations.eig_boed.priors import eig_true
from ex.ablations.eig_elbo_boed.step3_process_results import (
    _agg_curve,
    _decode,
    _exact_update,
    _group_by_study,
    _read_table,
    _top_eigvec,
    _write_metric,
    median_iqr,
    post_kl_curves_for_study,
    shortfall_for_study,
)
from ex.utils.step2_runner.load_winners import list_methods, load_winners


def _eig_true64(Sigma, xi, sigma2):
    """eig_boed.priors.eig_true via a float64 torch round-trip; numpy in/out."""
    Sigma_t = torch.tensor(Sigma, dtype=torch.float64)
    xi_t = torch.tensor(xi, dtype=torch.float64)
    return eig_true(Sigma_t, xi_t, sigma2)


def mislocation_for_study(Sigma0, xi_r_seq, cfg):
    """per-round design mislocation: angle + true-EIG gap vs. the exact-optimal
    design given the same history.

    procedure: build the belief ENTERING each round along the exact-update
    chain driven by the LOGGED xi sequence (Sigma is y-independent):
      exact_pre[0] = Sigma0
      exact_pre[r] = _exact_update(exact_pre[r-1], xi_r_seq[r-1], sigma2), r=1..R-1
    then per round r:
      xi_hat  = xi_r_seq[r]                    (the method's logged design)
      xi_star = top-eigvec(exact_pre[r])       (exact-optimal given the history)
      angle_r  = |cos(xi_hat, xi_star)|, in [0,1]; 1 = perfectly located
      eiggap_r = eig_true(exact_pre[r], xi_star) - eig_true(exact_pre[r], xi_hat)
    eiggap_r >= 0 by construction: xi_star maximizes eig_true(exact_pre[r], .)
    over unit designs (it is exact_pre[r]'s top eigenvector), so it is a
    ceiling minus an attained value.

    args:
      Sigma0: (d,d) float64 prior covariance
      xi_r_seq: (R,d) logged design sequence, or None if unavailable
      cfg: dict with n_rounds, sigma2

    returns:
      (angle, eiggap): two (R,) float64 arrays; all-nan where xi_r_seq is
      missing/short, or where a chain matrix along the way is non-finite.
    """
    R = int(cfg["n_rounds"])
    sigma2 = float(cfg["sigma2"])
    angle = np.full(R, np.nan)
    eiggap = np.full(R, np.nan)
    if xi_r_seq is None or len(xi_r_seq) < R:
        return angle, eiggap

    try:
        exact_pre = [np.asarray(Sigma0, dtype=np.float64)]
        for r in range(1, R):
            exact_pre.append(_exact_update(exact_pre[r - 1], xi_r_seq[r - 1], sigma2))
    except (np.linalg.LinAlgError, RuntimeError, ValueError):
        return angle, eiggap

    for r in range(R):
        Sigma_r = exact_pre[r]
        if not np.all(np.isfinite(Sigma_r)):
            continue
        try:
            xi_hat = xi_r_seq[r]
            xi_star = _top_eigvec(Sigma_r)
            denom = np.linalg.norm(xi_hat) * np.linalg.norm(xi_star)
            if denom > 0:
                angle[r] = abs(np.dot(xi_hat, xi_star)) / denom
            gap = _eig_true64(Sigma_r, xi_star, sigma2) - _eig_true64(Sigma_r, xi_hat, sigma2)
            if np.isfinite(gap):
                eiggap[r] = gap
        except (np.linalg.LinAlgError, RuntimeError, ValueError):
            continue

    return angle, eiggap


def compute_study(mu0, Sigma0, rrows, atrows, cfg):
    """single study's metric bundle from its own per-round / alpha-trial rows.

    args:
      mu0: (d,) float64 prior mean (for the belief-chain reconstruction)
      Sigma0: (d,d) float64 prior covariance
      rrows: list of round dicts (post_kl_r, alpha_r, fell_back_r, xi_r,
        y_obs_r, ...), one per round
      atrows: list of alpha-BO trial dicts (alpha_round_idx, alpha_trial_idx,
        alpha_val, elbo_est, ...)
      cfg: config dict with n_rounds, sigma2, n_trials_alpha

    returns:
      dict: post_kl (R,) LOCAL, post_kl_global (R,), post_kl_{local,global}_
      anytime (R,T_alpha), alpha (R,), fell_back (R,) bool, shortfall (R,),
      mislocation_angle (R,), mislocation_eiggap (R,)
    """
    rrows = sorted(rrows, key=lambda r: int(r["round_idx"]))
    post_kl = np.array([float(r["post_kl_r"]) for r in rrows], dtype=float)
    alpha = np.array([float(r["alpha_r"]) for r in rrows], dtype=float)
    fell_back = np.array([bool(r["fell_back_r"]) for r in rrows], dtype=bool)

    has_xi = bool(rrows) and all("xi_r" in r for r in rrows)
    xi_r_seq = np.stack([r["xi_r"] for r in rrows]).astype(np.float64) if has_xi else None

    shortfall = shortfall_for_study(Sigma0, xi_r_seq, cfg)
    angle, eiggap = mislocation_for_study(Sigma0, xi_r_seq, cfg)
    curves = post_kl_curves_for_study(mu0, Sigma0, rrows, atrows, cfg)

    return {
        "post_kl": post_kl,
        "post_kl_global": curves["post_kl_global"],
        "post_kl_local_anytime": curves["post_kl_local_anytime"],
        "post_kl_global_anytime": curves["post_kl_global_anytime"],
        "alpha": alpha,
        "fell_back": fell_back,
        "shortfall": shortfall,
        "mislocation_angle": angle,
        "mislocation_eiggap": eiggap,
    }


def reduce_group(studies_list, cfg):
    """median/iqr reduction of a (method,geometry) group's per-study metrics.

    args:
      studies_list: list of compute_study(...) result dicts (one per study)
      cfg: config dict with n_rounds

    returns:
      flat dict of base_key -> value (+ base_key_lo/_hi where applicable),
      ready for the generic h5 writer. no regret_A / rank_tau keys.
    """
    R = int(cfg["n_rounds"])
    Ta = int(cfg["n_trials_alpha"])
    out = {}

    post_kl_stack = np.stack([s["post_kl"] for s in studies_list])  # (S,R) local per-round
    out["post_kl"], out["post_kl_lo"], out["post_kl_hi"] = _agg_curve(post_kl_stack)

    pkg_stack = np.stack([s["post_kl_global"] for s in studies_list])  # (S,R)
    out["post_kl_global"], out["post_kl_global_lo"], out["post_kl_global_hi"] = \
        _agg_curve(pkg_stack)

    for base in ("post_kl_local_anytime", "post_kl_global_anytime"):
        curves = [s[base] for s in studies_list if s[base].shape == (R, Ta)]
        if curves:
            out[base], out[f"{base}_lo"], out[f"{base}_hi"] = _agg_curve(np.stack(curves))
        else:
            nan_rta = np.full((R, Ta), np.nan, dtype=np.float32)
            out[base], out[f"{base}_lo"], out[f"{base}_hi"] = nan_rta, nan_rta.copy(), nan_rta.copy()

    shortfall_stack = np.stack([s["shortfall"] for s in studies_list])  # (S,R)
    out["shortfall"], out["shortfall_lo"], out["shortfall_hi"] = _agg_curve(shortfall_stack)

    angle_stack = np.stack([s["mislocation_angle"] for s in studies_list])  # (S,R)
    out["mislocation_angle"], out["mislocation_angle_lo"], out["mislocation_angle_hi"] = \
        _agg_curve(angle_stack)

    eiggap_stack = np.stack([s["mislocation_eiggap"] for s in studies_list])  # (S,R)
    out["mislocation_eiggap"], out["mislocation_eiggap_lo"], out["mislocation_eiggap_hi"] = \
        _agg_curve(eiggap_stack)

    # alpha stats: per round r, exclude studies that fell back to alpha=1 AT r
    bias, bias_lo, bias_hi = np.full(R, np.nan), np.full(R, np.nan), np.full(R, np.nan)
    err, err_lo, err_hi = np.full(R, np.nan), np.full(R, np.nan), np.full(R, np.nan)
    for r in range(R):
        signed = [s["alpha"][r] - 1.0 for s in studies_list if not s["fell_back"][r]]
        bias[r], bias_lo[r], bias_hi[r] = median_iqr(signed)
        err[r], err_lo[r], err_hi[r] = median_iqr([abs(v) for v in signed])
    out["alpha_bias"], out["alpha_bias_lo"], out["alpha_bias_hi"] = bias, bias_lo, bias_hi
    out["alpha_abs_err"], out["alpha_abs_err_lo"], out["alpha_abs_err_hi"] = err, err_lo, err_hi

    fallback_fracs = [float(np.mean(s["fell_back"])) for s in studies_list]
    out["fallback_rate"] = float(np.mean(fallback_fracs)) if fallback_fracs else np.nan

    return out


def aggregate(gathered_h5_path, cfg):
    """load gathered.h5, group studies by (method,geometry), reduce each group.

    reads the studies_/rounds_/alpha_trials_ tables: the analytic channel has no
    design-trial table (n_trials=0) so regret_A/rank_tau are absent, but the
    alpha-BO trials drive the anytime post_kl curves (post_kl_curves_for_study).

    args:
      gathered_h5_path: str path to gathered.h5
      cfg: config dict

    returns:
      dict {(method, geometry): reduce_group(...) result}
    """
    with h5py.File(gathered_h5_path, "r") as f:
        studies = _read_table(f, "studies_")
        rounds = _read_table(f, "rounds_")
        alpha_trials = _read_table(f, "alpha_trials_")

    n_studies = len(studies.get("study_id", []))
    rounds_by_study = _group_by_study(rounds, len(rounds.get("study_id", [])))
    alpha_trials_by_study = _group_by_study(
        alpha_trials, len(alpha_trials.get("study_id", [])))

    grouped = {}
    for i in range(n_studies):
        sid = int(studies["study_id"][i])
        method = _decode(studies["method"][i])
        geometry = _decode(studies["geometry"][i])
        mu0 = np.asarray(studies["mu0"][i], dtype=np.float64)        # (d,)
        Sigma0 = np.asarray(studies["Sigma0"][i], dtype=np.float64)  # (d,d)

        rrows = rounds_by_study.get(sid, [])
        if not rrows:
            continue
        atrows = alpha_trials_by_study.get(sid, [])

        metrics = compute_study(mu0, Sigma0, rrows, atrows, cfg)
        grouped.setdefault((method, geometry), []).append(metrics)

    return {key: reduce_group(studies_list, cfg) for key, studies_list in grouped.items()}


def write_h5(output_path, results, cfg):
    """write results to processed_results.h5 as <key>_<method>_<geometry> [+_lo/_hi].

    procedure: for each (method,geometry) group, for each base metric key
    (skipping the _lo/_hi companions), write the metric plus its _lo/_hi pair
    when present in the result dict (fallback_rate has none).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metric_names = set()

    with h5py.File(output_path, "w") as f:
        for (method, geometry), result in results.items():
            for key in result:
                if key.endswith(("_lo", "_hi")):
                    continue
                metric_names.add(key)
                ds_name = f"{key}_{method}_{geometry}"
                lo = result.get(f"{key}_lo")
                hi = result.get(f"{key}_hi")
                _write_metric(f, ds_name, result[key], lo, hi)

        f.attrs["n_rounds"] = cfg["n_rounds"]
        f.attrs["n_trials"] = cfg["n_trials"]
        f.attrs["n_startup"] = cfg["n_startup"]
        f.attrs["num_priors"] = cfg["num_priors"]
        f.attrs["n_seeds"] = cfg["n_seeds"]
        f.attrs["n_trials_alpha"] = cfg["n_trials_alpha"]
        f.attrs["alpha_lo"] = cfg["alpha_lo"]
        f.attrs["alpha_hi"] = cfg["alpha_hi"]
        f.attrs["sigma2"] = cfg["sigma2"]
        f.attrs["input_dim"] = cfg["data_dim"]

    print(f"wrote {output_path}, {len(results)} (method,geometry) cells, "
          f"{len(metric_names)} metrics")


def main(config_path="ex/ablations/elbo_boed/config.yaml",
         winners_path="ex/synth/elbo/winners.yaml"):
    """orchestrate step3: load config, aggregate gathered.h5, write processed_results.h5.

    the winners roster is used only as a defensive filter against stale method
    names lingering in an old gathered.h5; a missing winners file is a no-op
    (gather already restricts studies to the winners set at write time).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gathered_h5_path = os.path.join(cfg["processed_results_dir"], "gathered.h5")
    results = aggregate(gathered_h5_path, cfg)

    try:
        methods = set(list_methods(load_winners(winners_path)))
        results = {k: v for k, v in results.items() if k[0] in methods}
    except FileNotFoundError:
        pass

    output_h5_path = os.path.join(cfg["processed_results_dir"], "processed_results.h5")
    write_h5(output_h5_path, results, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="elbo_boed step3: aggregate gathered.h5 into processed_results.h5"
    )
    parser.add_argument("--config", default="ex/ablations/elbo_boed/config.yaml",
                         help="path to config.yaml")
    parser.add_argument("--winners", default="ex/synth/elbo/winners.yaml",
                         help="path to winners.yaml (defensive method filter)")
    args = parser.parse_args()

    main(args.config, args.winners)
