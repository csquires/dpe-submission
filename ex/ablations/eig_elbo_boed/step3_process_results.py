"""step3: reduce eig_elbo_boed gathered.h5 to per-(method,geometry) metrics.

all reference quantities are RUNNING GROUND TRUTHS: the regret ceiling and the
shortfall oracle are scored under the exact-conjugate chain from the prior along
the realized designs (alpha=1 at every round), never the method's improper/local
estimated (DRE / mis-tempered) posterior.

metrics:
- regret (R,T): true-posterior anytime design regret. best-so-far tracked by
  TRUE eig under the running exact belief (regret_true_by_round), NOT est_eig.
- post_kl (R,) LOCAL: per-round tempering gap KL(alpha_hat-update || alpha=1-
  update of the SAME drifted pre-round belief); diagnoses the alpha-BO.
- post_kl_global (R,): KL(method's compounded belief || running TRUE posterior),
  the compounded truth-distance (reference = exact chain from the prior;
  alpha=1 is the true posterior only at round 0).
- post_kl_{local,global}_anytime (R,T_alpha): the two above, replayed over the
  alpha-BO incumbents; endpoints cross-check the per-round values.
- shortfall (R,): eig-oracle design info deficit (water-filling eigvec oracle
  vs logged designs, both under exact updates); provably >= 0.
- alpha_bias / alpha_abs_err (tempering miscalibration), rank_tau (design-search
  rank quality), fallback_rate.
output: processed_results.h5 with flat <metric>_<method>_<geometry> [+_lo/_hi].
"""
import argparse
import math
import os

import h5py
import numpy as np
import torch
import yaml
from scipy.linalg import eigh

from ex.ablations.eig_boed import priors as eig_priors
from ex.ablations.eig_boed.priors import eig_true
from ex.ablations.eig_boed.step3_process_results import (
    _complete_scalar,
    median_iqr,
    rank_kendall_tau_b_startup,
)
from ex.ablations.eig_elbo_boed.posterior_opt import post_kl
from ex.utils.fractional_posterior import get_fractional_posterior
from ex.utils.step2_runner.load_winners import list_methods, load_winners


def _decode(x):
    """bytes -> str (h5py vlen-string readback); pass through otherwise."""
    return x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x


def _read_table(f, prefix):
    """flat prefixed h5 columns -> dict of column arrays; {} if prefix absent."""
    return {k[len(prefix):]: f[k][()] for k in f.keys() if k.startswith(prefix)}


def _group_by_study(table, n):
    """columnar table (dict of (n,...) arrays) -> {study_id: [row dict, ...]}."""
    out = {}
    for i in range(n):
        sid = int(table["study_id"][i])
        out.setdefault(sid, []).append({k: table[k][i] for k in table})
    return out


def _top_eigvec(Sigma):
    """top eigenvector (max eigenvalue) of a symmetric matrix, float64."""
    _, evecs = eigh(Sigma)
    return evecs[:, -1]


def _exact_update(Sigma, xi, sigma2):
    """conjugate covariance-only update via eig_boed.priors.posterior_update.

    Sigma_new is y-independent, so mu/y_obs are dummy placeholders and only
    Sigma_new is used. float64 in, float64 out
    (posterior_update itself works internally in float64 and returns float32;
    widened back so downstream slogdet accumulation stays stable).
    """
    d = Sigma.shape[0]
    mu = torch.zeros(d, dtype=torch.float64)
    Sigma_t = torch.tensor(Sigma, dtype=torch.float64)
    xi_t = torch.tensor(xi, dtype=torch.float64)
    _, Sigma_new = eig_priors.posterior_update(mu, Sigma_t, xi_t, y_obs=0.0, sigma2=sigma2)
    return Sigma_new.double().numpy()


def _cum_info(Sigma0, traj):
    """cumulative 0.5*(logdet(Sigma0)-logdet(Sigma_t)), t=1..len(traj)-1.

    float64 slogdet; a non-finite or non-positive determinant at t yields nan
    for that entry only. returns (len(traj)-1,) array.
    """
    sign0, logdet0 = np.linalg.slogdet(Sigma0)
    out = np.full(len(traj) - 1, np.nan)
    if sign0 <= 0 or not np.isfinite(logdet0):
        return out
    for t in range(1, len(traj)):
        sign_t, logdet_t = np.linalg.slogdet(traj[t])
        if sign_t > 0 and np.isfinite(logdet_t):
            out[t - 1] = 0.5 * (logdet0 - logdet_t)
    return out


def shortfall_for_study(Sigma0, xi_r_seq, cfg):
    """exact-oracle shortfall per round, (R,); provably >=0 by construction.

    oracle trajectory: greedy top-eigenvector design + exact update, from
    Sigma0. exact-on-approx trajectory: the same exact update applied to the
    LOGGED xi_r design sequence. both start from the same Sigma0.

    why shortfall >= 0: each round's precision update adds xi xi^T/sigma2, a
    unit-trace (trace=1) rank-1 PSD term, so after R rounds the total added
    precision has trace exactly R regardless of which directions were chosen.
    maximizing logdet(P) subject to a trace budget on the added PSD mass is
    the classical water-filling problem (as in MIMO capacity maximization):
    the maximum is attained by a perturbation diagonal in P's current
    eigenbasis, allocated to equalize eigenvalues; sequential top-eigenvector
    selection realizes this one unit of trace at a time. the
    oracle trajectory is this water-filling allocation, so its cumulative
    info at every round t is the max achievable over ANY length-t unit-design
    sequence from Sigma0, including the logged approx one; shortfall_t is
    therefore a max minus an attained value: >= 0.

    args:
      Sigma0: (d,d) float64 prior covariance
      xi_r_seq: (R,d) logged design sequence, or None if unavailable
      cfg: dict with n_rounds, sigma2

    returns:
      (R,) float64 array; all-nan if xi_r_seq is missing/short or any logdet
      along either trajectory is non-finite (guards a near-singular Sigma).
    """
    R = int(cfg["n_rounds"])
    sigma2 = float(cfg["sigma2"])
    if xi_r_seq is None or len(xi_r_seq) < R:
        return np.full(R, np.nan)

    try:
        oracle_traj = [Sigma0]
        Sigma = Sigma0
        for _ in range(R):
            xi_o = _top_eigvec(Sigma)
            Sigma = _exact_update(Sigma, xi_o, sigma2)
            oracle_traj.append(Sigma)

        approx_traj = [Sigma0]
        Sigma = Sigma0
        for r in range(R):
            Sigma = _exact_update(Sigma, xi_r_seq[r], sigma2)
            approx_traj.append(Sigma)

        oracle_info = _cum_info(Sigma0, oracle_traj)
        approx_info = _cum_info(Sigma0, approx_traj)
    except (np.linalg.LinAlgError, RuntimeError, ValueError):
        return np.full(R, np.nan)

    if not (np.all(np.isfinite(oracle_info)) and np.all(np.isfinite(approx_info))):
        return np.full(R, np.nan)
    return oracle_info - approx_info


def _eig_true64(Sigma, xi, sigma2):
    """eig_boed.priors.eig_true via a float64 torch round-trip; numpy in/out."""
    Sigma_t = torch.tensor(np.asarray(Sigma), dtype=torch.float64)
    xi_t = torch.tensor(np.asarray(xi), dtype=torch.float64)
    return eig_true(Sigma_t, xi_t, sigma2)


def regret_true_by_round(Sigma0, rrows, trows, cfg):
    """(R,T) anytime design regret against the RUNNING TRUE posterior.

    unlike eig_boed's approx-own-ceiling regret (best-so-far by est_eig, ceiling
    from the mis-tempered belief), BOTH the ceiling and the best-so-far design
    are scored by TRUE eig under the exact-conjugate belief entering each round
    (exact chain from Sigma0 along the realized designs; Sigma is
    y-independent). the improper/local estimated posterior never enters the
    reference.

    procedure:
      exact_pre[0]=Sigma0; exact_pre[r]=_exact_update(exact_pre[r-1],
        xi_r_seq[r-1], sigma2)
      per round r:
        ceil_r = eig_true(exact_pre[r], top_eigvec(exact_pre[r]), sigma2)
        for design trial i (complete, in trial order) with logged design xi_i:
          te_i = eig_true(exact_pre[r], xi_i, sigma2)
          best_true = max_{j<=i} te_j;  R[r,i] = ceil_r - best_true  (>= 0)

    args:
      Sigma0: (d,d) float64 prior covariance
      rrows: per-round dicts (need xi_r), one per round
      trows: design-trial dicts (round_idx, trial_idx, xi, state)
      cfg: dict with n_rounds, n_trials, sigma2

    returns:
      (R,T) float64; nan where a round has no complete trials or the chain is
      non-finite; (0,0) when designs/trials are missing.
    """
    R = int(cfg["n_rounds"])
    T = int(cfg["n_trials"])
    sigma2 = float(cfg["sigma2"])
    rrows = sorted(rrows, key=lambda r: int(r["round_idx"]))
    has_xi = bool(rrows) and all("xi_r" in r for r in rrows)
    if not has_xi or not trows or len(rrows) < R:
        return np.full((0, 0), np.nan)
    xi_r_seq = np.stack([r["xi_r"] for r in rrows]).astype(np.float64)

    try:
        exact_pre = [np.asarray(Sigma0, dtype=np.float64)]
        for r in range(1, R):
            exact_pre.append(_exact_update(exact_pre[r - 1], xi_r_seq[r - 1], sigma2))
    except (np.linalg.LinAlgError, RuntimeError, ValueError):
        return np.full((0, 0), np.nan)

    out = np.full((R, T), np.nan)
    for r in range(R):
        Sr = exact_pre[r]
        if not np.all(np.isfinite(Sr)):
            continue
        ceil_r = _eig_true64(Sr, _top_eigvec(Sr), sigma2)
        tr = sorted((t for t in trows if int(t["round_idx"]) == r
                     and _complete_scalar(t["state"])),
                    key=lambda t: int(t["trial_idx"]))
        best_true = -np.inf
        for i, t in enumerate(tr):
            if i >= T:
                break
            te = _eig_true64(Sr, np.asarray(t["xi"], dtype=np.float64), sigma2)
            if np.isfinite(te) and te > best_true:
                best_true = te
            out[r, i] = (ceil_r - best_true) if np.isfinite(best_true) else np.nan
    return out


def _reconstruct_beliefs(mu0, Sigma0, rrows, cfg):
    """method (alpha_hat) and true-posterior (alpha=1) belief chains along the
    logged (xi_r, y_obs_r) sequence, float64 torch.

    the true posterior after round r is the running exact chain from the prior
    (alpha=1 at EVERY round), NOT alpha=1 applied to the method's drifted
    belief. means are y-dependent (kept), so both chains use the same logged
    (xi_r, y_obs_r); rrows must be round-sorted by the caller.

    returns dict of length-R lists of (mu (d,), Sigma (d,d)) float64 tensors:
      m_pre/m_post: method belief before/after each round's alpha_hat update
      t_pre/t_post: true posterior before/after each round's exact update
    """
    sigma = math.sqrt(float(cfg["sigma2"]))
    mm = torch.tensor(np.asarray(mu0), dtype=torch.float64)
    Sm = torch.tensor(np.asarray(Sigma0), dtype=torch.float64)
    mt, St = mm.clone(), Sm.clone()
    m_pre, m_post, t_pre, t_post = [], [], [], []
    for r in range(len(rrows)):
        xi = torch.tensor(np.asarray(rrows[r]["xi_r"]), dtype=torch.float64).reshape(-1, 1)
        y = torch.tensor(float(rrows[r]["y_obs_r"]), dtype=torch.float64)
        a = float(rrows[r]["alpha_r"])
        m_pre.append((mm, Sm))
        t_pre.append((mt, St))
        mm, Sm = get_fractional_posterior(mm, Sm, xi, y, a, sigma=sigma)
        mt, St = get_fractional_posterior(mt, St, xi, y, 1.0, sigma=sigma)
        m_post.append((mm, Sm))
        t_post.append((mt, St))
    return {"m_pre": m_pre, "m_post": m_post, "t_pre": t_pre, "t_post": t_post}


def post_kl_curves_for_study(mu0, Sigma0, rrows, atrows, cfg):
    """local + global posterior-KL curves (per-round global + both anytime).

    local reference  = alpha=1 update of the method's OWN drifted pre-round
      belief (the stored post_kl_r reference): per-round alpha-BO accuracy.
    global reference = the running true posterior (exact chain from the prior),
      so it also reflects compounded past drift.
    incumbent per alpha-BO trial = running argmax finite elbo_est (honest
    protocol; alpha=1 fallback until any finite), replayed through get_fractional
    _posterior + post_kl. anytime endpoints cross-check: local_anytime[r,-1] ==
    stored post_kl_r, global_anytime[r,-1] == post_kl_global[r].

    returns dict:
      post_kl_global         (R,)          KL(method belief_r || true posterior_r)
      post_kl_local_anytime  (R, T_alpha)  KL(incumbent update || local alpha=1)
      post_kl_global_anytime (R, T_alpha)  KL(incumbent update || true posterior_r)
    """
    R = int(cfg["n_rounds"])
    Ta = int(cfg["n_trials_alpha"])
    sigma = math.sqrt(float(cfg["sigma2"]))
    pk_global = np.full(R, np.nan)
    pk_local_at = np.full((R, Ta), np.nan)
    pk_global_at = np.full((R, Ta), np.nan)
    nan_ret = {"post_kl_global": pk_global,
               "post_kl_local_anytime": pk_local_at,
               "post_kl_global_anytime": pk_global_at}
    rrows = sorted(rrows, key=lambda r: int(r["round_idx"]))
    if len(rrows) < R:
        return nan_ret
    try:
        B = _reconstruct_beliefs(mu0, Sigma0, rrows, cfg)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return nan_ret

    by_round = {}
    for a in atrows:
        by_round.setdefault(int(a["alpha_round_idx"]), []).append(a)

    for r in range(R):
        mu_pre, Sigma_pre = B["m_pre"][r]
        xi = torch.tensor(np.asarray(rrows[r]["xi_r"]), dtype=torch.float64).reshape(-1, 1)
        y = torch.tensor(float(rrows[r]["y_obs_r"]), dtype=torch.float64)
        mu_loc1, Sigma_loc1 = get_fractional_posterior(mu_pre, Sigma_pre, xi, y, 1.0, sigma=sigma)
        mu_tru, Sigma_tru = B["t_post"][r]
        mu_mpost, Sigma_mpost = B["m_post"][r]
        pk_global[r] = post_kl(mu_mpost, Sigma_mpost, mu_tru, Sigma_tru)

        tr = sorted(by_round.get(r, []), key=lambda a: int(a["alpha_trial_idx"]))
        best_elbo = -np.inf
        inc_alpha = 1.0
        for i, a in enumerate(tr):
            if i >= Ta:
                break
            e = float(a["elbo_est"])
            if np.isfinite(e) and e > best_elbo:
                best_elbo = e
                inc_alpha = float(a["alpha_val"])
            mu_inc, Sigma_inc = get_fractional_posterior(mu_pre, Sigma_pre, xi, y, inc_alpha, sigma=sigma)
            pk_local_at[r, i] = post_kl(mu_inc, Sigma_inc, mu_loc1, Sigma_loc1)
            pk_global_at[r, i] = post_kl(mu_inc, Sigma_inc, mu_tru, Sigma_tru)

    return {"post_kl_global": pk_global,
            "post_kl_local_anytime": pk_local_at,
            "post_kl_global_anytime": pk_global_at}


def compute_study(mu0, Sigma0, rrows, trows, atrows, cfg):
    """single study's metric bundle from its own rounds/trials/alpha-trials rows.

    args:
      mu0: (d,) float64 prior mean (zeros; for the belief-chain reconstruction)
      Sigma0: (d,d) float64 prior covariance
      rrows: list of round dicts (post_kl_r, alpha_r, fell_back_r, eig_star_r,
        xi_r, y_obs_r, ...), one per round
      trows: list of design-trial dicts (round_idx, trial_idx, xi, state, ...);
        [] for the analytic channel (no design search)
      atrows: list of alpha-BO trial dicts (alpha_round_idx, alpha_trial_idx,
        alpha_val, elbo_est, ...)
      cfg: config dict

    returns:
      dict: post_kl (R,) LOCAL per-round, post_kl_global (R,), post_kl_local_
      anytime (R,T_alpha), post_kl_global_anytime (R,T_alpha), alpha (R,),
      fell_back (R,) bool, regret (R,T) true-posterior, rank_tau, shortfall (R,)
    """
    rrows = sorted(rrows, key=lambda r: int(r["round_idx"]))
    post_kl = np.array([float(r["post_kl_r"]) for r in rrows], dtype=float)  # local per-round
    alpha = np.array([float(r["alpha_r"]) for r in rrows], dtype=float)
    fell_back = np.array([bool(r["fell_back_r"]) for r in rrows], dtype=bool)

    regret = regret_true_by_round(Sigma0, rrows, trows, cfg)

    trows_cols = {k: np.array([t[k] for t in trows]) for k in trows[0]} if trows else {}
    rank_tau = rank_kendall_tau_b_startup(trows_cols, cfg) if trows_cols else np.nan

    has_xi = bool(rrows) and all("xi_r" in r for r in rrows)
    xi_r_seq = np.stack([r["xi_r"] for r in rrows]) if has_xi else None
    shortfall = shortfall_for_study(Sigma0, xi_r_seq, cfg)

    curves = post_kl_curves_for_study(mu0, Sigma0, rrows, atrows, cfg)

    return {
        "post_kl": post_kl,
        "post_kl_global": curves["post_kl_global"],
        "post_kl_local_anytime": curves["post_kl_local_anytime"],
        "post_kl_global_anytime": curves["post_kl_global_anytime"],
        "alpha": alpha,
        "fell_back": fell_back,
        "regret": regret,
        "rank_tau": rank_tau,
        "shortfall": shortfall,
    }


def _agg_curve(stack):
    """per-column median/iqr; stack is (S, ...) -> (med, lo, hi) of shape (...).

    generic over (S,R) and (S,R,T): median_iqr is applied along axis 0 for
    every trailing-index cell independently, without special-casing the rank.
    """
    trailing = stack.shape[1:]
    med = np.full(trailing, np.nan, dtype=np.float32)
    lo = np.full(trailing, np.nan, dtype=np.float32)
    hi = np.full(trailing, np.nan, dtype=np.float32)
    for idx in np.ndindex(trailing):
        m, q25, q75 = median_iqr(stack[(slice(None),) + idx])
        med[idx], lo[idx], hi[idx] = m, q25, q75
    return med, lo, hi


def reduce_group(studies_list, cfg):
    """median/iqr reduction of a (method,geometry) group's per-study metrics.

    args:
      studies_list: list of compute_study(...) result dicts (one per study)
      cfg: config dict with n_rounds, n_trials

    returns:
      flat dict of base_key -> value (+ base_key_lo/_hi where applicable),
      ready for the generic h5 writer.
    """
    R = int(cfg["n_rounds"])
    T = int(cfg["n_trials"])
    Ta = int(cfg["n_trials_alpha"])
    out = {}

    post_kl_stack = np.stack([s["post_kl"] for s in studies_list])  # (S,R) local per-round
    out["post_kl"], out["post_kl_lo"], out["post_kl_hi"] = _agg_curve(post_kl_stack)
    out["post_kl_summary"] = out["post_kl"][-1]
    out["post_kl_summary_lo"] = out["post_kl_lo"][-1]
    out["post_kl_summary_hi"] = out["post_kl_hi"][-1]

    pkg_stack = np.stack([s["post_kl_global"] for s in studies_list])  # (S,R)
    out["post_kl_global"], out["post_kl_global_lo"], out["post_kl_global_hi"] = \
        _agg_curve(pkg_stack)

    # anytime KL curves (S,R,T_alpha): local vs own-belief exact, global vs true posterior
    for base in ("post_kl_local_anytime", "post_kl_global_anytime"):
        curves = [s[base] for s in studies_list if s[base].shape == (R, Ta)]
        if curves:
            out[base], out[f"{base}_lo"], out[f"{base}_hi"] = _agg_curve(np.stack(curves))
        else:
            nan_rta = np.full((R, Ta), np.nan, dtype=np.float32)
            out[base], out[f"{base}_lo"], out[f"{base}_hi"] = nan_rta, nan_rta.copy(), nan_rta.copy()

    # alpha stats: per round r, exclude studies that fell back to alpha=1 AT r
    bias, bias_lo, bias_hi = np.full(R, np.nan), np.full(R, np.nan), np.full(R, np.nan)
    err, err_lo, err_hi = np.full(R, np.nan), np.full(R, np.nan), np.full(R, np.nan)
    for r in range(R):
        signed = [s["alpha"][r] - 1.0 for s in studies_list if not s["fell_back"][r]]
        bias[r], bias_lo[r], bias_hi[r] = median_iqr(signed)
        err[r], err_lo[r], err_hi[r] = median_iqr([abs(v) for v in signed])
    out["alpha_bias"], out["alpha_bias_lo"], out["alpha_bias_hi"] = bias, bias_lo, bias_hi
    out["alpha_abs_err"], out["alpha_abs_err_lo"], out["alpha_abs_err_hi"] = err, err_lo, err_hi

    regret_list = [s["regret"] for s in studies_list if s["regret"].shape == (R, T)]
    if regret_list:
        regret_stack = np.stack(regret_list)  # (S,R,T)
        out["regret_A"], out["regret_A_lo"], out["regret_A_hi"] = _agg_curve(regret_stack)
    else:
        nan_rt = np.full((R, T), np.nan, dtype=np.float32)
        out["regret_A"], out["regret_A_lo"], out["regret_A_hi"] = nan_rt, nan_rt.copy(), nan_rt.copy()

    shortfall_stack = np.stack([s["shortfall"] for s in studies_list])  # (S,R)
    out["shortfall"], out["shortfall_lo"], out["shortfall_hi"] = _agg_curve(shortfall_stack)

    out["rank_tau"], out["rank_tau_lo"], out["rank_tau_hi"] = \
        median_iqr([s["rank_tau"] for s in studies_list])

    fallback_fracs = [float(np.mean(s["fell_back"])) for s in studies_list]
    out["fallback_rate"] = float(np.mean(fallback_fracs)) if fallback_fracs else np.nan

    return out


def aggregate(gathered_h5_path, cfg):
    """load gathered.h5, group studies by (method,geometry), reduce each group.

    args:
      gathered_h5_path: str path to gathered.h5
      cfg: config dict

    returns:
      dict {(method, geometry): reduce_group(...) result}
    """
    with h5py.File(gathered_h5_path, "r") as f:
        studies = _read_table(f, "studies_")
        rounds = _read_table(f, "rounds_")
        trials = _read_table(f, "trials_")
        alpha_trials = _read_table(f, "alpha_trials_")

    n_studies = len(studies.get("study_id", []))
    rounds_by_study = _group_by_study(rounds, len(rounds.get("study_id", [])))
    trials_by_study = _group_by_study(trials, len(trials.get("study_id", [])))
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
        trows = trials_by_study.get(sid, [])
        atrows = alpha_trials_by_study.get(sid, [])

        metrics = compute_study(mu0, Sigma0, rrows, trows, atrows, cfg)
        grouped.setdefault((method, geometry), []).append(metrics)

    return {key: reduce_group(studies_list, cfg) for key, studies_list in grouped.items()}


def _write_metric(f, ds_name, value, lo, hi):
    """write value (+lo/hi if given) as float32 dataset(s) named ds_name[.._lo/_hi]."""
    f.create_dataset(ds_name, data=np.asarray(value, dtype=np.float32))
    if lo is not None:
        f.create_dataset(f"{ds_name}_lo", data=np.asarray(lo, dtype=np.float32))
        f.create_dataset(f"{ds_name}_hi", data=np.asarray(hi, dtype=np.float32))


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


def main(config_path="ex/ablations/eig_elbo_boed/config.yaml",
         winners_path="scratch/gold_winners/winners.elbo.yaml"):
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
        description="eig_elbo_boed step3: aggregate gathered.h5 into processed_results.h5"
    )
    parser.add_argument("--config", default="ex/ablations/eig_elbo_boed/config.yaml",
                         help="path to config.yaml")
    parser.add_argument("--winners", default="scratch/gold_winners/winners.elbo.yaml",
                         help="path to winners.yaml (defensive method filter)")
    args = parser.parse_args()

    main(args.config, args.winners)
