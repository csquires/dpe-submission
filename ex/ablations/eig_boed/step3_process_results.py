"""step3: reduce gathered.h5 to per-(method,geometry) metrics.

primary metric: oracle_shortfall (info_oracle - info_actual).
secondary: info_gained, regret_by_round, rank_tau, rank_slope, failure_rate.
output: processed_results.h5 with oracle_shortfall_<m>_<g>+_lo/_hi, etc.
"""
import warnings

import numpy as np
import h5py
import yaml
import os
from scipy import stats as sp_stats


def median_iqr(values):
    """median and raw IQR (25th, 75th percentile) of the finite values.

    the standard median/IQR over the existing evaluation aggregation unit
    (the prior x seed studies for a given method,geometry).
    returns (median, q25, q75); (nan,nan,nan) if no finite values.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def _is_complete(state_arr):
    """vectorized state == COMPLETE check; handles bytes and str.

    args:
      state_arr: (N,) array of state values (bytes or str)

    returns:
      (N,) boolean array
    """
    return np.array([
        (s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s)) == "TrialState.COMPLETE"
        for s in state_arr
    ])


def _complete_scalar(s):
    """scalar state == COMPLETE check (single value, bytes or str)."""
    if isinstance(s, (bytes, np.bytes_)):
        s = s.decode()
    return str(s) == "TrialState.COMPLETE"


def info_gained(Sigma0, SigmaR, cfg):
    """information gained: 0.5*log(det(S0)/det(SR)); guard det<=0.

    args:
      Sigma0: (d, d) prior covariance
      SigmaR: (d, d) posterior after updates
      cfg: config dict with 'data_dim'

    returns:
      scalar float | nan

    procedure:
      1. det0 = det(Sigma0), detR = det(SigmaR)
      2. guard: det0 <= 0 or detR <= 0 or detR >= det0 -> nan
      3. return 0.5*log(det0/detR)
    """
    try:
        det0 = np.linalg.det(Sigma0)
        detR = np.linalg.det(SigmaR)
        if det0 <= 0 or detR <= 0 or detR >= det0:
            return np.nan
        return 0.5 * np.log(det0 / detR)
    except (np.linalg.LinAlgError, FloatingPointError):
        return np.nan


def oracle_shortfall(Sigma0, rounds_for_study, trials_for_study, cfg):
    """oracle replay with oracle's evolving Sigma each round.

    args:
      Sigma0: (d, d) prior covariance
      rounds_for_study: list of dicts with 'round_idx', 'Sigma_r', 'xi_r'
      trials_for_study: list of dicts with 'round_idx', 'trial_idx', 'xi', 'state'
      cfg: config dict with 'data_dim', 'sigma2'

    returns:
      scalar float | nan

    procedure:
      1. extract R from max(round_idx) in rounds; validate sorted
      2. SigmaR = rounds_for_study[-1]['Sigma_r'] (final posterior)
      3. info_actual = info_gained(Sigma0, SigmaR, cfg)
      4. oracle replay:
         Sig = Sigma0.copy()
         for r in 0..R-1:
           C = [trials where round_idx==r and state==COMPLETE]
           if len(C) == 0: return nan
           e = [0.5*log1p(c @ Sig @ c / sigma2) for c in C]
           idx = argmax(e)  (ties: first)
           xi_oracle = C[idx]
           Sig = inv(inv(Sig) + outer(xi_oracle,xi_oracle)/sigma2)
         info_oracle = info_gained(Sigma0, Sig, cfg)
      5. return info_oracle - info_actual
    """
    try:
        if not rounds_for_study or not trials_for_study:
            return np.nan

        R = max(r["round_idx"] for r in rounds_for_study) + 1
        SigmaR = rounds_for_study[-1]["Sigma_r"]
        info_actual = info_gained(Sigma0, SigmaR, cfg)

        Sig_oracle = Sigma0.copy()
        sigma2 = cfg["sigma2"]

        for r in range(R):
            # candidates from round r, complete only
            C = [
                t["xi"]
                for t in trials_for_study
                if t["round_idx"] == r and t["state"] == b"TrialState.COMPLETE"
            ]
            if len(C) == 0:
                return np.nan
            C = np.array(C)

            # oracle eig under oracle's belief
            e = np.array([0.5 * np.log1p(c @ Sig_oracle @ c / sigma2) for c in C])
            idx_oracle = int(np.argmax(e))  # ties: first
            xi_oracle_r = C[idx_oracle]

            # update oracle's belief
            try:
                Sig_inv = np.linalg.inv(Sig_oracle)
                Sig_inv = Sig_inv + np.outer(xi_oracle_r, xi_oracle_r) / sigma2
                Sig_oracle = np.linalg.inv(Sig_inv)
            except np.linalg.LinAlgError:
                return np.nan

        info_oracle = info_gained(Sigma0, Sig_oracle, cfg)
        return info_oracle - info_actual
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        return np.nan


def regret_by_round(Sigma0, rounds_for_study, trials_for_study, cfg):
    """per-round ANYTIME design regret; shape (R, T).

    R(r, t) = eig_star_r - true_eig(xi_hat(t)), where xi_hat(t) is the round-r
    design with the highest METHOD estimate among trials 0..t (honest
    best-so-far). non-increasing WITHIN a round; resets each round (fresh ceiling +
    fresh BO). unfurling the rows gives an anytime curve across all R*T trials.

    args:
      Sigma0: unused (kept for call-site compatibility)
      rounds_for_study: list of dicts with 'round_idx', 'eig_star_r'
      trials_for_study: list of dicts with 'round_idx','trial_idx','est_eig',
        'true_eig','state'
      cfg: config dict with 'n_rounds', 'n_trials'

    returns:
      (R, T) array; nan where a round has no complete trials, (0,0) on empty input
    """
    R = int(cfg["n_rounds"])
    T = int(cfg["n_trials"])
    try:
        if not rounds_for_study or not trials_for_study:
            return np.full((0, 0), np.nan)

        ceil_r = {int(rd["round_idx"]): float(rd["eig_star_r"]) for rd in rounds_for_study}
        out = np.full((R, T), np.nan)

        for r in range(R):
            # round-r complete trials, in trial order
            tr = sorted(
                (t for t in trials_for_study
                 if int(t["round_idx"]) == r and _complete_scalar(t["state"])),
                key=lambda t: int(t["trial_idx"]),
            )
            if not tr:
                continue
            cr = ceil_r.get(r, np.nan)
            best_est = -np.inf
            best_true = np.nan
            for i, t in enumerate(tr):
                if i >= T:
                    break
                e = float(t["est_eig"])
                if np.isfinite(e) and e > best_est:  # honest: select by OWN estimate
                    best_est = e
                    best_true = float(t["true_eig"])  # evaluate at TRUE eig
                out[r, i] = (cr - best_true) if np.isfinite(best_true) else np.nan

        return out
    except (FloatingPointError, ValueError, IndexError, KeyError):
        return np.full((0, 0), np.nan)


def rank_kendall_tau_b_startup(trials_for_study, cfg):
    """kendall tau-b on startup trials only (trial_idx < n_startup, state==COMPLETE).

    args:
      trials_for_study: dict-of-arrays with 'trial_idx', 'est_eig', 'true_eig', 'state' keys
      cfg: config dict with 'n_startup'

    returns:
      scalar float in [-1, 1] | nan

    procedure:
      1. filter: trial_idx < n_startup AND state==COMPLETE (vectorized mask)
      2. extract (est_eig, true_eig) pairs
      3. if len < 2: return nan
      4. scipy.stats.kendalltau(..., variant='b') -> tau
      5. return tau
    """
    n_startup = cfg["n_startup"]
    mask = (trials_for_study["trial_idx"] < n_startup) & _is_complete(trials_for_study["state"])
    # (N,) boolean array for startup complete trials
    est = trials_for_study["est_eig"][mask]
    true = trials_for_study["true_eig"][mask]
    if len(est) < 2:
        return np.nan
    tau, _ = sp_stats.kendalltau(est, true, variant="b")
    return tau


def rank_theil_sen_slope_startup(trials_for_study, cfg):
    """theil-sen slope of true_eig vs est_eig on startup trials.

    args:
      trials_for_study: dict-of-arrays with 'trial_idx', 'est_eig', 'true_eig', 'state' keys
      cfg: config dict with 'n_startup'

    returns:
      scalar float | nan

    procedure:
      1. filter: trial_idx < n_startup AND state==COMPLETE (vectorized mask)
      2. extract (est_eig, true_eig) pairs, keep finite
      3. if len < 2: return nan
      4. scipy.stats.theilslopes(...) -> slope
      5. return slope
    """
    n_startup = cfg["n_startup"]
    mask = (trials_for_study["trial_idx"] < n_startup) & _is_complete(trials_for_study["state"])
    # (N,) boolean array for startup complete trials
    est = trials_for_study["est_eig"][mask]
    true = trials_for_study["true_eig"][mask]
    mask_fin = np.isfinite(est) & np.isfinite(true)
    est_fin = est[mask_fin]
    true_fin = true[mask_fin]
    if len(est_fin) < 2:
        return np.nan
    slope, _, _, _ = sp_stats.theilslopes(true_fin, est_fin)
    return slope


def failure_rate(trials_for_study):
    """fraction of trials with state != COMPLETE.

    args:
      trials_for_study: dict-of-arrays with 'state' key

    returns:
      scalar float in [0, 1]

    procedure:
      1. n_complete = count(state == COMPLETE) via vectorized mask
      2. return n_complete / len(state)
    """
    if "state" not in trials_for_study or len(trials_for_study["state"]) == 0:
        return np.nan
    complete_mask = _is_complete(trials_for_study["state"])
    # (N,) boolean array
    n_complete = np.sum(complete_mask)
    return 1.0 - n_complete / len(trials_for_study["state"])


def per_round_failures(trials_for_study):
    """count failures per round.

    args:
      trials_for_study: dict-of-arrays with 'round_idx', 'state' keys

    returns:
      (R,) int array of counts

    procedure:
      1. R = max(round_idx) + 1
      2. for r in 0..R-1: count(round_idx==r and state!=COMPLETE) via vectorized mask
      3. return array
    """
    if "round_idx" not in trials_for_study or len(trials_for_study["round_idx"]) == 0:
        return np.array([], dtype=int)
    R = int(np.max(trials_for_study["round_idx"])) + 1
    complete_mask = _is_complete(trials_for_study["state"])
    # (N,) boolean array
    failures = np.zeros(R, dtype=int)
    for r in range(R):
        round_mask = trials_for_study["round_idx"] == r
        fail_mask = round_mask & ~complete_mask
        # (N,) boolean array for failures in round r
        failures[r] = int(np.sum(fail_mask))
    return failures


def bootstrap_resample_metric_ci(studies_by_method_geom, metric_fn, cfg):
    """bootstrap ci: resample (prior_idx, seed_rep) with replacement.

    args:
      studies_by_method_geom: list of study dicts
      metric_fn: callable (study -> scalar | nan)
      cfg: config dict with 'config_seed'

    returns:
      (point, lo, hi) tuple of floats

    procedure:
      1. values = [metric_fn(s) for s in studies]
      2. finite_values = filter nan
      3. if len==0: return (nan,nan,nan)
      4. point = median(finite_values)
      5. rng = default_rng(cfg["config_seed"])
      6. boots = []
         for b in range(1000):
           idx = rng.choice(len(studies), size=len(studies), replace=True)
           resampled = [values[i] for i in idx]
           resampled_finite = filter nan
           if len>0: boots.append(median(resampled_finite))
           else: boots.append(nan)
      7. finite_boots = filter nan
      8. if len==0: return (nan,nan,nan)
      9. lo = percentile(finite_boots, 25)
         hi = percentile(finite_boots, 75)
      10. return (point, lo, hi)
    """
    values = [metric_fn(s) for s in studies_by_method_geom]
    finite_values = [v for v in values if np.isfinite(v)]
    if len(finite_values) == 0:
        return (np.nan, np.nan, np.nan)

    point = np.median(finite_values)

    rng = np.random.default_rng(cfg["config_seed"])
    n_bootstrap = 1000
    boots = []
    for b in range(n_bootstrap):
        idx = rng.choice(len(studies_by_method_geom), size=len(studies_by_method_geom), replace=True)
        resampled = [values[i] for i in idx]
        resampled_finite = [v for v in resampled if np.isfinite(v)]
        if len(resampled_finite) > 0:
            boots.append(np.median(resampled_finite))
        else:
            boots.append(np.nan)

    finite_boots = [v for v in boots if np.isfinite(v)]
    if len(finite_boots) == 0:
        return (np.nan, np.nan, np.nan)

    lo = np.percentile(finite_boots, 25)
    hi = np.percentile(finite_boots, 75)
    return (point, lo, hi)


def wilcoxon_signed_rank_pair(method_a_dict, method_b_dict):
    """wilcoxon signed-rank test: paired comparison.

    args:
      method_a_dict: dict {study_id -> scalar | nan}
      method_b_dict: dict {study_id -> scalar | nan}

    returns:
      dict with 'statistic', 'pvalue', 'n_pairs'

    procedure:
      1. find common study_ids (intersection), sort stable order
      2. n_pairs = len(common), unconditional
      3. extract pairwise differences: d[i] = a[common[i]] - b[common[i]]
      4. filter finite; if none or n_pairs<1: return nan,nan + n_pairs
      5. scipy.stats.wilcoxon(finite_diffs, method='auto') with try/except -> nan on fail
      6. return stat, pval, n_pairs (from common, not finite_diffs)
    """
    common = sorted(set(method_a_dict.keys()) & set(method_b_dict.keys()))
    # (M,) sorted common study ids
    n_pairs = len(common)

    if n_pairs < 1:
        return {"statistic": np.nan, "pvalue": np.nan, "n_pairs": n_pairs}

    diffs = np.array([method_a_dict[k] - method_b_dict[k] for k in common])
    # (M,) paired differences
    diffs_finite = diffs[np.isfinite(diffs)]

    if len(diffs_finite) < 1:
        return {"statistic": np.nan, "pvalue": np.nan, "n_pairs": n_pairs}

    try:
        stat, pval = sp_stats.wilcoxon(diffs_finite, method="auto")
    except (ValueError, RuntimeError):
        return {"statistic": np.nan, "pvalue": np.nan, "n_pairs": n_pairs}

    return {"statistic": stat, "pvalue": pval, "n_pairs": n_pairs}


def friedman_omnibus_test(oracle_shortfall_by_method):
    """friedman test: differences across methods?

    args:
      oracle_shortfall_by_method: dict {method_name -> {study_id -> scalar | nan}}

    returns:
      dict with 'statistic', 'pvalue', 'n_methods'

    procedure:
      1. intersect study_ids across methods
      2. if n_common < 3: return nan,nan,0
      3. build (n_methods, n_common) matrix M[m,s] = shortfall[m][s]
      4. scipy.stats.friedmanchisquare(*rows)
      5. return dict
    """
    if not oracle_shortfall_by_method:
        return {"statistic": np.nan, "pvalue": np.nan, "n_methods": 0}

    methods = list(oracle_shortfall_by_method.keys())
    common = None
    for method in methods:
        if common is None:
            common = set(oracle_shortfall_by_method[method].keys())
        else:
            common = common & set(oracle_shortfall_by_method[method].keys())
    common = list(common)

    if len(common) < 3:
        return {"statistic": np.nan, "pvalue": np.nan, "n_methods": 0}

    rows = []
    for method in methods:
        row = [oracle_shortfall_by_method[method][s] for s in common]
        rows.append(row)

    stat, pval = sp_stats.friedmanchisquare(*rows)
    return {"statistic": stat, "pvalue": pval, "n_methods": len(methods)}


def impute_study_attrition(oracle_shortfall_dict, cfg):
    """impute nan shortfalls with cfg["eig_max"] (worst-case).

    args:
      oracle_shortfall_dict: dict {study_id -> float | nan}
      cfg: config dict with 'eig_max'

    returns:
      dict with same keys, nans replaced
    """
    imputed = oracle_shortfall_dict.copy()
    eig_max = cfg["eig_max"]
    for k in imputed:
        if not np.isfinite(imputed[k]):
            imputed[k] = eig_max
    return imputed


def compute_study_metrics(study_id, studies_row, rounds_rows, trials_rows, cfg):
    """single study: compute all metrics.

    args:
      study_id: int
      studies_row: dict with 'Sigma0'
      rounds_rows: list of dicts with 'round_idx', 'Sigma_r', 'xi_r'
      trials_rows: list of dicts with 'round_idx', 'trial_idx', 'xi', 'est_eig', 'true_eig', 'state'
      cfg: config dict

    returns:
      dict with keys:
        'oracle_shortfall', 'info_gained', 'regret_by_round',
        'rank_tau', 'rank_slope', 'failure_rate', 'per_round_failures',
        'startup_est', 'startup_true'
    """
    Sigma0 = studies_row["Sigma0"]
    Sigma_R = rounds_rows[-1]["Sigma_r"] if rounds_rows else Sigma0

    # convert trials_rows from list-of-dicts to dict-of-arrays (columnar)
    if trials_rows:
        trials_cols = {col: np.array([t[col] for t in trials_rows]) for col in trials_rows[0]}
    else:
        trials_cols = {}

    metrics = {}
    metrics["info_gained"] = info_gained(Sigma0, Sigma_R, cfg)
    metrics["oracle_shortfall"] = oracle_shortfall(Sigma0, rounds_rows, trials_rows, cfg)
    metrics["regret_by_round"] = regret_by_round(Sigma0, rounds_rows, trials_rows, cfg)
    metrics["rank_tau"] = rank_kendall_tau_b_startup(trials_cols, cfg)
    metrics["rank_slope"] = rank_theil_sen_slope_startup(trials_cols, cfg)
    metrics["failure_rate"] = failure_rate(trials_cols)
    metrics["per_round_failures"] = per_round_failures(trials_cols)

    # startup arrays via vectorized mask
    n_startup = cfg["n_startup"]
    if trials_cols:
        mask = (trials_cols["trial_idx"] < n_startup) & _is_complete(trials_cols["state"])
        # (N,) boolean array for startup complete
        metrics["startup_est"] = trials_cols["est_eig"][mask]
        metrics["startup_true"] = trials_cols["true_eig"][mask]
    else:
        metrics["startup_est"] = np.array([])
        metrics["startup_true"] = np.array([])

    return metrics


def aggregate_by_method_geometry(gathered_h5_path, cfg):
    """load gathered.h5; group by (method, geometry); compute metrics; aggregate.

    args:
      gathered_h5_path: str path to gathered.h5
      cfg: config dict

    returns:
      dict of {(method, geometry) -> result_dict}
      plus: wilcoxon_stats, friedman_stats (separate dicts)
    """
    with h5py.File(gathered_h5_path, "r") as f:
        # read prefixed datasets
        studies_keys = [k for k in f.keys() if k.startswith("studies_")]
        rounds_keys = [k for k in f.keys() if k.startswith("rounds_")]
        trials_keys = [k for k in f.keys() if k.startswith("trials_")]

        # strip prefix and build study lookup
        studies_data = {}
        for key in studies_keys:
            col = key[8:]  # "studies_X" -> "X"
            studies_data[col] = f[key][()]

        rounds_data = {}
        for key in rounds_keys:
            col = key[7:]  # "rounds_X" -> "X"
            rounds_data[col] = f[key][()]

        trials_data = {}
        for key in trials_keys:
            col = key[7:]  # "trials_X" -> "X"
            trials_data[col] = f[key][()]

        # build per-study dicts
        n_studies = len(studies_data.get("Sigma0", []))
        studies_by_id = {}
        for i in range(n_studies):
            # key by the STORED study_id (matches rounds_/trials_ study_id);
            # positional i mismatches when study_ids are not dense 0..n
            sid = int(studies_data["study_id"][i]) if "study_id" in studies_data else i
            studies_by_id[sid] = {col: studies_data[col][i] for col in studies_data}

        # build per-study round/trial lists
        n_rounds_total = len(rounds_data.get("Sigma_r", []))
        n_trials_total = len(trials_data.get("xi", []))

        rounds_by_study = {}
        for i in range(n_rounds_total):
            study_id = int(rounds_data["study_id"][i])
            if study_id not in rounds_by_study:
                rounds_by_study[study_id] = []
            round_dict = {col: rounds_data[col][i] for col in rounds_data}
            rounds_by_study[study_id].append(round_dict)

        trials_by_study = {}
        for i in range(n_trials_total):
            study_id = int(trials_data["study_id"][i])
            if study_id not in trials_by_study:
                trials_by_study[study_id] = []
            trial_dict = {col: trials_data[col][i] for col in trials_data}
            trials_by_study[study_id].append(trial_dict)

    # compute per-study metrics
    all_metrics_by_study = {}
    for study_id in sorted(studies_by_id.keys()):
        studies_row = studies_by_id[study_id]
        rounds_rows = sorted(rounds_by_study.get(study_id, []), key=lambda r: r["round_idx"])
        trials_rows = trials_by_study.get(study_id, [])
        metrics = compute_study_metrics(study_id, studies_row, rounds_rows, trials_rows, cfg)
        all_metrics_by_study[study_id] = metrics
        studies_by_id[study_id].update(metrics)

    # group by (method, geometry)
    stratified = {}
    for study_id, study_dict in studies_by_id.items():
        method = study_dict["method"]
        if isinstance(method, bytes):
            method = method.decode("utf-8")
        geometry = study_dict["geometry"]
        if isinstance(geometry, bytes):
            geometry = geometry.decode("utf-8")
        key = (method, geometry)
        if key not in stratified:
            stratified[key] = []
        stratified[key].append(study_dict)

    # aggregate via bootstrap for each (method, geometry)
    results = {}
    for (method, geometry), studies_list in stratified.items():
        # impute attrition first
        shortfall_dict = {}
        for s in studies_list:
            study_id = s["study_id"]
            shortfall_dict[study_id] = s["oracle_shortfall"]
        shortfall_dict = impute_study_attrition(shortfall_dict, cfg)

        # re-inject imputed values for bootstrap
        for s in studies_list:
            study_id = s["study_id"]
            s["oracle_shortfall"] = shortfall_dict[study_id]

        # median + raw IQR (25/75) across the prior x seed studies
        result = {}
        result["oracle_shortfall"], result["oracle_shortfall_lo"], result["oracle_shortfall_hi"] = \
            median_iqr([s["oracle_shortfall"] for s in studies_list])
        result["info_gained"], result["info_gained_lo"], result["info_gained_hi"] = \
            median_iqr([s["info_gained"] for s in studies_list])

        # regret_by_round: per-(r,t) median + IQR across studies; PLUS a per-round
        # best-25% regret threshold (bottom quartile of regret pooled over trials x
        # cells) and trials-to-reach-it (earliest within-round trial per rollout
        # where regret <= that round's threshold; inf if never; pooled across
        # rounds AND rollouts -> median + IQR)
        regret_list = [np.asarray(s["regret_by_round"]) for s in studies_list
                       if s.get("regret_by_round") is not None
                       and np.asarray(s["regret_by_round"]).size > 0]
        if regret_list:
            shape = regret_list[0].shape  # (R, T)
            stack = np.stack([a for a in regret_list if a.shape == shape]).astype(float)  # (S,R,T)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
                result["regret_by_round"] = np.nanmedian(stack, axis=0).astype(np.float32)
                result["regret_by_round_lo"] = np.nanpercentile(stack, 25, axis=0).astype(np.float32)
                result["regret_by_round_hi"] = np.nanpercentile(stack, 75, axis=0).astype(np.float32)
            R_ = shape[0]
            q25 = np.full(R_, np.nan)
            ttq = []
            for r in range(R_):
                pool = stack[:, r, :].reshape(-1)
                pool = pool[np.isfinite(pool)]
                if pool.size == 0:
                    continue
                thr = float(np.percentile(pool, 25))  # best-25% regret level, round r
                q25[r] = thr
                for s in range(stack.shape[0]):
                    reg = stack[s, r, :]
                    hit = np.where(np.isfinite(reg) & (reg <= thr))[0]
                    ttq.append(float(hit[0]) if hit.size else np.inf)  # earliest, else inf
            result["regret_q25"] = q25.astype(np.float32)
            if ttq:
                arr = np.asarray(ttq, dtype=float)
                result["trials_to_q25"] = float(np.median(arr))
                result["trials_to_q25_lo"] = float(np.percentile(arr, 25))
                result["trials_to_q25_hi"] = float(np.percentile(arr, 75))
            else:
                result["trials_to_q25"] = result["trials_to_q25_lo"] = result["trials_to_q25_hi"] = np.nan
        else:
            result["regret_by_round"] = np.array([], dtype=np.float32)
            result["regret_by_round_lo"] = np.array([], dtype=np.float32)
            result["regret_by_round_hi"] = np.array([], dtype=np.float32)
            result["regret_q25"] = np.array([], dtype=np.float32)
            result["trials_to_q25"] = result["trials_to_q25_lo"] = result["trials_to_q25_hi"] = np.nan

        # rank metrics: median + IQR
        result["rank_tau"], result["rank_tau_lo"], result["rank_tau_hi"] = \
            median_iqr([s["rank_tau"] for s in studies_list])
        result["rank_slope"], result["rank_slope_lo"], result["rank_slope_hi"] = \
            median_iqr([s["rank_slope"] for s in studies_list])

        # failure metrics
        result["failure_rate"] = np.median([s["failure_rate"] for s in studies_list if np.isfinite(s["failure_rate"])])
        per_round_failures_list = [s["per_round_failures"] for s in studies_list]
        if per_round_failures_list and len(per_round_failures_list[0]) > 0:
            result["per_round_failures"] = np.sum(per_round_failures_list, axis=0)
        else:
            result["per_round_failures"] = np.array([])

        # startup bias scatter arrays
        startup_est_all = []
        startup_true_all = []
        for s in studies_list:
            startup_est_all.extend(s["startup_est"])
            startup_true_all.extend(s["startup_true"])
        result["startup_est"] = np.array(startup_est_all, dtype=np.float32)
        result["startup_true"] = np.array(startup_true_all, dtype=np.float32)

        results[(method, geometry)] = result

    # wilcoxon and friedman tests
    by_method = {}
    for (method, geometry), result in results.items():
        if method not in by_method:
            by_method[method] = {}
        by_method[method][geometry] = result

    wilcoxon_stats = {}
    methods_list = sorted(by_method.keys())
    for i, m1 in enumerate(methods_list):
        for m2 in methods_list[i + 1 :]:
            geom_common = set(by_method[m1].keys()) & set(by_method[m2].keys())
            for geom in geom_common:
                # extract oracle_shortfall dicts
                dict_m1 = {}
                dict_m2 = {}
                for (meth, gm), studies_list in stratified.items():
                    if meth == m1 and gm == geom:
                        for s in studies_list:
                            dict_m1[s["study_id"]] = s["oracle_shortfall"]
                    if meth == m2 and gm == geom:
                        for s in studies_list:
                            dict_m2[s["study_id"]] = s["oracle_shortfall"]
                wil = wilcoxon_signed_rank_pair(dict_m1, dict_m2)
                wilcoxon_stats[(m1, m2, geom)] = wil

    friedman_stats = {}
    geom_list = set()
    for method in by_method:
        geom_list.update(by_method[method].keys())
    for geom in geom_list:
        by_method_geom = {}
        for method in by_method:
            dict_meth = {}
            for (meth, gm), studies_list in stratified.items():
                if meth == method and gm == geom:
                    for s in studies_list:
                        dict_meth[s["study_id"]] = s["oracle_shortfall"]
            if dict_meth:
                by_method_geom[method] = dict_meth
        if by_method_geom:
            friedman_stats[geom] = friedman_omnibus_test(by_method_geom)

    return results, wilcoxon_stats, friedman_stats


def main(config_path="ex/ablations/eig_boed/config.yaml"):
    """orchestrate full step3 pipeline.

    args:
      config_path: str, path to config.yaml

    procedure:
      1. load config from yaml
      2. compute gathered_h5_path
      3. call aggregate_by_method_geometry
      4. open output_h5_path for writing
      5. for each (method,geometry) -> result:
         write datasets <metric>_<m>_<g>, _lo, _hi
      6. set root attrs
      7. print summary
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gathered_h5_path = os.path.join(cfg["processed_results_dir"], "gathered.h5")
    results, wilcoxon_stats, friedman_stats = aggregate_by_method_geometry(gathered_h5_path, cfg)

    output_h5_path = os.path.join(cfg["processed_results_dir"], "processed_results.h5")
    with h5py.File(output_h5_path, "w") as f:
        # write datasets
        for (method, geometry), result_dict in results.items():
            method_str = method if isinstance(method, str) else method.decode("utf-8")
            geometry_str = geometry if isinstance(geometry, str) else geometry.decode("utf-8")

            for key in result_dict:
                if key.endswith(("_lo", "_hi")):
                    continue
                value = result_dict[key]
                ds_name = f"{key}_{method_str}_{geometry_str}"

                if isinstance(value, (int, float, np.number)):
                    f.create_dataset(ds_name, data=np.float32(value))
                    f.create_dataset(f"{ds_name}_lo", data=np.float32(result_dict.get(f"{key}_lo", np.nan)))
                    f.create_dataset(f"{ds_name}_hi", data=np.float32(result_dict.get(f"{key}_hi", np.nan)))
                elif isinstance(value, np.ndarray):
                    f.create_dataset(ds_name, data=value.astype(np.float32))
                    lo = result_dict.get(f"{key}_lo", np.full_like(value, np.nan))
                    hi = result_dict.get(f"{key}_hi", np.full_like(value, np.nan))
                    f.create_dataset(f"{ds_name}_lo", data=lo.astype(np.float32))
                    f.create_dataset(f"{ds_name}_hi", data=hi.astype(np.float32))

        # root attrs
        f.attrs["n_rounds"] = cfg["n_rounds"]
        f.attrs["n_trials"] = cfg["n_trials"]
        f.attrs["n_startup"] = cfg["n_startup"]
        f.attrs["num_priors"] = cfg["num_priors"]
        f.attrs["n_seeds"] = cfg["n_seeds"]
        f.attrs["n_bootstrap"] = 1000
        f.attrs["bootstrap_seed"] = cfg["config_seed"]

    print(f"wrote {output_h5_path}")


if __name__ == "__main__":
    main()
