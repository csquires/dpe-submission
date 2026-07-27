"""step3: turn gathered step2 estimates into per-(kl, p*_idx) metrics.

reads  raw_results/{experiment_name(tag,route)}_results.h5  (est_ldrs_arr_<method>)
       dataset.h5  (true_ldrs_arr, true_eldr_arr)
writes processed_results/{route}_{tag}.h5

computes per-KL (7 strata) metrics: bias_signed, variance, bias_sq_var, eldr_err_mean,
per-sample MAE, plus standard errors. no scalar pools across p*_idx. missing winners
are logged and skipped.
"""
import argparse
import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np

from ex.ablations.dokls import variants


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_winners(tag, route, dpe_data_root):
    """load best_hp for all methods that have completed hpo.

    returns: dict[str, dict] mapping method -> best_hp dict. missing methods logged.
    """
    exp_name = variants.experiment_name(tag, route)
    holdout_dir = Path(dpe_data_root) / "holdout" / exp_name
    winners = {}

    if not holdout_dir.exists():
        log.info(f"no holdout dir for {exp_name}")
        return winners

    for method_dir in holdout_dir.iterdir():
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        # unsliced dokls writes best_hp.json directly in the method dir; sliced
        # experiments nest it under slice_<x>/. match both (assemble_winners uses **).
        hp_file = list(method_dir.glob("best_hp.json")) + list(method_dir.glob("*/best_hp.json"))
        if hp_file:
            with open(hp_file[0]) as f:
                winners[method] = json.load(f)
        else:
            log.info(f"skipped {route} {method}: no winner")

    return winners


def compute_metrics(est_ldrs, true_ldrs, true_eldr):
    """compute per-kl metrics from (70,) and (70, N) arrays.

    est_ldrs: shape (70, N_eff) estimated log-ratios
    true_ldrs: shape (70, N_eff) ground-truth log-ratios
    true_eldr: shape (70,) ground-truth ELDR

    returns: dict mapping metric_name -> array of shape (7,)
    """
    # in-sample MC average
    est_eldr = est_ldrs.mean(axis=1)  # (70,)

    # signed error
    signed_err = est_eldr - true_eldr  # (70,)

    # per-sample error
    per_sample_err = np.abs(est_ldrs - true_ldrs)  # (70, N_eff)
    per_sample_mae = per_sample_err.mean(axis=1)  # (70,)

    # reshape to (7, 10) for per-kl aggregation
    signed_err_kl = signed_err.reshape(7, 10)  # kl_idx = row // 10
    per_sample_mae_kl = per_sample_mae.reshape(7, 10)
    est_eldr_kl = est_eldr.reshape(7, 10)
    true_eldr_kl = true_eldr.reshape(7, 10)

    # aggregate over instances (axis=1, 10 per KL)
    bias_signed = signed_err_kl.mean(axis=1)  # (7,)
    bias_signed_se = signed_err_kl.std(axis=1, ddof=1) / np.sqrt(10)  # (7,)

    variance = signed_err_kl.var(axis=1, ddof=1)  # (7,)
    variance = np.clip(variance, 1e-12, None)  # guard against numerical noise

    sq_dev_kl = (signed_err_kl - bias_signed[:, None]) ** 2  # (7, 10)
    variance_se = sq_dev_kl.std(axis=1, ddof=1) / np.sqrt(10)  # (7,)

    bias_sq_var = (bias_signed ** 2) / (variance + 1e-12)  # (7,)

    unsigned_err = np.abs(est_eldr_kl - true_eldr_kl)  # (7, 10)
    eldr_err_mean = unsigned_err.mean(axis=1)  # (7,)
    eldr_err_se = unsigned_err.std(axis=1, ddof=1) / np.sqrt(10)  # (7,)

    mae = per_sample_mae_kl.mean(axis=1)  # (7,)

    return {
        'bias_signed': bias_signed,
        'bias_signed_se': bias_signed_se,
        'variance': variance,
        'variance_se': variance_se,
        'bias_sq_var': bias_sq_var,
        'eldr_err_mean': eldr_err_mean,
        'eldr_err_se': eldr_err_se,
        'mae': mae,
    }


def regret_stats(est_by_method, true_eldr, seed, n_boot=500):
    """cross-method per-cell normalized ELDR regret, per KL stratum.

    for each cell c (70 = 7 kl x 10 inst): regret[m,c] =
    (err[m,c] - min_m' err[m',c]) / (max_m' - min_m'); 0 = best method on that
    cell, 1 = worst, 0 on an exact tie. the point per KL is the median over the
    10 instances; the band is the bootstrap std of that median. being scale-free
    it shows relative standing and de-emphasizes magnitude/blowups (ported from
    ex.synth.model_selection.step3, dropping its ntest axis). est_by_method[m]:
    (70, N); true_eldr: (70,). returns (means, ses) dicts of (7,) float32 arrays.
    """
    methods = list(est_by_method)
    err = np.stack([np.abs(est_by_method[m].mean(axis=1) - true_eldr).reshape(7, 10)
                    for m in methods]).astype(np.float64)   # (M, 7, 10)
    finite = np.isfinite(err)
    best = np.where(finite, err, np.inf).min(axis=0)         # (7, 10)
    worst = np.where(finite, err, -np.inf).max(axis=0)       # (7, 10)
    span = worst - best
    denom = np.where(span > 0, span, np.nan)
    reg = (err - best[None]) / denom[None]                   # (M, 7, 10)
    tied = (span == 0) & np.isfinite(best)
    reg = np.where(np.broadcast_to(tied[None], reg.shape) & finite, 0.0, reg)

    rng = np.random.default_rng(seed)
    boot = rng.integers(0, 10, size=(n_boot, 10))
    means, ses = {}, {}
    for mi, m in enumerate(methods):
        r = reg[mi]                                          # (7, 10)
        means[m] = np.nanmedian(r, axis=1).astype(np.float32)          # (7,)
        bmed = np.nanmedian(r[:, boot], axis=2)              # (7, n_boot)
        ses[m] = np.nanstd(bmed, axis=1).astype(np.float32)            # (7,)
    return means, ses


def process_tag_route(tag, route, dpe_data_root):
    """process a single (tag, route) pair. returns output path or None if skipped."""
    tag, resolved_route, config = variants.resolve(tag, route)

    data_dir = config['data_dir']
    raw_dir = config['raw_results_dir']
    proc_dir = config['processed_results_dir']

    # load ground truth
    with h5py.File(f'{data_dir}/dataset.h5', 'r') as f:
        true_eldr = f['true_eldr_arr'][:]  # (70, 2)
        true_ldrs = f['true_ldrs_arr'][:]  # (70, 2, 8192)

    # get p*_idx and N from tag
    # N = p0/p1 budget (goes in the result keys); Nstar = p* budget (slices the
    # per-sample p* ldrs and est_ldrs, which are Nstar-long for decoupled cells).
    p_star_idx, N, Nstar = variants.VARIANTS[tag]
    true_eldr_eff = true_eldr[:, p_star_idx]  # (70,)
    true_ldrs_eff = true_ldrs[:, p_star_idx, :Nstar]  # (70, Nstar)

    # load winners to find available methods
    winners = load_winners(tag, resolved_route, dpe_data_root)

    # load raw_results for this (tag, route)
    exp_name = variants.experiment_name(tag, resolved_route)
    raw_path = f'{raw_dir}/{exp_name}_results.h5'
    if not Path(raw_path).exists():
        log.warning(f"raw_results not found: {raw_path}")
        return None

    with h5py.File(raw_path, 'r') as f:
        est_by_method = {}
        for method in winners.keys():
            key = f'est_ldrs_arr_{method}'
            if key not in f:
                log.info(f"missing {key} in raw_results")
                continue
            est_by_method[method] = f[key][:, :Nstar]  # (70, Nstar)

    if not est_by_method:
        log.warning(f"no methods available for {tag} {resolved_route}")
        return None

    # compute metrics for each method
    results = {}
    for method, est_ldrs in est_by_method.items():
        if est_ldrs.shape[0] != 70:
            log.error(f"est_ldrs shape {est_ldrs.shape} != (70, N)")
            raise ValueError(f"est_ldrs shape mismatch for {method}")

        metrics = compute_metrics(est_ldrs, true_ldrs_eff, true_eldr_eff)

        for metric_name, arr in metrics.items():
            key = f'{metric_name}_{resolved_route}_{method}_{p_star_idx}_{N}'
            results[key] = arr.astype(np.float32)

    # cross-method regret (needs all methods' est together; computed once)
    reg_mean, reg_se = regret_stats(est_by_method, true_eldr_eff,
                                    config.get('seed', 1729))
    for method in est_by_method:
        results[f'regret_mean_{resolved_route}_{method}_{p_star_idx}_{N}'] = reg_mean[method]
        results[f'regret_se_{resolved_route}_{method}_{p_star_idx}_{N}'] = reg_se[method]

    # guard: no pooling across p*_idx (every key must have p*_idx in it)
    for key in results.keys():
        # key format: {metric}_{route}_{method}_{p*_idx}_{N}
        # p*_idx is second-to-last element when split by '_' and considering trailing int N
        parts = key.split('_')
        N_val = int(parts[-1])
        p_idx_val = int(parts[-2])
        assert p_idx_val == p_star_idx, f"p*_idx mismatch in key {key}: got {p_idx_val}, expected {p_star_idx}"

    # write output
    os.makedirs(proc_dir, exist_ok=True)
    out_path = f'{proc_dir}/{resolved_route}_{tag}.h5'
    with h5py.File(out_path, 'w') as f:
        for key, arr in results.items():
            dset = f.create_dataset(key, data=arr)
            # parse key: {metric}_{route}_{method}_{p*_idx}_{N}
            # route and N are known; extract method from the middle
            suffix = f'_{p_star_idx}_{N}'
            if not key.endswith(suffix):
                raise ValueError(f"key {key} doesn't end with expected {suffix}")
            prefix = f'_{resolved_route}_'
            if prefix not in key:
                raise ValueError(f"key {key} doesn't contain expected {prefix}")
            # find the prefix position
            idx = key.index(prefix)
            metric = key[:idx]
            # method+suffix is everything after route
            middle = key[idx+len(prefix):]
            method = middle[:-len(suffix)]
            dset.attrs['n_kl'] = 7
            dset.attrs['n_instances_per_kl'] = 10
            dset.attrs['N_eff'] = N
            dset.attrs['p_star_idx'] = p_star_idx
            dset.attrs['route_method_N_combo'] = f"{resolved_route}_{method}_{N}"

    log.info(f'wrote {out_path} ({len(results)} metrics)')
    return out_path


def main(variant=None, route=None):
    if not variant:
        variant = os.environ.get('DPE_DOKLS_TAG')
    if not variant:
        variant = variants.DEFAULT_TAG

    # validate tag
    if variant not in variants.VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; valid: {sorted(variants.VARIANTS.keys())}")

    if not route:
        route = os.environ.get('DPE_DOKLS_ROUTE')

    dpe_data_root = os.environ.get('DPE_DATA_ROOT')
    if not dpe_data_root:
        raise RuntimeError("DPE_DATA_ROOT not set")

    # process route(s)
    all_routes = getattr(variants, 'ROUTES', ['two_leg', 'direct'])
    routes_to_process = [route] if route else all_routes
    for r in routes_to_process:
        if r not in all_routes:
            raise KeyError(f"unknown route {r!r}; valid: {all_routes}")
        process_tag_route(variant, r, dpe_data_root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', default=None,
                   help='variant tag; precedence --variant > $DPE_DOKLS_TAG > default')
    p.add_argument('--route', default=None,
                   help='route; precedence --route > $DPE_DOKLS_ROUTE > default')
    args = p.parse_args()
    main(args.variant, args.route)
