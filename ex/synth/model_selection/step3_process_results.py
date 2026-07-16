"""step3: turn gathered step2 estimates into per-(kl, test_set) metrics.

reads  {raw_results_dir}/new_pstar.h5   (est_ldrs_arr_<method>, (nrows, ntest, nsamp))
       {data_dir}/dataset_newpstar.h5   (true_ldrs_arr, true_eldr_analytic_arr)
writes {processed_results_dir}/new_pstar.h5

the eldr error is the headline metric: |E_samples[est] - true_eldr|. its ground
truth is the ANALYTIC population ELDR (closed form for gaussians, zero MC noise);
the MC sample-mean version is kept alongside under an mc_ prefix for comparison.
the two measure different things -- the MC truth shares p*'s sampling noise with
est_eldr so it partly cancels (isolating the estimator's ratio-function bias),
whereas the analytic truth is the honest gap of the sample-estimated expectation
to the true population value.

paths and sample counts come from the variant config (see variants.py).
"""
import argparse
import os

import h5py
from einops import reduce
import numpy as np
from scipy import stats

from ex.synth.model_selection.dists import true_eldr_arr
from ex.synth.model_selection.variants import resolve


def spearman(est_ldrs, true_ldrs):
    """rank correlation per (row, test_set) -> (nrows, ntest)."""
    nrows, ntest = est_ldrs.shape[:2]
    out = np.zeros((nrows, ntest))
    for i in range(nrows):
        for j in range(ntest):
            out[i, j] = stats.spearmanr(est_ldrs[i, j, :], true_ldrs[i, j, :])[0]
    return out


def median_ae(est_ldrs, true_ldrs):
    """median |est - true| per (row, test_set) -> (nrows, ntest)."""
    return np.median(np.abs(est_ldrs - true_ldrs), axis=2)


def trimmed_mae_iqr(est_ldrs, true_ldrs):
    """mae over points inside the iqr of the true ldrs -> (nrows, ntest)."""
    nrows, ntest = est_ldrs.shape[:2]
    out = np.zeros((nrows, ntest))
    for i in range(nrows):
        for j in range(ntest):
            true_vals, est_vals = true_ldrs[i, j, :], est_ldrs[i, j, :]
            q1, q3 = np.percentile(true_vals, [25, 75])
            mask = (true_vals >= q1) & (true_vals <= q3)
            out[i, j] = (np.mean(np.abs(est_vals[mask] - true_vals[mask]))
                         if mask.sum() else np.nan)
    return out


def stratified_mae_quartiles(est_ldrs, true_ldrs):
    """mae stratified by quartile of the true ldr -> {q1..q4: (nrows, ntest)}."""
    nrows, ntest = est_ldrs.shape[:2]
    out = {f'q{q}': np.zeros((nrows, ntest)) for q in range(1, 5)}
    for i in range(nrows):
        for j in range(ntest):
            true_vals, est_vals = true_ldrs[i, j, :], est_ldrs[i, j, :]
            p25, p50, p75 = np.percentile(true_vals, [25, 50, 75])
            masks = [true_vals <= p25,
                     (true_vals > p25) & (true_vals <= p50),
                     (true_vals > p50) & (true_vals <= p75),
                     true_vals > p75]
            for q, mask in enumerate(masks, 1):
                out[f'q{q}'][i, j] = (np.mean(np.abs(est_vals[mask] - true_vals[mask]))
                                      if mask.sum() else np.nan)
    return out


def eldr_err_stats(est_by_alg, true_eldr, n_kl, n_inst, ntest):
    """|mean_samples(est) - true_eldr|, aggregated to per-(kl, test) mean +/- se."""
    means, ses = {}, {}
    for alg, est in est_by_alg.items():
        err = np.abs(est.mean(axis=2) - true_eldr).reshape(n_kl, n_inst, ntest)
        means[alg] = err.mean(axis=1)
        ses[alg] = err.std(axis=1, ddof=1) / np.sqrt(n_inst)
    return means, ses


def main(variant=None):
    tag, config = resolve(variant)
    data_dir = config['data_dir']
    raw_dir = config['raw_results_dir']
    proc_dir = config['processed_results_dir']
    kl_distances = config['kl_distances']
    n_inst = config['num_instances_per_kl']
    ntest = config['ntest_sets']
    n_kl = len(kl_distances)

    with h5py.File(f'{raw_dir}/new_pstar.h5', 'r') as f:
        est_by_alg = {k.replace('est_ldrs_arr_', ''): f[k][:]
                      for k in f.keys() if k.startswith('est_ldrs_arr_')}

    with h5py.File(f'{data_dir}/dataset_newpstar.h5', 'r') as f:
        true_ldrs = f['true_ldrs_arr'][:]  # (nrows, ntest, nsamp)
        # analytic (population) eldr: prefer the stored field, else rebuild from
        # the gaussian params (datasets generated before the field was added).
        if 'true_eldr_analytic_arr' in f:
            true_eldr_analytic = f['true_eldr_analytic_arr'][:]
        else:
            true_eldr_analytic = true_eldr_arr(
                f['mu0_arr'][:], f['Sigma0_arr'][:],
                f['mu1_arr'][:], f['Sigma1_arr'][:]).astype(np.float32)

    maes, med_aes, spear, trim, strat = {}, {}, {}, {}, {}
    for alg, est in est_by_alg.items():
        shape = (n_kl, n_inst, ntest)
        maes[alg] = reduce(np.abs(est - true_ldrs), 'n t d -> n t', 'mean').reshape(shape)
        med_aes[alg] = median_ae(est, true_ldrs).reshape(shape)
        spear[alg] = spearman(est, true_ldrs).reshape(shape)
        trim[alg] = trimmed_mae_iqr(est, true_ldrs).reshape(shape)
        strat[alg] = {q: a.reshape(shape)
                      for q, a in stratified_mae_quartiles(est, true_ldrs).items()}

    true_eldr_mc = true_ldrs.mean(axis=2)  # (nrows, ntest)
    eldr_mean, eldr_se = eldr_err_stats(est_by_alg, true_eldr_analytic, n_kl, n_inst, ntest)
    mc_mean, mc_se = eldr_err_stats(est_by_alg, true_eldr_mc, n_kl, n_inst, ntest)

    os.makedirs(proc_dir, exist_ok=True)
    with h5py.File(f'{proc_dir}/new_pstar.h5', 'w') as f:
        for alg in est_by_alg:
            f.create_dataset(f'maes_by_kl_{alg}', data=maes[alg])
            f.create_dataset(f'median_aes_by_kl_{alg}', data=med_aes[alg])
            f.create_dataset(f'spearman_by_kl_{alg}', data=spear[alg])
            f.create_dataset(f'trimmed_mae_iqr_by_kl_{alg}', data=trim[alg])
            for q, arr in strat[alg].items():
                f.create_dataset(f'stratified_mae_{q}_by_kl_{alg}', data=arr)
            # primary metric (step4 plots these): eldr_err_* vs the ANALYTIC truth.
            f.create_dataset(f'eldr_err_{alg}_mean', data=eldr_mean[alg])
            f.create_dataset(f'eldr_err_{alg}_se', data=eldr_se[alg])
            # comparison copy vs the MC sample-mean truth; the mc_ prefix keeps it
            # out of step4's 'eldr_err_*_mean' method discovery.
            f.create_dataset(f'mc_eldr_err_{alg}_mean', data=mc_mean[alg])
            f.create_dataset(f'mc_eldr_err_{alg}_se', data=mc_se[alg])

    nsamp = config['nsamples_test']
    print('=' * 72)
    print(f'[{tag}] ELDR-error ground truth: ANALYTIC (population) vs MC ({nsamp}-sample mean)')
    print(f'{"method":26s} {"analytic":>10s} {"mc":>10s} {"delta":>10s}')
    for alg in sorted(eldr_mean):
        a, m = float(eldr_mean[alg].mean()), float(mc_mean[alg].mean())
        print(f'{alg:26s} {a:10.4f} {m:10.4f} {a - m:+10.4f}')
    print('=' * 72)
    print(f'wrote {proc_dir}/new_pstar.h5  ({len(est_by_alg)} methods)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', default=None,
                   help='variant tag; precedence --variant > $DPE_MS_VARIANT > default')
    main(p.parse_args().variant)
