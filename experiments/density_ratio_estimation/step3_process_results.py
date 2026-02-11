import os

import h5py
from einops import reduce
import numpy as np
from scipy import stats
import yaml


config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)

# directories
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
PROCESSED_RESULTS_DIR = config['processed_results_dir']
# dataset parameters
DATA_DIM = config['data_dim']
KL_DIVERGENCES = config['kl_divergences']
NSAMPLES_TRAIN = config['nsamples_train']
NSAMPLES_TEST = config['nsamples_test']
NTEST_SETS = config['ntest_sets']
NUM_INSTANCES_PER_KL = config['num_instances_per_kl']

# raw_results_filename = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST},ntestsets={NTEST_SETS}.h5'
# dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST},ntestsets={NTEST_SETS}.h5'
# processed_results_filename = f'{PROCESSED_RESULTS_DIR}/metrics_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST},ntestsets={NTEST_SETS}.h5'

dataset_filename = f'{DATA_DIR}/dataset_newpstar.h5'
raw_results_filename = f'{RAW_RESULTS_DIR}/new_pstar.h5'
processed_results_filename = f'{PROCESSED_RESULTS_DIR}/new_pstar.h5'

with h5py.File(raw_results_filename, 'r') as f:
    result_keys = [key for key in f.keys() if key.startswith('est_ldrs_arr_')]
    est_ldrs_by_alg = {key.replace('est_ldrs_arr_', ''): f[key][:] for key in result_keys}

with h5py.File(dataset_filename, 'r') as f:
    true_ldrs_arr = f['true_ldrs_arr'][:]  # (nrows, NTEST_SETS, NSAMPLES_TEST)

nrows = true_ldrs_arr.shape[0]


def compute_spearman_correlation(est_ldrs, true_ldrs):
    """Compute Spearman correlation for each (row, test_set) pair."""
    # est_ldrs, true_ldrs: (nrows, NTEST_SETS, NSAMPLES_TEST)
    result = np.zeros((nrows, NTEST_SETS))
    for i in range(nrows):
        for j in range(NTEST_SETS):
            corr, _ = stats.spearmanr(est_ldrs[i, j, :], true_ldrs[i, j, :])
            result[i, j] = corr
    return result


def compute_median_ae(est_ldrs, true_ldrs):
    """Compute Median Absolute Error for each (row, test_set) pair."""
    absolute_errors = np.abs(est_ldrs - true_ldrs)
    return np.median(absolute_errors, axis=2)  # (nrows, NTEST_SETS)


def compute_trimmed_mae_iqr(est_ldrs, true_ldrs):
    """Compute MAE only for points within the IQR of true LDRs."""
    result = np.zeros((nrows, NTEST_SETS))
    for i in range(nrows):
        for j in range(NTEST_SETS):
            true_vals = true_ldrs[i, j, :]
            est_vals = est_ldrs[i, j, :]
            q1, q3 = np.percentile(true_vals, [25, 75])
            mask = (true_vals >= q1) & (true_vals <= q3)
            if mask.sum() > 0:
                result[i, j] = np.mean(np.abs(est_vals[mask] - true_vals[mask]))
            else:
                result[i, j] = np.nan
    return result


def compute_stratified_mae_quartiles(est_ldrs, true_ldrs):
    """Compute MAE stratified by quartiles of true LDR.

    Returns: dict with keys 'q1', 'q2', 'q3', 'q4', each (nrows, NTEST_SETS)
    """
    quartile_maes = {f'q{q}': np.zeros((nrows, NTEST_SETS)) for q in range(1, 5)}
    for i in range(nrows):
        for j in range(NTEST_SETS):
            true_vals = true_ldrs[i, j, :]
            est_vals = est_ldrs[i, j, :]
            percentiles = np.percentile(true_vals, [25, 50, 75])
            # Q1: 0-25th percentile
            mask_q1 = true_vals <= percentiles[0]
            # Q2: 25-50th percentile
            mask_q2 = (true_vals > percentiles[0]) & (true_vals <= percentiles[1])
            # Q3: 50-75th percentile
            mask_q3 = (true_vals > percentiles[1]) & (true_vals <= percentiles[2])
            # Q4: 75-100th percentile
            mask_q4 = true_vals > percentiles[2]

            for q, mask in enumerate([mask_q1, mask_q2, mask_q3, mask_q4], 1):
                if mask.sum() > 0:
                    quartile_maes[f'q{q}'][i, j] = np.mean(np.abs(est_vals[mask] - true_vals[mask]))
                else:
                    quartile_maes[f'q{q}'][i, j] = np.nan
    return quartile_maes


# Compute all metrics for each algorithm
maes_by_kl = {}
median_aes_by_kl = {}
spearman_by_kl = {}
trimmed_mae_iqr_by_kl = {}
stratified_mae_by_kl = {}  # nested: {alg: {q1: ..., q2: ..., q3: ..., q4: ...}}

for alg_name, est_ldrs_arr in est_ldrs_by_alg.items():
    # MAE (original)
    absolute_errors = np.abs(est_ldrs_arr - true_ldrs_arr)
    maes = reduce(absolute_errors, 'n t d -> n t', 'mean')
    maes_by_kl[alg_name] = maes.reshape(len(KL_DIVERGENCES), NUM_INSTANCES_PER_KL, NTEST_SETS)

    # Median AE
    median_aes = compute_median_ae(est_ldrs_arr, true_ldrs_arr)
    median_aes_by_kl[alg_name] = median_aes.reshape(len(KL_DIVERGENCES), NUM_INSTANCES_PER_KL, NTEST_SETS)

    # Spearman correlation
    spearman = compute_spearman_correlation(est_ldrs_arr, true_ldrs_arr)
    spearman_by_kl[alg_name] = spearman.reshape(len(KL_DIVERGENCES), NUM_INSTANCES_PER_KL, NTEST_SETS)

    # Trimmed MAE within IQR
    trimmed_mae = compute_trimmed_mae_iqr(est_ldrs_arr, true_ldrs_arr)
    trimmed_mae_iqr_by_kl[alg_name] = trimmed_mae.reshape(len(KL_DIVERGENCES), NUM_INSTANCES_PER_KL, NTEST_SETS)

    # Stratified MAE by quartiles
    strat_maes = compute_stratified_mae_quartiles(est_ldrs_arr, true_ldrs_arr)
    stratified_mae_by_kl[alg_name] = {
        q: arr.reshape(len(KL_DIVERGENCES), NUM_INSTANCES_PER_KL, NTEST_SETS)
        for q, arr in strat_maes.items()
    }

# save results
os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)
with h5py.File(processed_results_filename, 'w') as f:
    for alg_name, maes_arr in maes_by_kl.items():
        f.create_dataset(f'maes_by_kl_{alg_name}', data=maes_arr)
    for alg_name, median_aes_arr in median_aes_by_kl.items():
        f.create_dataset(f'median_aes_by_kl_{alg_name}', data=median_aes_arr)
    for alg_name, spearman_arr in spearman_by_kl.items():
        f.create_dataset(f'spearman_by_kl_{alg_name}', data=spearman_arr)
    for alg_name, trimmed_mae_arr in trimmed_mae_iqr_by_kl.items():
        f.create_dataset(f'trimmed_mae_iqr_by_kl_{alg_name}', data=trimmed_mae_arr)
    for alg_name, strat_dict in stratified_mae_by_kl.items():
        for quartile, arr in strat_dict.items():
            f.create_dataset(f'stratified_mae_{quartile}_by_kl_{alg_name}', data=arr)
