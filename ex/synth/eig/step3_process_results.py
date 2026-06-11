"""reduce raw eig estimates to per-method per-beta normalized regret.

pipeline (per method m):
  1. per cell c = (prior p, beta b, design idx d): err[m, c] = |est_eig - true_eig|.
  2. per cell c: regret[m, c] = (err[m, c] - min_m' err[m', c]) / (max_m' - min_m').
     ties (max == min) collapse to 0; non-finite err for a method maps to NaN.
  3. reshape into (P, B, D) with the construction order from step1
     (prior outer / beta middle / design inner).
  4. median across priors per (beta, design): regret_pd[m, b, d] -> (B, D).
  5. median across designs per beta: med_of_med[m, b] -> (B,).
  6. bootstrap 1000 times by resampling prior indices (size P, with replacement)
     and design indices (size D, with replacement) independently; recompute
     median-of-medians per resample. IQR endpoints stored as lo/hi.

written datasets (per method m):
  regret_by_beta_<m>   shape (B,)  -- point estimate of median-of-medians
  regret_lo_by_beta_<m> shape (B,) -- 25th percentile of bootstrap samples
  regret_hi_by_beta_<m> shape (B,) -- 75th percentile of bootstrap samples
plus the shared `design_eig_percentages` axis.
"""
import os

import h5py
import numpy as np
import yaml


config = yaml.load(open('ex/synth/eig/config1.yaml', 'r'), Loader=yaml.FullLoader)
DATA_DIR = os.path.expandvars(config['data_dir'])
RAW_RESULTS_DIR = config['raw_results_dir']
PROCESSED_RESULTS_DIR = config['processed_results_dir']
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']
NUM_PRIORS = config['num_priors']
NUM_DESIGNS_PER_SETTING = config['num_designs_per_setting']
DESIGN_EIG_PERCENTAGES = config['design_eig_percentages']
SEED = config['seed']

N_BOOTSTRAP = 1000

P = NUM_PRIORS
B = len(DESIGN_EIG_PERCENTAGES)
D = NUM_DESIGNS_PER_SETTING

dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM},nsamples={NSAMPLES}.h5'
raw_results_filename = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM},nsamples={NSAMPLES}.h5'
processed_results_filename = f'{PROCESSED_RESULTS_DIR}/regret_by_beta_d={DATA_DIM},nsamples={NSAMPLES}.h5'


with h5py.File(dataset_filename, 'r') as dataset_file:
    design_eig_percentages = dataset_file['design_eig_percentage_arr'][:].squeeze()

with h5py.File(raw_results_filename, 'r') as results_file:
    true_eigs = results_file['true_eigs_arr'][:]
    est_keys = [key for key in results_file.keys() if key.startswith('est_eigs_arr_')]
    est_eigs_by_alg = {key.replace('est_eigs_arr_', ''): results_file[key][:].squeeze(-1)
                       for key in est_keys}

# sanity: construction order from step1 is prior-outer, beta-middle, design-inner.
expected_nrows = P * B * D
if true_eigs.shape[0] != expected_nrows:
    raise ValueError(
        f'true_eigs has {true_eigs.shape[0]} rows; expected P*B*D = {expected_nrows}.'
    )
betas_grid = np.array(DESIGN_EIG_PERCENTAGES, dtype=np.float32)
beta_axis_check = np.tile(np.repeat(betas_grid, D), P)
if not np.allclose(design_eig_percentages, beta_axis_check):
    raise ValueError('design_eig_percentage_arr ordering does not match (prior, beta, design).')


method_names = list(est_eigs_by_alg.keys())
M = len(method_names)

# stack per-cell |est - true| into (M, P, B, D).
err = np.full((M, P, B, D), np.nan, dtype=np.float64)
for m_idx, name in enumerate(method_names):
    cell_err = np.abs(est_eigs_by_alg[name].astype(np.float64) - true_eigs.astype(np.float64))
    cell_err[~np.isfinite(cell_err)] = np.nan
    err[m_idx] = cell_err.reshape(P, B, D)

# per-cell best / worst across methods -> normalized regret.
finite_mask = np.isfinite(err)
err_for_min = np.where(finite_mask, err, np.inf)
err_for_max = np.where(finite_mask, err, -np.inf)
best = err_for_min.min(axis=0)                       # (P, B, D)
worst = err_for_max.max(axis=0)                      # (P, B, D)
span = worst - best
denom = np.where(span > 0, span, np.nan)
regret = (err - best[None]) / denom[None]            # (M, P, B, D); nan if tie
# explicit zero where all methods tied (best == worst) and method has finite err.
tied = (span == 0) & np.isfinite(best)
regret = np.where(np.broadcast_to(tied[None], regret.shape) & finite_mask, 0.0, regret)


def median_of_medians(mat: np.ndarray) -> np.ndarray:
    """mat shape (..., P, D) -> scalar (or batched) median-of-medians.

    inner median over priors (axis -2) gives per-design median; outer median
    over designs (axis -1) gives the reported statistic.
    """
    per_design = np.nanmedian(mat, axis=-2)
    return np.nanmedian(per_design, axis=-1)


# point estimate per (method, beta).
point = np.empty((M, B), dtype=np.float64)
for m_idx in range(M):
    for b_idx in range(B):
        point[m_idx, b_idx] = median_of_medians(regret[m_idx, :, b_idx, :])


# bootstrap: resample prior indices (size P) and design indices (size D) jointly.
rng = np.random.default_rng(SEED)
prior_draws = rng.integers(0, P, size=(N_BOOTSTRAP, P))    # (N_BOOTSTRAP, P)
design_draws = rng.integers(0, D, size=(N_BOOTSTRAP, D))   # (N_BOOTSTRAP, D)

lo = np.empty((M, B), dtype=np.float64)
hi = np.empty((M, B), dtype=np.float64)
for m_idx in range(M):
    for b_idx in range(B):
        mat = regret[m_idx, :, b_idx, :]                   # (P, D)
        boots = np.empty(N_BOOTSTRAP, dtype=np.float64)
        for k in range(N_BOOTSTRAP):
            resampled = mat[prior_draws[k][:, None], design_draws[k][None, :]]
            boots[k] = median_of_medians(resampled)
        finite_boots = boots[np.isfinite(boots)]
        if finite_boots.size == 0:
            lo[m_idx, b_idx] = np.nan
            hi[m_idx, b_idx] = np.nan
        else:
            lo[m_idx, b_idx] = np.percentile(finite_boots, 25)
            hi[m_idx, b_idx] = np.percentile(finite_boots, 75)


os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)
with h5py.File(processed_results_filename, 'w') as out_file:
    out_file.create_dataset('design_eig_percentages', data=betas_grid)
    for m_idx, name in enumerate(method_names):
        out_file.create_dataset(f'regret_by_beta_{name}',    data=point[m_idx].astype(np.float32))
        out_file.create_dataset(f'regret_lo_by_beta_{name}', data=lo[m_idx].astype(np.float32))
        out_file.create_dataset(f'regret_hi_by_beta_{name}', data=hi[m_idx].astype(np.float32))

print(f'wrote {processed_results_filename}')
print(f'  methods: {", ".join(method_names)}')
print(f'  betas:   {list(betas_grid)}')
