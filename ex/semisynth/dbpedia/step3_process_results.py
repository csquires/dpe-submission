"""thin driver for step3: compute 3 uniform metrics on gathered results.

loads config, reconstructs full-grid cell order (AUTHORITATIVE decode, not
step2_pool), builds cell_data_paths, calls compute_metrics, prints per-method
summary table (nanmean over cells).
"""
import os
import h5py
import yaml
import numpy as np
from ex.utils.semisynth_metrics import compute_metrics

CONFIG_PATH = 'ex/semisynth/dbpedia/config.yaml'


def main():
    """load config; build cell paths via full-grid decode; call compute_metrics; print table."""
    # load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = os.path.expandvars(config['data_dir'])
    raw_results_dir = config['raw_results_dir']
    processed_results_dir = config['processed_results_dir']
    seed = config['seed']

    # open gathered to get n_cells from est_ldrs shape
    gathered_h5 = os.path.join(raw_results_dir, 'results_all_cells.h5')
    with h5py.File(gathered_h5, 'r') as f:
        # find first est_ldrs_* key to get n_cells
        est_keys = [k for k in f.keys() if k.startswith('est_ldrs_')]
        if not est_keys:
            raise ValueError(f'no est_ldrs_* datasets found in {gathered_h5}')
        n_cells = f[est_keys[0]].shape[0]

    # full-grid decode: cell i -> alpha_{i//n_pairs}_pair_{i%n_pairs}
    n_alphas = len(config['alphas'])
    n_pairs = n_cells // n_alphas

    cell_data_paths = []
    for i in range(n_cells):
        alpha_idx = i // n_pairs
        pair_idx = i % n_pairs
        path = os.path.join(data_dir, f'alpha_{alpha_idx}_pair_{pair_idx}.h5')
        cell_data_paths.append(path)

    # ensure output dir exists
    os.makedirs(processed_results_dir, exist_ok=True)

    # compute metrics
    out_path = os.path.join(processed_results_dir, 'summary.h5')
    methods = compute_metrics(gathered_h5, cell_data_paths, out_path, seed=seed)

    # print per-method summary table
    print(f'\n{"method":<30s}  {"eldr_abs_err":<15s}  {"mae_train":<15s}  {"regret":<10s}')
    print('-' * 75)

    with h5py.File(out_path, 'r') as f:
        for method in methods:
            eldr = np.nanmean(f[f'eldr_abs_err_{method}'][:])
            mae = np.nanmean(f[f'mae_train_{method}'][:])
            regret = np.nanmean(f[f'regret_{method}'][:])
            print(f'{method:<30s}  {eldr:<15.4e}  {mae:<15.4e}  {regret:<10.4f}')


if __name__ == '__main__':
    main()
