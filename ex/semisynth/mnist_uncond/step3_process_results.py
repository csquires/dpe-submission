"""thin driver for mnist_uncond step3: compute per-cell eldr_abs_err, mae_train, regret.

load config, open gathered results_all_cells.h5 to infer n_cells, reconstruct
cell_data_paths using the authoritative full-grid ordering, call compute_metrics,
print per-method summary table.
"""
import os
import h5py
import numpy as np
import yaml
from ex.utils.semisynth_metrics import compute_metrics

CONFIG_PATH = 'ex/semisynth/mnist_uncond/config.yaml'


if __name__ == '__main__':
    # load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    data_dir = os.path.expandvars(config['data_dir'])
    raw_results_dir = config['raw_results_dir']
    processed_results_dir = config['processed_results_dir']
    n_alphas = len(config['alphas'])

    # locate gathered results: config raw_results_dir first, else the in-repo
    # raw_results (where the step2 gather actually wrote for this experiment).
    gathered_h5 = os.path.join(raw_results_dir, 'results_all_cells.h5')
    if not os.path.exists(gathered_h5):
        gathered_h5 = 'ex/semisynth/mnist_uncond/raw_results/results_all_cells.h5'
    if not os.path.exists(gathered_h5):
        raise FileNotFoundError(f"gathered results not found: {gathered_h5}")

    # infer n_cells from gathered h5 shape
    with h5py.File(gathered_h5, 'r') as f:
        # find first est_ldrs_<method> key to get n_cells
        est_keys = [k for k in f.keys() if k.startswith('est_ldrs_')]
        if not est_keys:
            raise ValueError("no est_ldrs_* keys in gathered h5")
        n_cells = f[est_keys[0]].shape[0]

    # authoritative cell ordering: cell i -> alpha_{i//n_pairs}_pair_{i%n_pairs}
    n_pairs = n_cells // n_alphas

    cell_data_paths = []
    for i in range(n_cells):
        alpha_idx = i // n_pairs
        pair_idx = i % n_pairs
        path = os.path.join(data_dir, f'alpha_{alpha_idx}_pair_{pair_idx}.h5')
        cell_data_paths.append(path)

    # prepare output directory
    os.makedirs(processed_results_dir, exist_ok=True)

    # derive output path (authoritative: summary.h5, not processed_summary.h5)
    summary_h5 = os.path.join(processed_results_dir, 'summary.h5')

    # mnist_uncond est was evaluated on the training set (p*), so mae uses true_ldrs
    # (train GT), while eldr uses samples_test_true_ldrs (held-out test GT).
    methods = compute_metrics(gathered_h5, cell_data_paths, out_h5=summary_h5,
                              seed=config['seed'],
                              mae_gt_key='true_ldrs',
                              eldr_gt_key='samples_test_true_ldrs')

    # print summary table
    print("MNIST_UNCOND ELDR Step3: Per-Cell Metrics")
    print(f"n_cells={len(cell_data_paths)}, n_methods={len(methods)}")
    print(f"saved: {summary_h5}")

    with h5py.File(summary_h5, 'r') as f:
        for m in methods:
            eldr_key = f'eldr_abs_err_{m}'
            mae_key = f'mae_train_{m}'
            regret_key = f'regret_{m}'

            # read arrays; skip NaN when computing mean
            eldr_arr = np.array(f[eldr_key][:])
            mae_arr = np.array(f[mae_key][:])
            regret_arr = np.array(f[regret_key][:])

            eldr_mean = np.nanmean(eldr_arr)
            mae_mean = np.nanmean(mae_arr)
            regret_mean = np.nanmean(regret_arr)

            print(f"  {m:25s}: eldr={eldr_mean:.4g}, mae={mae_mean:.4g}, regret={regret_mean:.4g}")
