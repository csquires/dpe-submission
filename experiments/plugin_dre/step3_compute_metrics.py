"""
Step 3: Compute Metrics for Plugin DRE Experiment

Computes per-point absolute errors and overall MAE for each algorithm.
"""
import os

import h5py
import numpy as np
import yaml


config = yaml.load(open('experiments/plugin_dre/config.yaml', 'r'), Loader=yaml.FullLoader)
# directories
DATA_DIR = config['data_dir']
RESULTS_DIR = config['results_dir']

dataset_filename = f'{DATA_DIR}/dataset.h5'
raw_results_filename = f'{RESULTS_DIR}/raw_results.h5'
metrics_filename = f'{RESULTS_DIR}/metrics.h5'

# Load true LDRs
with h5py.File(dataset_filename, 'r') as f:
    true_ldrs_grid = f['true_ldrs_grid_arr'][:]
    kl_divergences = f['kl_divergence_arr'][:]

# Discover algorithms from raw results
with h5py.File(raw_results_filename, 'r') as f:
    alg_names = [key.replace('est_ldrs_grid_', '') for key in f.keys() if key.startswith('est_ldrs_grid_') and 'TriangularTDRE' not in key]

print(f"Found algorithms: {alg_names}")
print(f"True LDRs shape: {true_ldrs_grid.shape}")  # (nrows, num_grid_points)

# Compute metrics for each algorithm
with h5py.File(metrics_filename, 'w') as metrics_file:
    # Store KL distances for reference
    metrics_file.create_dataset('kl_divergences', data=kl_divergences)

    with h5py.File(raw_results_filename, 'r') as results_file:
        for alg_name in alg_names:
            est_ldrs_grid = results_file[f'est_ldrs_grid_{alg_name}'][:]

            # Per-point absolute errors: shape (nrows, num_grid_points)
            abs_errors = np.abs(est_ldrs_grid - true_ldrs_grid)

            # Overall MAE per KL: shape (nrows,)
            mae_per_instance = np.mean(abs_errors, axis=1)

            # Store metrics
            metrics_file.create_dataset(f'abs_errors_{alg_name}', data=abs_errors)
            metrics_file.create_dataset(f'mae_{alg_name}', data=mae_per_instance)

            print(f"\n{alg_name}:")
            print(f"  Absolute errors shape: {abs_errors.shape}")
            print(f"  MAE by KL distance:")
            for i, kl in enumerate(kl_divergences):
                print(f"    KL={kl:.1f}: MAE={mae_per_instance[i]:.4f}")

print(f"\nMetrics saved to: {metrics_filename}")
