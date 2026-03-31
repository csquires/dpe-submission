"""
Output markdown table for hidden dimension scaling experiment.
Reports MAE with standard deviation error bars across hidden dimensions.

Data flow:
- Input: processed_results/metrics.h5
- Output: markdown table to stdout
"""

import os
import h5py
import numpy as np
import yaml


config = yaml.load(
    open('experiments/hidden_dim_scaling/config.yaml', 'r'),
    Loader=yaml.FullLoader
)

PROCESSED_RESULTS_DIR = config['processed_results_dir']
HIDDEN_DIMS = config['hidden_dims']
ALGORITHMS = config.get('algorithms', ['TSM', 'TriangularMDRE'])

processed_results_filename = f'{PROCESSED_RESULTS_DIR}/metrics.h5'

if not os.path.exists(processed_results_filename):
    print(f"Error: {processed_results_filename} not found.")
    print("Please run step3_process_results.py first.")
    exit(1)

# load metrics
with h5py.File(processed_results_filename, 'r') as f:
    hidden_dims = f['hidden_dims'][:]

    mae_by_alg = {}
    std_by_alg = {}

    for alg in ALGORITHMS:
        mae_key = f'mae_{alg}'
        std_key = f'std_{alg}'

        if mae_key in f:
            mae_by_alg[alg] = f[mae_key][:]
        else:
            print(f"Warning: {mae_key} not found, skipping {alg}")
            continue

        if std_key in f:
            std_by_alg[alg] = f[std_key][:]
        else:
            std_by_alg[alg] = np.zeros_like(mae_by_alg[alg])


def fmt(val, err):
    """format value with error bar using latex \\pm."""
    if np.isnan(val):
        return "---"
    return f"${val:.4f} \\pm {err:.4f}$"


# build table
algs = sorted(mae_by_alg.keys())
header = ["Hidden Dim"] + algs
print("| " + " | ".join(header) + " |")
print("|" + "|".join(["---"] * len(header)) + "|")

for i, hd in enumerate(hidden_dims):
    row = [f"{int(hd)}"]
    for alg in algs:
        row.append(fmt(mae_by_alg[alg][i], std_by_alg[alg][i]))
    print("| " + " | ".join(row) + " |")
