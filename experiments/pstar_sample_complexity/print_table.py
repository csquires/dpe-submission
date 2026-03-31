"""
Output markdown table for pstar sample complexity experiment.
Reports MAE with standard deviation error bars across nsamples_pstar values.

Data flow:
- Input: processed_results/metrics.h5
- Output: markdown table to stdout
"""

import os
import h5py
import numpy as np
import yaml


config = yaml.load(
    open('experiments/pstar_sample_complexity/config.yaml', 'r'),
    Loader=yaml.FullLoader
)

PROCESSED_RESULTS_DIR = config['processed_results_dir']
NSAMPLES_PSTAR_VALUES = config['nsamples_pstar_values']
ALGORITHMS = config.get('algorithms', [])

processed_results_filename = f'{PROCESSED_RESULTS_DIR}/metrics.h5'

if not os.path.exists(processed_results_filename):
    print(f"Error: {processed_results_filename} not found.")
    print("Please run step3_process_results.py first.")
    exit(1)

# load metrics
with h5py.File(processed_results_filename, 'r') as f:
    nsamples_pstar_values = f['nsamples_pstar_values'][:]

    mae_by_alg = {}
    std_by_alg = {}

    for alg in ALGORITHMS:
        mae_key = f'mae_{alg}'
        std_key = f'mae_std_{alg}'

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
header = ["n_pstar"] + algs
print("| " + " | ".join(header) + " |")
print("|" + "|".join(["---"] * len(header)) + "|")

for i, ns in enumerate(nsamples_pstar_values):
    row = [f"{int(ns)}"]
    for alg in algs:
        row.append(fmt(mae_by_alg[alg][i], std_by_alg[alg][i]))
    print("| " + " | ".join(row) + " |")
