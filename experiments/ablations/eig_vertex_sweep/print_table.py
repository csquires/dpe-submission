"""
Output markdown table for EIG vertex sweep experiment.
Computes MAE with standard deviation error bars across beta values.

Data flow:
- Input: raw_results/*.h5 (per-design estimates and true values)
- Output: markdown table to stdout
"""

import h5py
import numpy as np
import yaml


config = yaml.load(
    open('experiments/eig_vertex_sweep/config.yaml', 'r'),
    Loader=yaml.FullLoader
)

DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']
DESIGN_EIG_PERCENTAGES = config['design_eig_percentages']
VERTEX_WAYPOINTS = config['vertex_waypoints']

dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM},nsamples={NSAMPLES}.h5'
raw_results_filename = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM},nsamples={NSAMPLES}.h5'

# load data
with h5py.File(dataset_filename, 'r') as f:
    design_eig_percentages = f['design_eig_percentage_arr'][:].squeeze()

with h5py.File(raw_results_filename, 'r') as f:
    true_eigs = f['true_eigs_arr'][:]
    est_keys = [k for k in f.keys() if k.startswith('est_eigs_arr_vertex_')]
    est_eigs_by_vertex = {
        k.replace('est_eigs_arr_', ''): f[k][:] for k in est_keys
    }

# compute mae and std per beta
mae_by_vertex = {}
std_by_vertex = {}

for vertex_name, est_eigs in est_eigs_by_vertex.items():
    abs_errors = np.abs(est_eigs - true_eigs)
    beta_maes = []
    beta_stds = []

    for beta in DESIGN_EIG_PERCENTAGES:
        mask = np.isclose(design_eig_percentages, beta)
        errors = abs_errors[mask]
        beta_maes.append(errors.mean())
        beta_stds.append(errors.std())

    mae_by_vertex[vertex_name] = np.array(beta_maes)
    std_by_vertex[vertex_name] = np.array(beta_stds)


def fmt(val, err):
    """format value with error bar using latex \\pm."""
    return f"${val:.4f} \\pm {err:.4f}$"


# build table
vertex_names = sorted(mae_by_vertex.keys(), key=lambda x: int(x.split('_')[1]))
header = ["$\\beta$"] + [f"vertex={v.split('_')[1]}" for v in vertex_names]
print("| " + " | ".join(header) + " |")
print("|" + "|".join(["---"] * len(header)) + "|")

for i, beta in enumerate(DESIGN_EIG_PERCENTAGES):
    row = [f"{beta:.3f}"]
    for v in vertex_names:
        row.append(fmt(mae_by_vertex[v][i], std_by_vertex[v][i]))
    print("| " + " | ".join(row) + " |")
