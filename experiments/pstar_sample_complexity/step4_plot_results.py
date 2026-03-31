"""
Generate publication-quality visualization comparing MAE (Mean Absolute Error)
across sample complexity values for three triangular density ratio estimation
methods. Plots sample complexity curve: MAE vs n_pstar with error bands.

Data flow:
- Input: processed_results/metrics.h5 (aggregated metrics from step3)
- Output: Single PDF figure in figures/ directory
"""

import os
import h5py
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Configuration Loading

config = yaml.load(open('experiments/pstar_sample_complexity/config.yaml', 'r'), Loader=yaml.FullLoader)

PROCESSED_RESULTS_DIR = config['processed_results_dir']
FIGURES_DIR = config['figures_dir']
ALGORITHMS = config['algorithms']

print(f"Loaded configuration:")
print(f"  - processed_results_dir: {PROCESSED_RESULTS_DIR}")
print(f"  - figures_dir: {FIGURES_DIR}")
print(f"  - algorithms: {ALGORITHMS}")


# 2. Style and Color Configuration

COLORS = {
    'TriangularMDRE': '#1f77b4',
    'TriangularTDRE': '#ff7f0e',
    'MultiHeadTriangularTDRE': '#2ca02c',
    'TSM': '#d62728',
}

MARKERS = {
    'TriangularMDRE': 'o',
    'TriangularTDRE': 's',
    'MultiHeadTriangularTDRE': '^',
    'TSM': 'D',
}

LINEWIDTH = 1.5
MARKERSIZE = 6

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 10


# 3. File Path Definition and Validation

processed_results_filename = f'{PROCESSED_RESULTS_DIR}/metrics.h5'

if not os.path.exists(processed_results_filename):
    print(f"Error: {processed_results_filename} not found.")
    print("Please run step3_process_results.py first.")
    exit(1)


# 4. Load Metrics from HDF5

with h5py.File(processed_results_filename, 'r') as f:
    nsamples_pstar_values = f['nsamples_pstar_values'][:]

    mae_by_alg = {}
    mae_std_by_alg = {}

    for alg in ALGORITHMS:
        mae_key = f'mae_{alg}'
        mae_std_key = f'mae_std_{alg}'

        if mae_key in f:
            mae_by_alg[alg] = f[mae_key][:]
        else:
            print(f"Warning: {mae_key} not found, skipping {alg}")
            continue

        if mae_std_key in f:
            mae_std_by_alg[alg] = f[mae_std_key][:]
        else:
            mae_std_by_alg[alg] = np.zeros_like(mae_by_alg[alg])


# 5. Data Validation

if not mae_by_alg:
    print("Error: No algorithms loaded successfully.")
    exit(1)

for alg in mae_by_alg:
    assert len(mae_by_alg[alg]) == len(nsamples_pstar_values), \
        f"Algorithm {alg}: mae array length {len(mae_by_alg[alg])} != nsamples_pstar length {len(nsamples_pstar_values)}"
    assert len(mae_std_by_alg[alg]) == len(nsamples_pstar_values), \
        f"Algorithm {alg}: mae_std array length {len(mae_std_by_alg[alg])} != nsamples_pstar length {len(nsamples_pstar_values)}"

print(f"Loaded metrics from {processed_results_filename}")
print(f"  - nsamples_pstar_values: {nsamples_pstar_values.tolist()}")
print(f"  - algorithms: {list(mae_by_alg.keys())}")


# 6. Create Output Directory

os.makedirs(FIGURES_DIR, exist_ok=True)


# 7. Plot Function

def plot_mae_vs_nsamples_pstar(nsamples, mae_by_alg, mae_std_by_alg, output_dir, output_name):
    """plot MAE vs n_pstar with error bands for each algorithm.

    args:
        nsamples: (7,) array of sample sizes
        mae_by_alg: dict mapping algorithm_name -> (7,) array of MAE values
        mae_std_by_alg: dict mapping algorithm_name -> (7,) array of MAE std deviations
        output_dir: directory to save PDF
        output_name: filename without extension (e.g., 'mae_vs_nsamples_pstar')

    returns:
        None; saves figure to {output_dir}/{output_name}.pdf

    procedure:
        1. create figure with plt.subplots(figsize=(6, 4))
        2. for each algorithm in sorted order:
           a. retrieve mae values and std from dictionaries
           b. plot line: ax.plot(nsamples, mae, label=alg, color, marker, linewidth, markersize)
           c. plot error band: ax.fill_between(nsamples, mae-std, mae+std, alpha=0.2, color)
        3. set x/y scales: ax.set_xscale('log'), ax.set_yscale('log')
        4. set labels: xlabel=r'$n_{p^*}$', ylabel='MAE'
        5. add legend and call plt.tight_layout()
        6. save to {output_dir}/{output_name}.pdf with dpi=150, bbox_inches='tight'
        7. close figure and print save path
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # sort algorithms for consistent order
    algs_sorted = sorted([alg for alg in mae_by_alg.keys() if alg in COLORS])

    for alg in algs_sorted:
        mae = mae_by_alg[alg]
        mae_std = mae_std_by_alg[alg]

        # skip if all nan
        if np.all(np.isnan(mae)):
            print(f"Warning: {alg} has all-NaN MAE values, skipping")
            continue

        # plot line
        ax.plot(
            nsamples, mae,
            label=alg, color=COLORS[alg], marker=MARKERS[alg],
            linewidth=LINEWIDTH, markersize=MARKERSIZE
        )

        # plot error band
        ax.fill_between(
            nsamples, mae - mae_std, mae + mae_std,
            alpha=0.2, color=COLORS[alg]
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$n_{p^*}$')
    ax.set_ylabel('MAE')
    ax.legend()

    plt.tight_layout()
    filepath = f'{output_dir}/{output_name}.pdf'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved {filepath}")


# 8. Generate Figure

plot_mae_vs_nsamples_pstar(
    nsamples=nsamples_pstar_values,
    mae_by_alg=mae_by_alg,
    mae_std_by_alg=mae_std_by_alg,
    output_dir=FIGURES_DIR,
    output_name='mae_vs_nsamples_pstar'
)


# 9. Summary Report

print("\n" + "="*80)
print("Sample complexity figure saved:")
print(f"{FIGURES_DIR}/mae_vs_nsamples_pstar.pdf")
print("="*80)
