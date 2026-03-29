"""
Generate publication-quality visualizations comparing TSM vs TriangularMDRE
across hidden dimensions. Plots metrics including MAE, parameter count,
training time, and peak GPU memory to characterize scalability and accuracy
tradeoffs.

Data flow:
- Input: processed_results/metrics.h5 (aggregated metrics from step3)
- Output: Five PDF figures in figures/ directory
"""

import os
import h5py
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Configuration Loading

config = yaml.load(open('experiments/hidden_dim_scaling/config.yaml', 'r'), Loader=yaml.FullLoader)

PROCESSED_RESULTS_DIR = config['processed_results_dir']
FIGURES_DIR = config['figures_dir']

HIDDEN_DIMS = config['hidden_dims']
ALGORITHMS = config.get('algorithms', ['TSM', 'TriangularMDRE'])


# 2. Style and Color Configuration

COLORS = {
    'TSM': '#d62728',
    'TriangularMDRE': '#aec7e8',
}

MARKERS = {
    'TSM': 'o',
    'TriangularMDRE': 's',
}

LINEWIDTH = 1.5
MARKERSIZE = 6

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (5, 3.5)
plt.rcParams['font.size'] = 10


# 3. File Path Definition and Validation

processed_results_filename = f'{PROCESSED_RESULTS_DIR}/metrics.h5'

if not os.path.exists(processed_results_filename):
    print(f"Error: {processed_results_filename} not found.")
    print("Please run step3_process_results.py first.")
    exit(1)


# 4. Load Metrics from HDF5

with h5py.File(processed_results_filename, 'r') as f:
    hidden_dims_loaded = f['hidden_dims'][:]

    mae_by_alg = {}
    std_by_alg = {}
    timing_mean_by_alg = {}
    timing_std_by_alg = {}
    peak_memory_by_alg = {}
    param_count_by_alg = {}

    for alg in ALGORITHMS:
        mae_key = f'mae_{alg}'
        std_key = f'std_{alg}'
        timing_mean_key = f'timing_mean_{alg}'
        timing_std_key = f'timing_std_{alg}'
        peak_memory_key = f'peak_memory_{alg}'
        param_count_key = f'param_count_{alg}'

        if mae_key in f:
            mae_by_alg[alg] = f[mae_key][:]
        else:
            print(f"Warning: {mae_key} not found, skipping {alg}")
            continue

        if std_key in f:
            std_by_alg[alg] = f[std_key][:]
        else:
            std_by_alg[alg] = np.zeros_like(mae_by_alg[alg])

        if timing_mean_key in f:
            timing_mean_by_alg[alg] = f[timing_mean_key][:]
        else:
            timing_mean_by_alg[alg] = np.full_like(mae_by_alg[alg], np.nan)

        if timing_std_key in f:
            timing_std_by_alg[alg] = f[timing_std_key][:]
        else:
            timing_std_by_alg[alg] = np.zeros_like(mae_by_alg[alg])

        if peak_memory_key in f:
            peak_memory_by_alg[alg] = f[peak_memory_key][:]
        else:
            peak_memory_by_alg[alg] = np.full_like(mae_by_alg[alg], np.nan)

        if param_count_key in f:
            param_count_by_alg[alg] = f[param_count_key][:]
        else:
            param_count_by_alg[alg] = np.full_like(mae_by_alg[alg], np.nan)


# 5. Validation and Logging

if not mae_by_alg:
    print("Error: No algorithms loaded successfully.")
    exit(1)

for alg in mae_by_alg:
    assert len(mae_by_alg[alg]) == len(hidden_dims_loaded), \
        f"Algorithm {alg}: mae array length {len(mae_by_alg[alg])} != hidden_dims length {len(hidden_dims_loaded)}"

print(f"Loaded metrics from {processed_results_filename}")
print(f"  - hidden_dims: {hidden_dims_loaded.tolist()}")
print(f"  - algorithms: {list(mae_by_alg.keys())}")


# 6. Create Output Directory

os.makedirs(FIGURES_DIR, exist_ok=True)


# 7. Generic Plot Function

def plot_line_with_errorbars(x_data, y_data_by_alg, y_err_by_alg, x_label, y_label,
                              figure_name, use_log_x=False, use_log_y=False):
    """plot line with error bars for each algorithm.

    args:
        x_data: (n,) array of x-axis values (e.g., hidden_dims)
        y_data_by_alg: dict mapping alg_name -> (n,) array of y-values
        y_err_by_alg: dict mapping alg_name -> (n,) array of error bars (or None)
        x_label: string for x-axis label
        y_label: string for y-axis label
        figure_name: output filename without extension (e.g., 'mae_vs_hidden_dim')
        use_log_x: bool, whether to use log scale on x-axis
        use_log_y: bool, whether to use log scale on y-axis

    returns:
        None; saves figure to {FIGURES_DIR}/{figure_name}.pdf

    procedure:
        1. create figure with plt.subplots()
        2. for each algorithm in sorted order:
           a. retrieve y_data and y_err (if provided)
           b. plot with ax.errorbar() or ax.plot()
           c. apply algorithm color, marker, linewidth from COLORS/MARKERS dicts
        3. set x/y labels, title, legend
        4. apply log scales if requested
        5. call plt.tight_layout()
        6. save to {FIGURES_DIR}/{figure_name}.pdf with dpi=150, bbox_inches='tight'
        7. close figure
    """
    fig, ax = plt.subplots()

    # sort algorithms for consistent order across plots
    algs_sorted = sorted([alg for alg in y_data_by_alg.keys() if alg in COLORS])

    for alg in algs_sorted:
        y_data = y_data_by_alg[alg]
        y_err = y_err_by_alg.get(alg) if y_err_by_alg else None

        # skip if all nan
        if np.all(np.isnan(y_data)):
            print(f"Warning: {alg} has all-NaN y-values, skipping")
            continue

        # determine capsize based on whether we have error bars
        capsize = 3 if y_err is not None else 0

        if y_err is not None:
            ax.errorbar(
                x_data, y_data, yerr=y_err,
                label=alg, color=COLORS[alg], marker=MARKERS[alg],
                linewidth=LINEWIDTH, markersize=MARKERSIZE, capsize=capsize
            )
        else:
            ax.plot(
                x_data, y_data,
                label=alg, color=COLORS[alg], marker=MARKERS[alg],
                linewidth=LINEWIDTH, markersize=MARKERSIZE
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    if use_log_x:
        ax.set_xscale('log')
    if use_log_y:
        ax.set_yscale('log')

    plt.tight_layout()
    filepath = f'{FIGURES_DIR}/{figure_name}.pdf'
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved {filepath}")


# 8. Figure 1: MAE vs Hidden Dimension

plot_line_with_errorbars(
    x_data=hidden_dims_loaded,
    y_data_by_alg=mae_by_alg,
    y_err_by_alg=std_by_alg,
    x_label='Hidden Dimension',
    y_label='Mean Absolute Error',
    figure_name='mae_vs_hidden_dim',
    use_log_x=False,
    use_log_y=False
)


# 9. Figure 2: MAE vs Parameter Count (log-x)

plot_line_with_errorbars(
    x_data=param_count_by_alg['TSM'],
    y_data_by_alg={alg: mae_by_alg[alg] for alg in mae_by_alg},
    y_err_by_alg={alg: std_by_alg[alg] for alg in std_by_alg},
    x_label='Parameter Count',
    y_label='Mean Absolute Error',
    figure_name='mae_vs_param_count',
    use_log_x=True,
    use_log_y=False
)


# 10. Figure 3: Parameter Count vs Hidden Dimension

plot_line_with_errorbars(
    x_data=hidden_dims_loaded,
    y_data_by_alg={alg: param_count_by_alg[alg] for alg in param_count_by_alg},
    y_err_by_alg=None,
    x_label='Hidden Dimension',
    y_label='Parameter Count',
    figure_name='param_count_vs_hidden_dim',
    use_log_x=False,
    use_log_y=False
)


# 11. Figure 4: Training Time vs Hidden Dimension

plot_line_with_errorbars(
    x_data=hidden_dims_loaded,
    y_data_by_alg=timing_mean_by_alg,
    y_err_by_alg=timing_std_by_alg,
    x_label='Hidden Dimension',
    y_label='Training Time (seconds)',
    figure_name='timing_vs_hidden_dim',
    use_log_x=False,
    use_log_y=False
)


# 12. Figure 5: Peak GPU Memory vs Hidden Dimension

peak_memory_MB_by_alg = {alg: peak_memory_by_alg[alg] / (1024**2)
                         for alg in peak_memory_by_alg}

plot_line_with_errorbars(
    x_data=hidden_dims_loaded,
    y_data_by_alg=peak_memory_MB_by_alg,
    y_err_by_alg=None,
    x_label='Hidden Dimension',
    y_label='Peak GPU Memory (MB)',
    figure_name='memory_vs_hidden_dim',
    use_log_x=False,
    use_log_y=False
)


# 13. Summary Report

print("\n" + "="*80)
print("Publication-quality figures saved:")
print(f"  1. {FIGURES_DIR}/mae_vs_hidden_dim.pdf")
print(f"  2. {FIGURES_DIR}/mae_vs_param_count.pdf")
print(f"  3. {FIGURES_DIR}/param_count_vs_hidden_dim.pdf")
print(f"  4. {FIGURES_DIR}/timing_vs_hidden_dim.pdf")
print(f"  5. {FIGURES_DIR}/memory_vs_hidden_dim.pdf")
print("="*80)
