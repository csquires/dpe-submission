import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
# directories
PROCESSED_RESULTS_DIR = config['processed_results_dir']
FIGURES_DIR = config['figures_dir']
# dataset parameters
DATA_DIM = config['data_dim']
KL_DISTANCES = config['kl_distances']
NSAMPLES_TRAIN = config['nsamples_train']
NSAMPLES_TEST = config['nsamples_test']
NTEST_SETS = config['ntest_sets']

# filename = f'{PROCESSED_RESULTS_DIR}/metrics_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST},ntestsets={NTEST_SETS}.h5'
# filename = f'{PROCESSED_RESULTS_DIR}/maes_by_kl_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST},ntestsets={NTEST_SETS}.h5'
# filename = f'{PROCESSED_RESULTS_DIR}/added_cauchy_01.h5'
filename = f'{PROCESSED_RESULTS_DIR}/new_pstar.h5'

# Load all metrics from file
with h5py.File(filename, 'r') as f:
    maes_by_kl = {key.replace('maes_by_kl_', ''): f[key][:] for key in f.keys() if key.startswith('maes_by_kl_')}
    median_aes_by_kl = {key.replace('median_aes_by_kl_', ''): f[key][:] for key in f.keys() if key.startswith('median_aes_by_kl_')}
    spearman_by_kl = {key.replace('spearman_by_kl_', ''): f[key][:] for key in f.keys() if key.startswith('spearman_by_kl_')}
    trimmed_mae_iqr_by_kl = {key.replace('trimmed_mae_iqr_by_kl_', ''): f[key][:] for key in f.keys() if key.startswith('trimmed_mae_iqr_by_kl_')}
    # Stratified MAE by quartiles
    stratified_mae_by_kl = {}
    for key in f.keys():
        if key.startswith('stratified_mae_q'):
            # key format: stratified_mae_q1_by_kl_ALGNAME
            parts = key.split('_by_kl_')
            quartile = parts[0].replace('stratified_mae_', '')  # e.g., 'q1'
            alg_name = parts[1]
            if alg_name not in stratified_mae_by_kl:
                stratified_mae_by_kl[alg_name] = {}
            stratified_mae_by_kl[alg_name][quartile] = f[key][:]

# colors - consistent across all experiments
colors = {
    "BDRE": "#1f77b4",
    "MDRE": "#2ca02c",
    "TSM": "#d62728",
    "TriangularTSM": "#17becf",
    "TriangularTDRE": "#c3d922",
    "TriangularTDRE_Gauss": "#000000",
    "TriangularMDRE": "#aec7e8",
    "TriangularMDRE_Gauss": "#9edae5",
    "TDRE_5": "#ff7f0e",
    "TDRE_10": "#8c564b",  # default TDRE
    "TDRE_15": "#9467bd",
    "TDRE_20": "#e377c2",
    "TDRE_30": "#7f7f7f",
    "MDRE_5": "#17becf",
    "MDRE_10": "#7f7f7f",  # default MDRE
    "MDRE_15": "#2ca02c",
    "MDRE_15_Gauss": "#98df8a",
    "MDRE_20": "#8c564b",
    "MDRE_30": "#e377c2",
    "VFM": "#9467bd",
    # "Spatial": "#9467bd",
}

tdre_order = ["TDRE_5"]
mdre_order = ["MDRE_15"]

TEST_SET_TITLES = [r'$p_* = p_0$', r'$p_* = p_1$', r'$p_* = q_0$', r'$p_* = q_1$']


def get_algorithms_to_plot(data_dict):
    """Get list of algorithms in standard order: BDRE -> TDRE -> MDRE -> TSM -> TriangularMDRE -> VFM."""
    algs = []
    if "BDRE" in data_dict:
        algs.append(("BDRE", "BDRE"))
    if "TSM" in data_dict:
        algs.append(("TSM", "TSM"))
    if "TriangularMDRE" in data_dict:
        algs.append(("TriangularMDRE", "TriangularMDRE"))
    for tdre_name in tdre_order:
        if tdre_name in data_dict:
            algs.append((tdre_name, "TDRE"))
    for mdre_name in mdre_order:
        if mdre_name in data_dict:
            algs.append((mdre_name, "MDRE"))
    if "Spatial" in data_dict:
        algs.append(("Spatial", "VFM"))
    if "VFM" in data_dict:
        algs.append(("VFM", "VFM"))
    return algs


def plot_metric(data_by_kl, ylabel, figure_name, stats_file, use_log_y=True, higher_is_better=False):
    """Generic plotting function for a metric.

    Args:
        data_by_kl: dict mapping alg_name -> array of shape (n_kl, n_instances, n_test_sets)
        ylabel: Y-axis label
        figure_name: Output filename (without extension)
        stats_file: File handle to write stats to
        use_log_y: Whether to use log scale on y-axis
        higher_is_better: If True, don't use log scale (e.g., for correlation)
    """
    avg_by_kl = {alg: np.mean(arr, axis=1) for alg, arr in data_by_kl.items()}
    all_vals = [arr for arr in avg_by_kl.values()]
    if all_vals:
        all_mins = [arr.min() for arr in all_vals]
        all_maxs = [arr.max() for arr in all_vals]
        y_min = min(all_mins)
        y_max = max(all_maxs)
    else:
        y_min, y_max = 0.0, 1.0

    plt.clf()
    sns.set_style('whitegrid')
    plt.style.use('full-width.mplstyle')
    fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=NTEST_SETS)

    algorithms = get_algorithms_to_plot(avg_by_kl)

    # Write header to stats file
    stats_file.write(f"\n{'='*60}\n")
    stats_file.write(f"Figure: {figure_name}\n")
    stats_file.write(f"Metric: {ylabel}\n")
    stats_file.write(f"{'='*60}\n")
    stats_file.write(f"X-axis: KL(p0 || p1) = {KL_DISTANCES}\n\n")

    for i in range(NTEST_SETS):
        stats_file.write(f"Test Set {i} ({TEST_SET_TITLES[i]}):\n")
        for alg_key, alg_label in algorithms:
            if alg_key in avg_by_kl:
                y_vals = avg_by_kl[alg_key][:, i]
                axes[i].plot(KL_DISTANCES, y_vals, label=alg_label, color=colors.get(alg_key, "#333333"), linewidth=1.0)
                stats_file.write(f"  {alg_label} ({alg_key}): {y_vals.tolist()}\n")
        stats_file.write("\n")

        if use_log_y and not higher_is_better:
            axes[i].set_ylim(y_min * 0.9, y_max * 1.1)
            axes[i].set_yscale('log')
        else:
            # For correlation, set reasonable bounds
            if higher_is_better:
                axes[i].set_ylim(min(0, y_min - 0.1), 1.05)
            else:
                axes[i].set_ylim(y_min * 0.9, y_max * 1.1)
        axes[i].set_xscale('log')
        axes[i].set_xlabel(r'KL$(p_0 \| p_1)$')
        axes[i].set_title(TEST_SET_TITLES[i])

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    # Remove duplicate labels and enforce order
    by_label = dict(zip(labels, handles))
    legend_order = ["BDRE", "TDRE", "MDRE", "TSM", "TriangularMDRE", "VFM"]
    ordered_labels = [lbl for lbl in legend_order if lbl in by_label]
    ordered_labels += [lbl for lbl in by_label.keys() if lbl not in ordered_labels]
    plt.legend([by_label[lbl] for lbl in ordered_labels], ordered_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/{figure_name}.pdf')


def plot_stratified_mae(stratified_data, stats_file):
    """Plot stratified MAE with one subplot per quartile."""
    quartiles = ['q1', 'q2', 'q3', 'q4']
    quartile_labels = ['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)']

    # For each test set, create a figure with 4 subplots (one per quartile)
    for test_idx in range(NTEST_SETS):
        plt.clf()
        sns.set_style('whitegrid')
        plt.style.use('full-width.mplstyle')
        fig, axes = plt.subplots(figsize=(12, 3), nrows=1, ncols=4)

        algorithms = get_algorithms_to_plot(stratified_data)

        stats_file.write(f"\n{'='*60}\n")
        stats_file.write(f"Figure: stratified_mae_test{test_idx}\n")
        stats_file.write(f"Test Set: {TEST_SET_TITLES[test_idx]}\n")
        stats_file.write(f"{'='*60}\n")
        stats_file.write(f"X-axis: KL(p0 || p1) = {KL_DISTANCES}\n\n")

        all_vals = []
        for alg_key, _ in algorithms:
            if alg_key in stratified_data:
                for q in quartiles:
                    arr = stratified_data[alg_key].get(q)
                    if arr is not None:
                        avg = np.mean(arr, axis=1)[:, test_idx]
                        all_vals.extend(avg[~np.isnan(avg)])

        if all_vals:
            y_min, y_max = min(all_vals), max(all_vals)
        else:
            y_min, y_max = 0.01, 10.0

        for q_idx, (q, q_label) in enumerate(zip(quartiles, quartile_labels)):
            stats_file.write(f"Quartile {q_label}:\n")
            for alg_key, alg_label in algorithms:
                if alg_key in stratified_data and q in stratified_data[alg_key]:
                    arr = stratified_data[alg_key][q]
                    avg = np.mean(arr, axis=1)[:, test_idx]
                    axes[q_idx].plot(KL_DISTANCES, avg, label=alg_label, color=colors.get(alg_key, "#333333"), linewidth=1.0)
                    stats_file.write(f"  {alg_label} ({alg_key}): {avg.tolist()}\n")

            axes[q_idx].set_xscale('log')
            axes[q_idx].set_yscale('log')
            axes[q_idx].set_ylim(y_min * 0.9, y_max * 1.1)
            axes[q_idx].set_xlabel(r'KL$(p_0 \| p_1)$')
            axes[q_idx].set_title(f'{q_label}')
            stats_file.write("\n")

        axes[0].set_ylabel('Stratified MAE')
        fig.suptitle(f'Stratified MAE by True LDR Quartile - {TEST_SET_TITLES[test_idx]}', y=1.02)
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend_order = ["BDRE", "TDRE", "MDRE", "TSM", "TriangularMDRE", "VFM"]
        ordered_labels = [lbl for lbl in legend_order if lbl in by_label]
        ordered_labels += [lbl for lbl in by_label.keys() if lbl not in ordered_labels]
        plt.legend([by_label[lbl] for lbl in ordered_labels], ordered_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/stratified_mae_test{test_idx}.pdf')


# Create output directory
os.makedirs(FIGURES_DIR, exist_ok=True)

# Open stats file for writing
stats_filename = f'{FIGURES_DIR}/plot_stats.txt'
with open(stats_filename, 'w') as stats_file:
    stats_file.write("Density Ratio Estimation - Plot Statistics\n")
    stats_file.write(f"Generated from: {filename}\n")
    stats_file.write(f"Data dimension: {DATA_DIM}\n")
    stats_file.write(f"Training samples: {NSAMPLES_TRAIN}\n")
    stats_file.write(f"Test samples: {NSAMPLES_TEST}\n")
    stats_file.write(f"Number of test sets: {NTEST_SETS}\n")

    # Plot 1: MAE (original)
    if maes_by_kl:
        plot_metric(maes_by_kl, 'Mean Absolute Error\n(Test Set)', 'mae', stats_file)

    # Plot 2: Median AE
    if median_aes_by_kl:
        plot_metric(median_aes_by_kl, 'Median Absolute Error\n(Test Set)', 'median_ae', stats_file)

    # Plot 3: Spearman Correlation
    if spearman_by_kl:
        plot_metric(spearman_by_kl, 'Spearman Correlation', 'spearman_correlation', stats_file,
                    use_log_y=False, higher_is_better=True)

    # Plot 4: Trimmed MAE (IQR)
    if trimmed_mae_iqr_by_kl:
        plot_metric(trimmed_mae_iqr_by_kl, 'Trimmed MAE (IQR)\n(Test Set)', 'trimmed_mae_iqr', stats_file)

    # Plot 5: Stratified MAE by quartiles
    if stratified_mae_by_kl:
        plot_stratified_mae(stratified_mae_by_kl, stats_file)

print(f"Stats written to: {stats_filename}")
