import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# inlined legacy method-name aliases. used at h5 load time to canonicalize
# entries written by older runs (pre-rename). new runs use canonical names
# and bypass this map entirely.
LEGACY_ALIASES: dict[str, str] = {
    "MHTTDRE": "MultiHeadTriangularTDRE",
    "MDRE": "MDRE_15",
    "TDRE": "TDRE_5",
    "TriangularCTSM": "TriangularCTSM_V1",
    "TriangularVFM": "TriangularVFM_V1",
}


config = yaml.load(open('ex/dre_sample_complexity/config.yaml', 'r'), Loader=yaml.FullLoader)
# directories
PROCESSED_RESULTS_DIR = config['processed_results_dir']
FIGURES_DIR = config['figures_dir']
# dataset parameters
DATA_DIM = config['data_dim']
KL_DIVERGENCES = config['kl_divergences']
NSAMPLES_TRAIN_VALUES = config['nsamples_train_values']
NSAMPLES_TEST = config['nsamples_test']

filename = f'{PROCESSED_RESULTS_DIR}/metrics.h5'


def _apply_alias(method_name):
    """apply legacy alias mapping to canonicalize method names."""
    return LEGACY_ALIASES.get(method_name, method_name)


# Load all metrics from file
with h5py.File(filename, 'r') as f:
    # Shape: (n_kl, n_instances, n_nsamples_train)
    # apply aliases at load time to transform old keys to canonical names
    maes_by_alg = {_apply_alias(key.replace('maes_', '')): f[key][:] for key in f.keys() if key.startswith('maes_')}
    median_aes_by_alg = {_apply_alias(key.replace('median_aes_', '')): f[key][:] for key in f.keys() if key.startswith('median_aes_')}
    spearman_by_alg = {_apply_alias(key.replace('spearman_', '')): f[key][:] for key in f.keys() if key.startswith('spearman_')}
    trimmed_mae_iqr_by_alg = {_apply_alias(key.replace('trimmed_mae_iqr_', '')): f[key][:] for key in f.keys() if key.startswith('trimmed_mae_iqr_')}
    # Stratified MAE by quartiles
    stratified_mae_by_alg = {}
    for key in f.keys():
        if key.startswith('stratified_mae_q'):
            # key format: stratified_mae_q1_ALGNAME
            parts = key.split('_')
            # parts = ['stratified', 'mae', 'q1', 'ALGNAME']
            quartile = parts[2]  # 'q1', 'q2', etc.
            alg_name = '_'.join(parts[3:])  # handle algorithm names with underscores
            alg_name = _apply_alias(alg_name)  # apply alias to canonicalize
            if alg_name not in stratified_mae_by_alg:
                stratified_mae_by_alg[alg_name] = {}
            stratified_mae_by_alg[alg_name][quartile] = f[key][:]

# colors - consistent across all experiments (canonical names + legacy aliases)
# colors - one per current method
colors = {
    "BDRE": "#1f77b4",
    "TDRE_5": "#ff7f0e",
    "TDRE": "#ff7f0e",  # legacy alias for backward compat
    "MDRE_15": "#2ca02c",
    "MDRE": "#2ca02c",  # legacy alias for backward compat
    "MDRE": "#2ca02c",
    "MHTDRE": "#ff7f0e",
    "TSM": "#d62728",
    "CTSM": "#8c564b",
    "VFM": "#9467bd",
    "TriangularCTSM_V1": "#17becf",
    "TriangularVFM_V1": "#bcbd22",
    "MultiHeadTriangularTDRE": "#e377c2",
    "FMDRE": "#e377c2",
    "FMDRE_S2": "#7f7f7f",
}

algorithm_order = [
    "BDRE",
    "MDRE",
    "MHTDRE",
    "TSM",
    "CTSM",
    "VFM",
    "FMDRE",
    "FMDRE_S2",
]

KL_TITLES = [rf'KL$(p_0 \| p_1) = {kl}$' for kl in KL_DIVERGENCES]


def get_algorithms_to_plot(data_dict):
    """Get list of algorithms in standard order: BDRE -> TDRE_5 -> MDRE_15 -> TSM -> TriangularMDRE -> VFM -> new methods.

    Returns tuples of (canonical_key, canonical_label) after aliasing at load time.
    """
    algs = []
    if "BDRE" in data_dict:
        algs.append(("BDRE", "BDRE"))
    if "TDRE_5" in data_dict:
        algs.append(("TDRE_5", "TDRE_5"))
    if "MDRE_15" in data_dict:
        algs.append(("MDRE_15", "MDRE_15"))
    if "TSM" in data_dict:
        algs.append(("TSM", "TSM"))
    if "TriangularMDRE" in data_dict:
        algs.append(("TriangularMDRE", "TriangularMDRE"))
    if "VFM" in data_dict:
        algs.append(("VFM", "VFM"))
    if "TriangularCTSM_V1" in data_dict:
        algs.append(("TriangularCTSM_V1", "TriangularCTSM_V1"))
    if "TriangularVFM_V1" in data_dict:
        algs.append(("TriangularVFM_V1", "TriangularVFM_V1"))
    if "MultiHeadTriangularTDRE" in data_dict:
        algs.append(("MultiHeadTriangularTDRE", "MultiHeadTriangularTDRE"))
    return algs
    """Get list of algorithms in the current experiment order."""
    return [(alg, alg) for alg in algorithm_order if alg in data_dict]


def plot_metric(data_by_alg, ylabel, figure_name, stats_file, use_log_y=True, higher_is_better=False):
    """Generic plotting function for a metric.

    Args:
        data_by_alg: dict mapping alg_name -> array of shape (n_kl, n_instances, n_nsamples_train)
        ylabel: Y-axis label
        figure_name: Output filename (without extension)
        stats_file: File handle to write stats to
        use_log_y: Whether to use log scale on y-axis
        higher_is_better: If True, don't use log scale (e.g., for correlation)
    """
    # Average over instances: (n_kl, n_nsamples_train)
    avg_by_alg = {alg: np.mean(arr, axis=1) for alg, arr in data_by_alg.items()}
    n_kl = len(KL_DIVERGENCES)

    all_vals = [arr for arr in avg_by_alg.values()]
    if all_vals:
        all_mins = [arr.min() for arr in all_vals]
        all_maxs = [arr.max() for arr in all_vals]
        y_min = min(all_mins)
        y_max = max(all_maxs)
    else:
        y_min, y_max = 0.0, 1.0

    plt.clf()
    sns.set_style('whitegrid')
    try:
        plt.style.use('full-width.mplstyle')
    except OSError:
        pass  # Style file not found, use defaults
    fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=n_kl)

    algorithms = get_algorithms_to_plot(avg_by_alg)

    # Write header to stats file
    stats_file.write(f"\n{'='*60}\n")
    stats_file.write(f"Figure: {figure_name}\n")
    stats_file.write(f"Metric: {ylabel}\n")
    stats_file.write(f"{'='*60}\n")
    stats_file.write(f"X-axis: nsamples_train = {NSAMPLES_TRAIN_VALUES}\n\n")

    for kl_idx in range(n_kl):
        stats_file.write(f"KL = {KL_DIVERGENCES[kl_idx]}:\n")
        for alg_key, alg_label in algorithms:
            if alg_key in avg_by_alg:
                y_vals = avg_by_alg[alg_key][kl_idx, :]
                axes[kl_idx].plot(NSAMPLES_TRAIN_VALUES, y_vals, label=alg_label, color=colors.get(alg_key, "#333333"), linewidth=1.0)
                stats_file.write(f"  {alg_label} ({alg_key}): {y_vals.tolist()}\n")
        stats_file.write("\n")

        if use_log_y and not higher_is_better:
            axes[kl_idx].set_ylim(y_min * 0.9, y_max * 1.1)
            axes[kl_idx].set_yscale('log')
        else:
            # For correlation, set reasonable bounds
            if higher_is_better:
                axes[kl_idx].set_ylim(min(0, y_min - 0.1), 1.05)
            else:
                axes[kl_idx].set_ylim(y_min * 0.9, y_max * 1.1)
        axes[kl_idx].set_xscale('log')
        axes[kl_idx].set_title(KL_TITLES[kl_idx])
        axes[kl_idx].set_xlabel(r'$n_{\mathrm{train}}$')

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    # Remove duplicate labels
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/{figure_name}.pdf')


def plot_stratified_mae(stratified_data, stats_file):
    """Plot stratified MAE with one subplot per quartile for each KL distance."""
    quartiles = ['q1', 'q2', 'q3', 'q4']
    quartile_labels = ['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)']
    n_kl = len(KL_DIVERGENCES)

    # For each KL distance, create a figure with 4 subplots (one per quartile)
    for kl_idx in range(n_kl):
        plt.clf()
        sns.set_style('whitegrid')
        try:
            plt.style.use('full-width.mplstyle')
        except OSError:
            pass
        fig, axes = plt.subplots(figsize=(12, 3), nrows=1, ncols=4)

        algorithms = get_algorithms_to_plot(stratified_data)

        stats_file.write(f"\n{'='*60}\n")
        stats_file.write(f"Figure: stratified_mae_kl{kl_idx}\n")
        stats_file.write(f"KL: {KL_DIVERGENCES[kl_idx]}\n")
        stats_file.write(f"{'='*60}\n")
        stats_file.write(f"X-axis: nsamples_train = {NSAMPLES_TRAIN_VALUES}\n\n")

        all_vals = []
        for alg_key, _ in algorithms:
            if alg_key in stratified_data:
                for q in quartiles:
                    arr = stratified_data[alg_key].get(q)
                    if arr is not None:
                        avg = np.mean(arr, axis=1)[kl_idx, :]
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
                    avg = np.mean(arr, axis=1)[kl_idx, :]
                    axes[q_idx].plot(NSAMPLES_TRAIN_VALUES, avg, label=alg_label, color=colors.get(alg_key, "#333333"), linewidth=1.0, marker='o', markersize=3)
                    stats_file.write(f"  {alg_label} ({alg_key}): {avg.tolist()}\n")

            axes[q_idx].set_xscale('log')
            axes[q_idx].set_yscale('log')
            axes[q_idx].set_ylim(y_min * 0.9, y_max * 1.1)
            axes[q_idx].set_xlabel(r'$n_{\mathrm{train}}$')
            axes[q_idx].set_title(f'{q_label}')
            stats_file.write("\n")

        axes[0].set_ylabel('Stratified MAE')
        fig.suptitle(f'Stratified MAE by True LDR Quartile - {KL_TITLES[kl_idx]}', y=1.02)
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/stratified_mae_kl{kl_idx}.pdf')


# Create output directory
os.makedirs(FIGURES_DIR, exist_ok=True)

# Open stats file for writing
stats_filename = f'{FIGURES_DIR}/plot_stats.txt'
with open(stats_filename, 'w') as stats_file:
    stats_file.write("DRE Sample Complexity - Plot Statistics\n")
    stats_file.write(f"Generated from: {filename}\n")
    stats_file.write(f"Data dimension: {DATA_DIM}\n")
    stats_file.write(f"Training sample sizes: {NSAMPLES_TRAIN_VALUES}\n")
    stats_file.write(f"Test samples: {NSAMPLES_TEST}\n")
    stats_file.write(f"KL distances: {KL_DIVERGENCES}\n")

    # Plot 1: MAE (original)
    if maes_by_alg:
        plot_metric(maes_by_alg, 'Mean Absolute Error', 'mae', stats_file)

    # Plot 2: Median AE
    if median_aes_by_alg:
        plot_metric(median_aes_by_alg, 'Median Absolute Error', 'median_ae', stats_file)

    # Plot 3: Spearman Correlation
    if spearman_by_alg:
        plot_metric(spearman_by_alg, 'Spearman Correlation', 'spearman_correlation', stats_file,
                    use_log_y=False, higher_is_better=True)

    # Plot 4: Trimmed MAE (IQR)
    if trimmed_mae_iqr_by_alg:
        plot_metric(trimmed_mae_iqr_by_alg, 'Trimmed MAE (IQR)', 'trimmed_mae_iqr', stats_file)

    # Plot 5: Stratified MAE by quartiles
    if stratified_mae_by_alg:
        plot_stratified_mae(stratified_mae_by_alg, stats_file)

print(f"Stats written to: {stats_filename}")
print(f"Figures saved to: {FIGURES_DIR}/")
