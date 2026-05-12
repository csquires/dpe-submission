"""
Shared utilities for processing and plotting ELDR estimation results.

Lifts main() bodies from step3_process_results.py and step4_plot_results.py
into reusable functions accepting config path as first argument.
"""

import os
import h5py
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


# method colors (for fallback: unknown methods use 'gray')
METHODS = {
    'TriangularMDRE': 'blue',
    'MultiHeadTriangularTDRE': 'orange',
    'MHTTDRE': 'orange',
    'VFM': 'green',
    'TSM': 'red',
    'CTSM': 'purple',
    'TriangularCTSM': 'brown',
    'BDRE': 'cyan',
    'MDRE_15': 'magenta',
    'TDRE_5': 'pink',
    'TriangularTSM': 'gray',
}
FIGURE_SIZE = (8, 5)
FONT_SIZE = 12
ERROR_BAND_ALPHA = 0.2


def process_results_main(config_path: str) -> None:
    """
    Process raw eldr estimation results into per-alpha MAE and std statistics.

    Loads ground truth LDRs and estimated LDRs, computes per-pair MAE, aggregates
    to per-alpha mean/std ready for plotting. Writes mae_summary.h5.

    Args:
        config_path: path to config yaml (e.g., 'ex/<expname>/config.yaml')
            expected keys: alphas, data_dir, raw_results_dir, processed_results_dir,
            num_pairs_per_alpha, algorithms
    """
    # load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    alphas = config['alphas']
    data_dir = config['data_dir']
    raw_results_dir = config['raw_results_dir']
    processed_results_dir = config['processed_results_dir']
    algorithms = config['algorithms']

    n_alphas = len(alphas)
    n_pairs = config['num_pairs_per_alpha']

    # initialize output data structures
    mae_by_method = {method: np.zeros(n_alphas, dtype=np.float32) for method in algorithms}
    std_by_method = {method: np.zeros(n_alphas, dtype=np.float32) for method in algorithms}
    per_pair_by_method = {method: np.zeros((n_alphas, n_pairs), dtype=np.float32) for method in algorithms}
    kl_per_pair = np.full((n_alphas, n_pairs), np.nan, dtype=np.float32)

    # compute mae for each (alpha, pair) combination
    for alpha_idx in range(n_alphas):
        for pair_idx in range(n_pairs):
            data_file = f'{data_dir}/alpha_{alpha_idx}_pair_{pair_idx}.h5'
            raw_results_file = f'{raw_results_dir}/alpha_{alpha_idx}_pair_{pair_idx}.h5'

            # validate files exist
            if not os.path.exists(data_file):
                print(f'warning: data file not found {data_file}')
                for method in algorithms:
                    per_pair_by_method[method][alpha_idx, pair_idx] = np.nan
                continue
            if not os.path.exists(raw_results_file):
                print(f'warning: raw results file not found {raw_results_file}')
                for method in algorithms:
                    per_pair_by_method[method][alpha_idx, pair_idx] = np.nan
                continue

            # load ground truth
            try:
                with h5py.File(data_file, 'r') as f:
                    true_ldrs = f['true_ldrs'][:]
                    if 'kl_weights' in f:
                        kl_per_pair[alpha_idx, pair_idx] = float(f['kl_weights'][()])
            except Exception as e:
                print(f'warning: error reading {data_file}: {e}')
                for method in algorithms:
                    per_pair_by_method[method][alpha_idx, pair_idx] = np.nan
                continue

            # load and compute mae for each method
            try:
                with h5py.File(raw_results_file, 'r') as results_file:
                    for method in algorithms:
                        dataset_name = f'est_ldrs_{method}'
                        if dataset_name not in results_file:
                            print(f'warning: {dataset_name} not found in {raw_results_file}')
                            mae_val = np.nan
                        else:
                            est_ldrs = results_file[dataset_name][:]
                            abs_errors = np.abs(est_ldrs - true_ldrs)
                            mae_val = np.mean(abs_errors)

                        per_pair_by_method[method][alpha_idx, pair_idx] = mae_val
            except Exception as e:
                print(f'warning: error processing {raw_results_file}: {e}')
                for method in algorithms:
                    per_pair_by_method[method][alpha_idx, pair_idx] = np.nan

    # aggregate per-pair mae to per-alpha statistics
    for method in algorithms:
        per_pair_arr = per_pair_by_method[method]
        mae_list = []
        std_list = []

        for alpha_idx in range(n_alphas):
            mae_values = per_pair_arr[alpha_idx, :]
            mae_mean = np.nanmean(mae_values)
            mae_std = np.nanstd(mae_values, ddof=1)

            mae_list.append(mae_mean)
            std_list.append(mae_std)

        mae_by_method[method] = np.array(mae_list, dtype=np.float32)
        std_by_method[method] = np.array(std_list, dtype=np.float32)

    # aggregate per-pair kl to per-alpha statistics
    kl_mean = np.nanmean(kl_per_pair, axis=1)
    kl_std = np.nanstd(kl_per_pair, axis=1, ddof=1)

    # validation before save
    for method in algorithms:
        mae_arr = mae_by_method[method]
        assert mae_arr.shape == (n_alphas,), f'mae array shape mismatch for {method}'
        assert len(std_by_method[method]) == n_alphas, f'std array shape mismatch for {method}'

        if np.all(np.isnan(mae_arr)):
            print(f'warning: {method} has all-nan mae values')

        if per_pair_by_method[method].shape != (n_alphas, n_pairs):
            print(f'warning: per_pair shape mismatch for {method}')

    # create output directory and save results
    os.makedirs(processed_results_dir, exist_ok=True)
    processed_results_file = f'{processed_results_dir}/mae_summary.h5'

    with h5py.File(processed_results_file, 'w') as out_file:
        # store alpha values
        out_file.create_dataset('alphas', data=np.array(alphas, dtype=np.float32))

        # store metrics for each algorithm
        for method in algorithms:
            out_file.create_dataset(f'mae_{method}', data=mae_by_method[method])
            out_file.create_dataset(f'std_{method}', data=std_by_method[method])
            out_file.create_dataset(f'per_pair_{method}', data=per_pair_by_method[method])

        # store kl divergence data
        out_file.create_dataset('kl_per_pair', data=kl_per_pair)
        out_file.create_dataset('kl_mean', data=kl_mean)
        out_file.create_dataset('kl_std', data=kl_std)

    # print summary report
    print("\n" + "="*80)
    print("MNIST ELDR Estimation: Step 3 Results Processing Complete")
    print("="*80)
    print(f"Processed results saved to: {processed_results_file}")
    print(f"\nAlphas: {alphas}")
    print(f"Pairs per alpha: {n_pairs}")
    print(f"Algorithms: {algorithms}")
    print(f"\nMetrics computed per algorithm:")
    for method in algorithms:
        mae_arr = mae_by_method[method]
        std_arr = std_by_method[method]
        mae_str = ', '.join([f'{x:.4f}' for x in mae_arr])
        std_str = ', '.join([f'{x:.4f}' for x in std_arr])
        print(f"  {method:25s}: MAE=[{mae_str}], Std=[{std_str}]")
    print(f"\nKL(w, w') per alpha:")
    for i, alpha in enumerate(alphas):
        print(f"  alpha={alpha:.1f}: mean={kl_mean[i]:.4f}, std={kl_std[i]:.4f}")
    print("="*80 + "\n")


def plot_results_main(config_path: str) -> None:
    """
    Plot alpha vs MAE with translucent error bands for all methods.

    Loads results from mae_summary.h5 and creates PDF and PNG outputs.
    Discovers methods from HDF5 keys (mae_*) to allow partial runs.
    Falls back to 'gray' for unknown method colors.

    Args:
        config_path: path to config yaml (e.g., 'ex/<expname>/config.yaml')
            expected keys: processed_results_dir, figures_dir

    Raises:
        FileNotFoundError: if mae_summary.h5 not found
        KeyError: if alphas dataset missing
        ValueError: if alphas empty or shape mismatch
    """
    # load config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    processed_results_dir = config['processed_results_dir']
    figures_dir = config['figures_dir']

    # ensure output directory exists
    os.makedirs(figures_dir, exist_ok=True)

    # load results from HDF5
    h5_path = os.path.join(processed_results_dir, 'mae_summary.h5')
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f'mae_summary.h5 not found at {h5_path}')

    results = {}
    try:
        with h5py.File(h5_path, 'r') as f:
            # load alphas
            if 'alphas' not in f:
                raise KeyError('alphas dataset missing in HDF5 file')
            alphas = f['alphas'][:]

            # verify alphas is non-empty
            if len(alphas) == 0:
                raise ValueError('alphas array is empty')

            results['alphas'] = alphas

            # load kl_mean if present
            kl_mean = f['kl_mean'][:] if 'kl_mean' in f else None
            results['kl_mean'] = kl_mean

            # load method results: discover from HDF5 keys (mae_*) so partial runs work
            for key in f.keys():
                if not key.startswith('mae_'):
                    continue
                method = key[len('mae_'):]
                std_key = f'std_{method}'
                if std_key not in f:
                    print(f'warning: {std_key} missing for {method}; skipping')
                    continue
                mae = f[key][:]
                std = f[std_key][:]
                if mae.shape != alphas.shape:
                    raise ValueError(f'{method} mae shape {mae.shape} does not match alphas {alphas.shape}')
                if std.shape != alphas.shape:
                    raise ValueError(f'{method} std shape {std.shape} does not match alphas {alphas.shape}')
                results[method] = (mae, std)

    except OSError as e:
        raise IOError(f'error reading HDF5 file: {e}')

    # configure plot style
    sns.set_style('whitegrid')
    matplotlib.rcParams['font.size'] = FONT_SIZE

    # create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # plot each loaded method, falling back to gray if not in METHODS color map
    alphas = results['alphas']
    method_keys = [m for m in results if m not in ('alphas', 'kl_mean')]
    for method in sorted(method_keys):
        mae, std = results[method]
        # skip methods that are all-NaN (no valid cells); else log-scale fails
        if not np.any(np.isfinite(mae)):
            print(f"  skipping {method}: all NaN")
            continue
        color = METHODS.get(method, 'gray')
        ax.plot(alphas, mae, label=method, color=color, linewidth=2)
        ax.fill_between(alphas, mae - std, mae + std, color=color, alpha=ERROR_BAND_ALPHA)

    # configure axes
    ax.set_xscale('log')
    ax.set_xlabel('Alpha', fontsize=FONT_SIZE)
    ax.set_ylabel('LDR MAE', fontsize=FONT_SIZE)
    ax.set_title('Log Density Ratio MAE vs Alpha', fontsize=FONT_SIZE)
    ax.legend(loc='best', fontsize=FONT_SIZE)
    ax.grid(True, alpha=0.3)

    if results.get('kl_mean') is not None and not np.any(np.isnan(results['kl_mean'])):
        kl_mean = results['kl_mean']
        sort_idx = np.argsort(alphas)
        alphas_sorted = alphas[sort_idx]
        kl_sorted = kl_mean[sort_idx]

        if np.all(np.diff(kl_sorted) < 0):
            def alpha_to_kl(a):
                return np.interp(a, alphas_sorted, kl_sorted)

            def kl_to_alpha(k):
                return np.interp(k, kl_sorted[::-1], alphas_sorted[::-1])

            sec_ax = ax.secondary_xaxis('top', functions=(alpha_to_kl, kl_to_alpha))
            sec_ax.set_xlabel('Mean KL(w, w\')', fontsize=FONT_SIZE)

    # save figures
    pdf_path = os.path.join(figures_dir, 'mae_vs_alpha.pdf')
    png_path = os.path.join(figures_dir, 'mae_vs_alpha.png')

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Figure saved to: {pdf_path}')
    print(f'PNG saved to: {png_path}')
