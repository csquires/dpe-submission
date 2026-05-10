"""
step5_compare.py

compare processed eldr estimation results between old (baseline) and new
(conditional flow) experiments. produces rank correlation metrics and
side-by-side comparison plots showing agreement in method ordering and magnitude.
"""

import argparse
import os
import csv
import yaml
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.stats import spearmanr
except ImportError:
    # fallback: rank correlation via Pearson on rank arrays
    def spearmanr(a, b):
        """compute Spearman correlation from rank arrays using Pearson."""
        rank_a = np.argsort(np.argsort(a)).astype(float)
        rank_b = np.argsort(np.argsort(b)).astype(float)
        return np.corrcoef(rank_a, rank_b)[0, 1], 1.0


def load_processed_results(processed_results_dir, algorithms, alphas):
    """
    load mae and std from mae_summary.h5 for each method and alpha.

    args:
        processed_results_dir: path to directory containing mae_summary.h5
        algorithms: list of method names to load
        alphas: list of alpha values (used for validation)

    returns:
        dict: {method_name: {alpha_value: (mae_mean, mae_std)}}
    """
    h5_path = os.path.join(processed_results_dir, 'mae_summary.h5')

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"mae_summary.h5 not found at {h5_path}; ensure step 3 has completed."
        )

    results = {}

    with h5py.File(h5_path, 'r') as f:
        # load and validate alphas
        file_alphas = f['alphas'][:]
        if len(file_alphas) != len(alphas):
            print(f"warning: alpha count mismatch: file={len(file_alphas)}, "
                  f"config={len(alphas)}")
        if not np.allclose(file_alphas, alphas):
            print(f"warning: alpha values mismatch; using file alphas")
        alphas_to_use = file_alphas

        # load metrics for each algorithm
        for method in algorithms:
            mae_key = f'mae_{method}'
            std_key = f'std_{method}'

            if mae_key not in f or std_key not in f:
                print(f"warning: {mae_key} or {std_key} not found in {h5_path}, "
                      f"skipping method")
                continue

            mae_arr = f[mae_key][:]
            std_arr = f[std_key][:]

            # validate shapes
            if mae_arr.shape != (len(alphas_to_use),):
                print(f"warning: mae array shape mismatch for {method}")
                continue
            if std_arr.shape != (len(alphas_to_use),):
                print(f"warning: std array shape mismatch for {method}")
                continue

            # check for invalid values
            for i, (m, s) in enumerate(zip(mae_arr, std_arr)):
                if m <= 0 or np.isnan(m):
                    print(f"warning: invalid MAE for {method} at alpha_idx={i}")
                if np.isnan(s):
                    print(f"warning: NaN std for {method} at alpha_idx={i}")

            # store as dict keyed by alpha value
            results[method] = {}
            for i, alpha_val in enumerate(alphas_to_use):
                results[method][float(alpha_val)] = (float(mae_arr[i]), float(std_arr[i]))

    return results


def compute_rank_correlation(old_maes, new_maes):
    """
    compute Spearman rank correlation of method ordering by MAE.

    args:
        old_maes: dict of {method: mae_value} for old experiment
        new_maes: dict of {method: mae_value} for new experiment

    returns:
        float: Spearman rho (or NaN if < 2 valid methods)
    """
    # find common methods
    common = set(old_maes.keys()) & set(new_maes.keys())

    if len(common) < 2:
        print(f"warning: fewer than 2 valid methods, rank correlation = NaN")
        return np.nan

    # extract MAE values in consistent method order
    methods = sorted(common)
    old_mae_list = []
    new_mae_list = []

    for method in methods:
        old_mae = old_maes[method]
        new_mae = new_maes[method]

        # skip invalid MAE values
        if old_mae <= 0 or np.isnan(old_mae) or new_mae <= 0 or np.isnan(new_mae):
            print(f"warning: invalid MAE for {method}, skipping from rank correlation")
            continue

        old_mae_list.append(old_mae)
        new_mae_list.append(new_mae)

    if len(old_mae_list) < 2:
        print(f"warning: fewer than 2 valid methods after filtering, rank correlation = NaN")
        return np.nan

    rho, _ = spearmanr(old_mae_list, new_mae_list)
    return float(rho)


def main():
    parser = argparse.ArgumentParser(
        description='compare old and new MNIST ELDR conditional flow experiments'
    )
    parser.add_argument('--old-config', required=True,
                       help='path to old experiment config YAML')
    parser.add_argument('--new-config', required=True,
                       help='path to new experiment config YAML')
    parser.add_argument('--output-dir', default='experiments/mnist/figures',
                       help='output directory for comparison plots and summary CSV')
    args = parser.parse_args()

    # load configurations
    with open(args.old_config, 'r') as f:
        old_config = yaml.safe_load(f)
    with open(args.new_config, 'r') as f:
        new_config = yaml.safe_load(f)

    old_processed_dir = old_config['processed_results_dir']
    new_processed_dir = new_config['processed_results_dir']
    old_algorithms = old_config['algorithms']
    new_algorithms = new_config['algorithms']
    old_alphas = old_config['alphas']
    new_alphas = new_config['alphas']

    # load processed results
    print(f"loading old results from {old_processed_dir}...")
    old_results = load_processed_results(old_processed_dir, old_algorithms, old_alphas)
    print(f"loading new results from {new_processed_dir}...")
    new_results = load_processed_results(new_processed_dir, new_algorithms, new_alphas)

    # determine common methods and alphas
    common_methods = sorted(set(old_results.keys()) & set(new_results.keys()))
    if len(common_methods) != len(old_algorithms):
        missing = set(old_algorithms) - set(common_methods)
        print(f"warning: methods in old but not new: {missing}")
    if len(common_methods) != len(new_algorithms):
        missing = set(new_algorithms) - set(common_methods)
        print(f"warning: methods in new but not old: {missing}")

    # extract common alpha set
    old_alpha_set = set(old_results[common_methods[0]].keys()) if common_methods else set()
    new_alpha_set = set(new_results[common_methods[0]].keys()) if common_methods else set()
    common_alphas = sorted(old_alpha_set & new_alpha_set)

    if len(common_alphas) != len(old_alphas):
        print(f"warning: alphas differ; using intersection: {common_alphas}")

    # build comparison table
    comparison_data = []
    for method in common_methods:
        for alpha in common_alphas:
            old_mae, old_std = old_results[method][alpha]
            new_mae, new_std = new_results[method][alpha]

            # compute ratio with zero-division guard
            if old_mae < 1e-12:
                ratio = np.nan
                print(f"warning: mae_old near zero for {method} at alpha={alpha}, "
                      f"ratio = NaN")
            else:
                ratio = new_mae / old_mae

            comparison_data.append({
                'method': method,
                'alpha': alpha,
                'mae_old': old_mae,
                'std_old': old_std,
                'mae_new': new_mae,
                'std_new': new_std,
                'ratio_new_over_old': ratio
            })

    # compute rank correlations per alpha
    rank_corr_data = []
    for alpha in common_alphas:
        old_maes = {m: old_results[m][alpha][0] for m in common_methods}
        new_maes = {m: new_results[m][alpha][0] for m in common_methods}
        rho = compute_rank_correlation(old_maes, new_maes)
        rank_corr_data.append({
            'alpha': alpha,
            'spearman_rho': rho,
            'n_methods_used': len(common_methods)
        })

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # save comparison CSV
    csv_path = os.path.join(args.output_dir, 'comparison_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['method', 'alpha', 'mae_old', 'std_old', 'mae_new',
                       'std_new', 'ratio_new_over_old']
        )
        writer.writeheader()
        for row in comparison_data:
            writer.writerow(row)

        # append rank correlation section with separate header
        f.write('\n')
        rank_writer = csv.DictWriter(f, fieldnames=['method', 'alpha', 'mae_old', 'std_old'])
        rank_writer.writerow({'method': '_rank_corr', 'alpha': 'alpha', 'mae_old': 'spearman_rho', 'std_old': 'n_methods_used'})
        for row in rank_corr_data:
            rank_writer.writerow({
                'method': '_rank_corr',
                'alpha': row['alpha'],
                'mae_old': row['spearman_rho'],
                'std_old': row['n_methods_used']
            })

    # generate comparison plots
    sns.set_style('whitegrid')
    plt.rcParams['font.size'] = 10

    n_methods = len(common_methods)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig_width = max(12, 6 * n_cols)
    fig_height = max(8, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_methods == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # plot each method
    for idx, method in enumerate(common_methods):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # extract MAE and std for this method across alphas
        old_mae_arr = np.array([old_results[method][a][0] for a in common_alphas])
        old_std_arr = np.array([old_results[method][a][1] for a in common_alphas])
        new_mae_arr = np.array([new_results[method][a][0] for a in common_alphas])
        new_std_arr = np.array([new_results[method][a][1] for a in common_alphas])

        # plot curves with error bands
        ax.plot(common_alphas, old_mae_arr, linewidth=2, label='Old', color='gray')
        ax.fill_between(common_alphas, old_mae_arr - old_std_arr, old_mae_arr + old_std_arr,
                        alpha=0.2, color='gray')

        ax.plot(common_alphas, new_mae_arr, linewidth=2, label='New', color='tab:blue')
        ax.fill_between(common_alphas, new_mae_arr - new_std_arr, new_mae_arr + new_std_arr,
                        alpha=0.2, color='tab:blue')

        ax.set_xlabel('alpha', fontsize=10)
        ax.set_ylabel('MAE', fontsize=10)
        ax.set_title(method, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', alpha=0.3)

    # hide unused subplots
    for idx in range(n_methods, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.tight_layout()
    png_path = os.path.join(args.output_dir, 'comparison.png')
    fig.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    # print summary report
    print("\n" + "="*80)
    print("MNIST ELDR Conditional Flow: Step 5 Comparison Complete")
    print("="*80)
    print(f"Old Config:    {args.old_config}")
    print(f"New Config:    {args.new_config}")
    print(f"Output Dir:    {args.output_dir}")
    print()
    print(f"Common Methods (intersection): {common_methods}")
    print(f"Common Alphas (intersection):  {common_alphas}")
    print()
    print("Per-Method MAE Summary:")
    for method in common_methods:
        print(f"  {method}:")
        for alpha in common_alphas:
            mae_old, std_old = old_results[method][alpha]
            mae_new, std_new = new_results[method][alpha]
            ratio = mae_new / mae_old if mae_old >= 1e-12 else np.nan
            ratio_str = f"{ratio:.3f}" if not np.isnan(ratio) else "NaN"
            print(f"    alpha={alpha:.2f}: old={mae_old:.4f}±{std_old:.4f}, "
                  f"new={mae_new:.4f}±{std_new:.4f}, ratio={ratio_str}")
    print()
    print("Spearman Rank Correlation (method ordering):")
    for row in rank_corr_data:
        rho_str = f"{row['spearman_rho']:.3f}" if not np.isnan(row['spearman_rho']) else "NaN"
        print(f"  alpha={row['alpha']:.2f}:   rho={rho_str}  ({row['n_methods_used']} methods)")
    print()
    print(f"Figures saved to: {png_path}")
    print(f"Summary saved to: {csv_path}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
