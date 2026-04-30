"""
Step 4: Plot Results for SMODICE ELDR Estimation

Load HDF5 summaries and generate three publication-grade figures:
1. Headline heatmap: 5×5 MAE grid per algorithm (faceted layout)
2. Disentanglement panel: Enc-1 (onehot_joint) vs Enc-2 (onehot_concat) side-by-side
3. Sigma sensitivity: line plots at selected (k1, k2) cells across sigma values
"""
import yaml
import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Rectangle
import warnings


# method colors (baseline from mnist_eldr_estimation, extended for new methods)
METHOD_COLORS = {
    'TabularPluginDRE': '#1f77b4',           # blue
    'BDRE': '#2ca02c',                       # green
    'MDRE_15': '#d62728',                    # red (renamed from MDRE)
    'TriangularMDRE': '#ff7f0e',             # orange
    'MultiHeadTriangularTDRE': '#9467bd',   # purple
    'TSM': '#8c564b',                        # brown
    'CTSM': '#e377c2',                       # pink
    'TriangularCTSM_V1': '#7f7f7f',          # gray (renamed from TriangularCTSM)
    'TriangularVFM_V1': '#bcbd22',           # olive (renamed from TriangularVFM)
    # new methods (distinct colors from standard palette)
    'TriangularTSM': '#17becf',              # cyan
    'FMDRE': '#ff9896',                      # light red
    'FMDRE_S2': '#aec7e8',                   # light blue
    'TriangularFMDRE': '#c5b0d5',            # light purple
    'VFM': '#c49c94',                        # light brown
    'TDRE_5': '#f7b6d2',                     # light pink
    'TriangularCTSM_V2': '#c7c7c7',          # light gray
    'TriangularCTSM_V3': '#9edae5',          # light cyan
    'TriangularVFM_V2': '#393b79',           # dark blue
    'TriangularVFM_V3': '#637939',           # dark green
}

# oracle reference: rendered as separate annotation, not a "method"
ORACLE_METHOD = 'SmoothedTabularPluginDRE'

# classifier-only subset (score methods skip onehot, so use for disentanglement panel)
CLASSIFIER_METHODS = {
    'TabularPluginDRE', 'BDRE', 'MDRE', 'TriangularMDRE',
    'MultiHeadTriangularTDRE'
}


def parse_args():
    """
    parse CLI: config path, encoding (default 'gaussian_blob'), metric.

    returns: argparse.Namespace with attributes:
      - config: path to config.yaml
      - encoding: encoding type ('gaussian_blob', 'onehot_joint', etc.)
      - metric: 'pointwise_mae' or 'eldr_err'
    """
    parser = argparse.ArgumentParser(
        description='Plot SMODICE ELDR estimation results'
    )
    parser.add_argument(
        '--config',
        default='experiments/smodice_eldr_estimation/config.yaml',
        help='path to config yaml',
    )
    parser.add_argument(
        '--encoding',
        default='gaussian_blob',
        help='encoding type (gaussian_blob, onehot_joint, onehot_concat, flow_pushforward)',
    )
    parser.add_argument(
        '--metric',
        default='pointwise_mae',
        choices=['pointwise_mae', 'eldr_err'],
        help='metric to plot (pointwise_mae or eldr_err)',
    )
    return parser.parse_args()


def load_summary_h5(h5_path, metric, encoding, k1_values, k2_values):
    """
    load HDF5 summary written by step3 -- FLAT layout (datasets at root).

    HDF5 schema:
      /{metric}_{method}_mean    -- float32 [G_k1, G_k2]
      /{metric}_{method}_se      -- float32 [G_k1, G_k2]
      /{metric}_{method}_n       -- int32   [G_k1, G_k2]
      /feasibility               -- int8    [G_k1, G_k2]
      /k1_values, /k2_values     -- float32

    method names are recovered by scanning dataset keys for the prefix
    f"{metric}_" and the suffix "_mean", and stripping both.

    args:
      h5_path: path to summary.h5
      metric: 'pointwise_mae' or 'eldr_err'
      encoding: encoding type (used only for diagnostics; layout is encoding-agnostic)
      k1_values, k2_values: grid coordinates (lists of floats)

    returns: dict mapping method -> (mean_arr[k1, k2], se_arr[k1, k2])
              we keep the variable name `std` in caller code for backwards readability,
              but the values are standard ERRORS as written by step3.

    raises: FileNotFoundError if h5_path missing.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f'summary.h5 not found at {h5_path}; '
            f'run step3_process_results.py first'
        )

    results = {}
    prefix = f"{metric}_"
    suffix = "_mean"
    expected_shape = (len(k1_values), len(k2_values))

    try:
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                if not (key.startswith(prefix) and key.endswith(suffix)):
                    continue
                method_name = key[len(prefix):-len(suffix)]
                # skip the "discrete" reference key (TabularPluginDRE on blob/flow);
                # that one is loaded explicitly when needed.
                if method_name.startswith("discrete_"):
                    continue

                mean_key = f"{prefix}{method_name}_mean"
                se_key = f"{prefix}{method_name}_se"
                if se_key not in f:
                    warnings.warn(
                        f'{method_name} missing {se_key}; skipping'
                    )
                    continue

                mean = f[mean_key][:]
                se = f[se_key][:]

                if mean.shape != expected_shape:
                    raise ValueError(
                        f'{method_name} {mean_key} shape {mean.shape} '
                        f'!= expected {expected_shape}'
                    )

                # tuple is (mean, standard error). caller can rename freely.
                results[method_name] = (mean, se)

    except OSError as e:
        raise IOError(f'error reading HDF5: {e}')

    return results


def get_shared_colorscale(results, vmin_percentile=2, vmax_percentile=98):
    """
    compute shared vmin/vmax across all methods' mean arrays.

    args:
      results: dict mapping method -> (mean_arr, std_arr)
      vmin_percentile, vmax_percentile: percentiles to robustly clip outliers

    returns: (vmin, vmax) scalars

    note: exclude NaN; if all NaN, default to (0, 1).
    """
    all_means = []
    for mean, _ in results.values():
        valid = mean[~np.isnan(mean)]
        if len(valid) > 0:
            all_means.extend(valid)

    if len(all_means) == 0:
        return 0.0, 1.0

    vmin = np.percentile(all_means, vmin_percentile)
    vmax = np.percentile(all_means, vmax_percentile)

    return vmin, vmax


def fig_headline(results, k1_values, k2_values, metric, encoding, sigma,
                 figures_dir):
    """
    generate 5x5 MAE heatmap per algorithm in faceted grid layout.

    layout:
      - sort methods alphabetically, exclude oracle reference.
      - arrange in grid (e.g., 3 rows x 3 cols for 9 classifiers).
      - one oracle subplot (dashed line annotation or separate inset).

    each heatmap:
      - x-axis: K1 (log-spaced tick labels)
      - y-axis: K2 (log-spaced tick labels)
      - color: shared MAE scale (cmap='viridis')
      - cells: imshow; annotate with "x.xx +/- y.yy" (fontsize 8)
      - infeasible (NaN): hatched gray

    args:
      results: dict method -> (mean, std)
      k1_values, k2_values: list of floats
      metric: 'pointwise_mae' or 'eldr_err' (for title)
      encoding: 'gaussian_blob' (for title)
      sigma: float; used in title if encoding is blob
      figures_dir: output directory

    side effects: creates headline_{encoding}_sigma_{s:.3f}_{metric}.pdf
    """

    # Load style
    style_path = '/home/aviamala/dpe-submission/full-width.mplstyle'
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        plt.rcParams['font.size'] = 14

    # Filter out oracle; sort methods
    method_names = sorted([
        m for m in results.keys() if m != ORACLE_METHOD
    ])

    # Get shared color scale
    vmin, vmax = get_shared_colorscale(results)

    # Layout: fit all methods into a grid
    n_methods = len(method_names)
    n_cols = int(np.ceil(np.sqrt(n_methods)))
    n_rows = int(np.ceil(n_methods / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        constrained_layout=True
    )

    # Flatten axes if needed
    if n_methods == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each method
    for idx, method_name in enumerate(method_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        mean, std = results[method_name]

        # Imshow with shared scale
        im = ax.imshow(
            mean.T,
            cmap='viridis',
            vmin=vmin, vmax=vmax,
            origin='lower',
            aspect='auto',
            interpolation='nearest'
        )

        # Set tick labels: x=K1, y=K2
        ax.set_xticks(np.arange(len(k1_values)))
        ax.set_yticks(np.arange(len(k2_values)))

        # Format tick labels (log scale appearance, but at actual indices)
        k1_labels = [f'{k:.2g}' for k in k1_values]
        k2_labels = [f'{k:.2g}' for k in k2_values]
        ax.set_xticklabels(k1_labels, rotation=45, ha='right')
        ax.set_yticklabels(k2_labels)

        ax.set_xlabel('K1', fontsize=10)
        ax.set_ylabel('K2', fontsize=10)
        ax.set_title(method_name, fontsize=11, fontweight='bold')

        # Annotate cells (skip if NaN)
        for k1_idx in range(len(k1_values)):
            for k2_idx in range(len(k2_values)):
                val = mean[k1_idx, k2_idx]
                err = std[k1_idx, k2_idx]

                if np.isnan(val):
                    # Hatched gray for infeasible
                    rect = Rectangle(
                        (k1_idx - 0.5, k2_idx - 0.5), 1, 1,
                        fill=True, facecolor='lightgray',
                        edgecolor='black', linewidth=0.5,
                        hatch='///', alpha=0.7,
                        zorder=2
                    )
                    ax.add_patch(rect)
                    ax.text(
                        k1_idx, k2_idx, 'N/A',
                        ha='center', va='center',
                        fontsize=7, color='black'
                    )
                else:
                    # Annotate with value +/- error
                    text = f'{val:.2f}\n+/-{err:.2f}'
                    ax.text(
                        k1_idx, k2_idx, text,
                        ha='center', va='center',
                        fontsize=7, color='white'
                    )

        ax.grid(True, alpha=0.2, linewidth=0.5)

    # Hide unused subplots
    for idx in range(n_methods, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(metric.replace('_', ' ').title(), fontsize=12)

    # Add oracle reference as text annotation
    if ORACLE_METHOD in results:
        oracle_mean, oracle_std = results[ORACLE_METHOD]
        oracle_text = (
            f'{ORACLE_METHOD} (oracle)\n'
            f'MAE: {np.nanmean(oracle_mean):.3f} +/- {np.nanmean(oracle_std):.3f}'
        )
        fig.text(
            0.5, 0.02, oracle_text,
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    # Title
    title = (
        f'MAE vs (K1, K2) Heatmap — {encoding} (sigma={sigma:.3f})\n'
        f'Metric: {metric.replace("_", " ").title()}'
    )
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    # Save
    pdf_name = (
        f'headline_{encoding}_sigma_{sigma:.3f}_{metric}.pdf'
    )
    pdf_path = os.path.join(figures_dir, pdf_name)
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Headline figure saved: {pdf_path}')


def fig_disentanglement(results_enc1, results_enc2, k1_values, k2_values,
                        metric, figures_dir):
    """
    generate side-by-side heatmaps: Enc-1 (onehot_joint) vs Enc-2 (onehot_concat).

    layout:
      - 2 columns (one per encoding)
      - rows: classifier methods only

    each heatmap: same as headline (imshow, annotate, grid).

    args:
      results_enc1, results_enc2: dicts from load_summary_h5
      k1_values, k2_values: list of floats
      metric: 'pointwise_mae' or 'eldr_err'
      figures_dir: output directory

    side effects: creates disentanglement_{metric}.pdf

    note: raises warning if either encoding missing; returns early.
    """

    # Load style
    style_path = '/home/aviamala/dpe-submission/half-width.mplstyle'
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        plt.rcParams['font.size'] = 18

    # Filter to classifier methods only
    method_names = sorted([
        m for m in CLASSIFIER_METHODS
        if m in results_enc1 or m in results_enc2
    ])

    if len(method_names) == 0:
        warnings.warn(
            'No classifier methods found in encodings; skipping disentanglement panel'
        )
        return

    # Shared color scale across both encodings
    all_results = {**results_enc1, **results_enc2}
    vmin, vmax = get_shared_colorscale(all_results)

    n_rows = len(method_names)
    n_cols = 2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.0 * n_rows),
        constrained_layout=True
    )

    # Handle single-row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each method x encoding pair
    for method_idx, method_name in enumerate(method_names):
        for enc_idx, (results, enc_label) in enumerate([
            (results_enc1, 'Enc-1\n(onehot_joint)'),
            (results_enc2, 'Enc-2\n(onehot_concat)'),
        ]):
            ax = axes[method_idx, enc_idx]

            if method_name not in results:
                ax.text(
                    0.5, 0.5, f'{method_name}\nno results',
                    ha='center', va='center', fontsize=10,
                    transform=ax.transAxes
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            mean, std = results[method_name]

            # Imshow
            im = ax.imshow(
                mean.T,
                cmap='viridis',
                vmin=vmin, vmax=vmax,
                origin='lower',
                aspect='auto',
                interpolation='nearest'
            )

            # Ticks
            ax.set_xticks(np.arange(len(k1_values)))
            ax.set_yticks(np.arange(len(k2_values)))

            k1_labels = [f'{k:.2g}' for k in k1_values]
            k2_labels = [f'{k:.2g}' for k in k2_values]
            ax.set_xticklabels(k1_labels, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(k2_labels, fontsize=9)

            # Labels
            if enc_idx == 0:
                ax.set_ylabel('K2', fontsize=10)
            if method_idx == n_rows - 1:
                ax.set_xlabel('K1', fontsize=10)

            # Title: method on left, encoding on top
            if method_idx == 0:
                ax.set_title(enc_label, fontsize=11, fontweight='bold')
            if enc_idx == 0:
                ax.text(
                    -0.5, 0.5, method_name,
                    transform=ax.transAxes, fontsize=10, fontweight='bold',
                    ha='right', va='center'
                )

            # Annotate
            for k1_idx in range(len(k1_values)):
                for k2_idx in range(len(k2_values)):
                    val = mean[k1_idx, k2_idx]
                    err = std[k1_idx, k2_idx]

                    if np.isnan(val):
                        rect = Rectangle(
                            (k1_idx - 0.5, k2_idx - 0.5), 1, 1,
                            fill=True, facecolor='lightgray',
                            edgecolor='black', linewidth=0.5,
                            hatch='///', alpha=0.7, zorder=2
                        )
                        ax.add_patch(rect)
                        ax.text(
                            k1_idx, k2_idx, 'N/A',
                            ha='center', va='center',
                            fontsize=6, color='black'
                        )
                    else:
                        text = f'{val:.2f}\n+/-{err:.2f}'
                        ax.text(
                            k1_idx, k2_idx, text,
                            ha='center', va='center',
                            fontsize=6, color='white'
                        )

            ax.grid(True, alpha=0.2, linewidth=0.5)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(metric.replace('_', ' ').title(), fontsize=10)

    # Title
    fig.suptitle(
        f'Disentanglement Panel: Classifier Methods on Enc-1 vs Enc-2\n'
        f'Metric: {metric.replace("_", " ").title()}',
        fontsize=12, fontweight='bold', y=0.995
    )

    # Save
    pdf_name = f'disentanglement_{metric}.pdf'
    pdf_path = os.path.join(figures_dir, pdf_name)
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Disentanglement figure saved: {pdf_path}')


def fig_sigma_sensitivity(h5_dir, config, metric, figures_dir):
    """
    generate line plots: MAE vs sigma at mid and corner cells.

    layout:
      - 2 columns (mid cell, corner cell from config['sigma_sweep']['cells'])
      - 1 row
      - x-axis: sigma (log scale)
      - y-axis: MAE (linear scale)
      - lines: one per method with error bars (+-SE over seeds)

    args:
      h5_dir: base directory containing sigma_{s:.3f}/ subdirectories
      config: full config dict; uses sigma_sweep.sigmas, sigma_sweep.cells
      metric: 'pointwise_mae' or 'eldr_err'
      figures_dir: output directory

    side effects: creates sigma_sensitivity_{metric}.pdf

    logic:
      - for each (sigma, cell_idx) pair:
        - load {h5_dir}/sigma_{s:.3f}/summary.h5
        - extract mean[cell_k1_idx, cell_k2_idx] and std
      - plot as line with error bar
      - legend below
      - shared y-axis limits

    note: if sigma_sweep disabled or no sigmas found, log warning and skip.
    """

    # Load style
    style_path = '/home/aviamala/dpe-submission/half-width.mplstyle'
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        plt.rcParams['font.size'] = 18

    # Extract sigma sweep config
    sigma_sweep_cfg = config.get('sigma_sweep', {})
    if not sigma_sweep_cfg.get('enabled', False):
        warnings.warn('sigma_sweep disabled in config; skipping panel')
        return

    sigmas = sigma_sweep_cfg.get('sigmas', [])
    cells = sigma_sweep_cfg.get('cells', [])

    if len(sigmas) == 0 or len(cells) == 0:
        warnings.warn('sigma_sweep has no sigmas or cells; skipping panel')
        return

    k1_values = config['kl_targets']['k1_values']
    k2_values = config['kl_targets']['k2_values']

    # Prepare cell indices from (k1_val, k2_val) tuples
    cell_indices = []
    for cell in cells:
        try:
            k1_idx = k1_values.index(cell[0])
            k2_idx = k2_values.index(cell[1])
            cell_indices.append((k1_idx, k2_idx))
        except ValueError:
            warnings.warn(
                f'Cell {cell} not found in k1/k2 grid; skipping'
            )

    if len(cell_indices) == 0:
        warnings.warn('No valid cells found; skipping sigma sensitivity panel')
        return

    # Load data for all (sigma, cell) pairs
    sigma_data = {}

    for sigma in sigmas:
        sigma_dir = os.path.join(h5_dir, f'sigma_{sigma:.3f}')
        h5_path = os.path.join(sigma_dir, 'summary.h5')

        if not os.path.exists(h5_path):
            warnings.warn(f'summary.h5 missing for sigma={sigma:.3f}; skipping')
            continue

        results = load_summary_h5(h5_path, metric, 'gaussian_blob',
                                   k1_values, k2_values)
        sigma_data[sigma] = results

    if len(sigma_data) == 0:
        warnings.warn('No sigma data loaded; skipping panel')
        return

    # Plot
    fig, axes = plt.subplots(
        1, len(cell_indices),
        figsize=(4 * len(cell_indices), 3.5),
        constrained_layout=True
    )

    if len(cell_indices) == 1:
        axes = [axes]

    # For each cell
    for cell_plot_idx, (k1_idx, k2_idx) in enumerate(cell_indices):
        ax = axes[cell_plot_idx]

        # Gather methods across all sigmas
        all_methods = set()
        for results in sigma_data.values():
            all_methods.update(results.keys())

        method_names = sorted([m for m in all_methods if m != ORACLE_METHOD])

        # Plot each method
        for method_name in method_names:
            sigmas_list = sorted(sigma_data.keys())
            means = []
            stds = []

            for sigma in sigmas_list:
                results = sigma_data[sigma]
                if method_name in results:
                    mean, std = results[method_name]
                    means.append(mean[k1_idx, k2_idx])
                    stds.append(std[k1_idx, k2_idx])
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            # Plot with error bars
            color = METHOD_COLORS.get(method_name, '#cccccc')
            ax.errorbar(
                sigmas_list, means, yerr=stds,
                marker='o', label=method_name,
                color=color, linewidth=1.5, markersize=5,
                capsize=3, capthick=1.0, alpha=0.7
            )

        # Axes
        ax.set_xlabel('sigma (log scale)', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Title: cell coordinates
        k1_val = k1_values[k1_idx]
        k2_val = k2_values[k2_idx]
        ax.set_title(
            f'Cell: K1={k1_val:.2g}, K2={k2_val:.2g}',
            fontsize=11, fontweight='bold'
        )

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=min(5, len(method_names)),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.05)
    )

    # Title
    fig.suptitle(
        f'Sigma Sensitivity Analysis\n'
        f'Metric: {metric.replace("_", " ").title()}',
        fontsize=12, fontweight='bold'
    )

    # Save
    pdf_name = f'sigma_sensitivity_{metric}.pdf'
    pdf_path = os.path.join(figures_dir, pdf_name)
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Sigma sensitivity figure saved: {pdf_path}')


def main():
    """
    orchestrate all three figures.

    workflow:
      1. parse CLI (config, encoding, metric)
      2. load config.yaml
      3. ensure figures_dir exists
      4. load summary.h5 for primary encoding (gaussian_blob)
      5. generate headline figure
      6. if metric == pointwise_mae and encoding == gaussian_blob:
         - load Enc-1 and Enc-2 summaries
         - generate disentanglement figure
      7. if sigma_sweep enabled:
         - generate sigma sensitivity figure
      8. print completion message
    """
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    processed_results_dir = config['processed_results_dir']
    figures_dir = config['figures_dir']
    k1_values = config['kl_targets']['k1_values']
    k2_values = config['kl_targets']['k2_values']

    # Extract sigma from encoding config
    sigma = config['encoding'].get('sigma', 0.2)

    # Ensure output directory
    os.makedirs(figures_dir, exist_ok=True)

    # path-prefix helper: onehot encodings live under sigma_na/, blob/flow under sigma_{:.3f}/
    def encoding_subdir(base, encoding_type, sigma):
        if encoding_type.startswith("onehot"):
            return os.path.join(base, encoding_type, "sigma_na")
        return os.path.join(base, encoding_type, f"sigma_{sigma:.3f}")

    # Load summary for primary encoding
    h5_path = os.path.join(
        encoding_subdir(processed_results_dir, args.encoding, sigma),
        'summary.h5'
    )

    try:
        results = load_summary_h5(
            h5_path, args.metric, args.encoding,
            k1_values, k2_values
        )
    except FileNotFoundError as e:
        print(f'Error: {e}')
        return
    except KeyError as e:
        print(f'Error: {e}')
        return

    # Generate headline figure
    fig_headline(
        results, k1_values, k2_values,
        args.metric, args.encoding, sigma,
        figures_dir
    )

    # Generate disentanglement figure (for gaussian_blob encoding only)
    if args.encoding == 'gaussian_blob' and args.metric == 'pointwise_mae':
        print('Loading Enc-1 and Enc-2 for disentanglement panel...')

        try:
            results_enc1 = load_summary_h5(
                os.path.join(
                    encoding_subdir(processed_results_dir, 'onehot_joint', sigma),
                    'summary.h5'
                ),
                args.metric, 'onehot_joint',
                k1_values, k2_values
            )
            results_enc2 = load_summary_h5(
                os.path.join(
                    encoding_subdir(processed_results_dir, 'onehot_concat', sigma),
                    'summary.h5'
                ),
                args.metric, 'onehot_concat',
                k1_values, k2_values
            )

            fig_disentanglement(
                results_enc1, results_enc2,
                k1_values, k2_values,
                args.metric,
                figures_dir
            )
        except FileNotFoundError as e:
            warnings.warn(
                f'Disentanglement panel skipped: {e}'
            )

    # Generate sigma sensitivity figure
    if args.encoding == 'gaussian_blob':
        h5_base_dir = os.path.join(
            processed_results_dir, args.encoding
        )
        fig_sigma_sensitivity(
            h5_base_dir, config, args.metric,
            figures_dir
        )

    print(f'All figures saved to: {figures_dir}')


if __name__ == '__main__':
    main()
