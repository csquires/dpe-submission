"""
Step 4: Plot Results for Pendulum ELDR Estimation

Generate per-family line figures (K1 vs pointwise MAE with +/- SE band) using
ex.utils.plot_style, plus the integrated ELDR error bar chart. The bar chart
keeps its original styling; the line plot is split across METHOD_GROUPS.
"""
import matplotlib
matplotlib.use('Agg')
import yaml
import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import warnings

from ex.utils.plot_style import (
    apply as apply_style,
    style_for,
    METHOD_GROUPS,
    GROUP_LABEL,
    ERROR_BAND_ALPHA,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns argparse.Namespace with:
      - config: path to config.yaml (default 'ex/semisynth/pendulum/config.yaml')
    """
    parser = argparse.ArgumentParser(description='plot pendulum eldr estimation results')
    parser.add_argument(
        '--config',
        default='ex/semisynth/pendulum/config.yaml',
        help='path to config yaml',
    )
    return parser.parse_args()


def load_summary_h5(path: str) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict]:
    """
    Load HDF5 summary from step3 (flat key schema, no _mean suffix).

    **Input schema (1D grid):**
    - /k1_values: float32 [G_k1]
    - /pointwise_mae_{method}: float32 [G_k1]
    - /pointwise_mae_se_{method}: float32 [G_k1]
    - /pointwise_mae_n_{method}: int32 [G_k1] (optional)
    - /eldr_err_{method}: float32 [G_k1]
    - /eldr_err_se_{method}: float32 [G_k1]
    - /eldr_err_n_{method}: int32 [G_k1] (optional)
    - Root attrs: beta_value (float), beta_count (int)

    Returns:
      - k1_values: [G_k1] float32 array
      - results: dict[method -> {mae_mean, mae_se, eldr_mean, eldr_se}]
      - attrs: dict with 'beta_value', 'beta_count'

    Raises:
      - FileNotFoundError: if path does not exist
      - KeyError: if k1_values missing
      - ValueError: if array shapes inconsistent
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'mae_summary.h5 not found at {path}')

    with h5py.File(path, 'r') as f:
        if '/k1_values' not in f:
            raise KeyError('k1_values dataset missing in HDF5 file')

        k1_values = f['/k1_values'][:]  # [G_k1]
        if len(k1_values) == 0:
            raise ValueError('k1_values array is empty')

        # initialize results and attrs
        results = {}
        attrs = {
            'beta_value': f.attrs.get('beta_value', np.nan),
            'beta_count': f.attrs.get('beta_count', 1),
        }

        # scan all keys for pointwise_mae_* and eldr_err_* datasets
        for key in f.keys():
            if key.startswith('pointwise_mae_') and not key.endswith('_se') and not key.endswith('_n'):
                # extract method name from pointwise_mae_{method}
                method = key[len('pointwise_mae_'):]
                mae_key = f'pointwise_mae_{method}'
                mae_se_key = f'pointwise_mae_se_{method}'

                if mae_se_key not in f:
                    warnings.warn(f'{mae_se_key} missing for {method}; skipping')
                    continue

                mae_mean = f[mae_key][:]  # [G_k1]
                mae_se = f[mae_se_key][:]  # [G_k1]

                if mae_mean.shape != k1_values.shape:
                    raise ValueError(f'{method} mae shape {mae_mean.shape} != k1_values shape {k1_values.shape}')
                if mae_se.shape != k1_values.shape:
                    raise ValueError(f'{method} mae_se shape {mae_se.shape} != k1_values shape {k1_values.shape}')

                # check for suspicious MAE values
                if np.any((mae_mean < 1e-6) & ~np.isnan(mae_mean)):
                    warnings.warn(f'{method}: MAE < 1e-6 detected; possible bug in step3')

                if np.all(np.isnan(mae_mean)):
                    warnings.warn(f'{method}: all-NaN mae_mean; will skip plotting')

                if np.all(np.isnan(mae_se)):
                    warnings.warn(f'{method}: all-NaN mae_se (seeds < 2); will plot line without band')

                if method not in results:
                    results[method] = {}
                results[method]['mae_mean'] = mae_mean
                results[method]['mae_se'] = mae_se

            elif key.startswith('eldr_err_') and not key.endswith('_se') and not key.endswith('_n'):
                # extract method from eldr_err_{method}
                method = key[len('eldr_err_'):]
                eldr_key = f'eldr_err_{method}'
                eldr_se_key = f'eldr_err_se_{method}'

                if eldr_se_key not in f:
                    warnings.warn(f'{eldr_se_key} missing for {method}; skipping')
                    continue

                eldr_mean = f[eldr_key][:]  # [G_k1]
                eldr_se = f[eldr_se_key][:]  # [G_k1]

                if eldr_mean.shape != k1_values.shape:
                    raise ValueError(f'{method} eldr_err shape {eldr_mean.shape} != k1_values shape {k1_values.shape}')
                if eldr_se.shape != k1_values.shape:
                    raise ValueError(f'{method} eldr_err_se shape {eldr_se.shape} != k1_values shape {k1_values.shape}')

                if np.all(np.isnan(eldr_mean)):
                    warnings.warn(f'{method}: all-NaN eldr_err_mean; will skip from bar chart')

                if method not in results:
                    results[method] = {}
                results[method]['eldr_mean'] = eldr_mean
                results[method]['eldr_se'] = eldr_se

    return k1_values, results, attrs


def plot_k1_vs_mae(
    k1_values: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    beta_value: float,
    out_dir: str
) -> None:
    """one figure per METHOD_GROUPS family, log-log K1 vs MAE with +/- SE band.
    shared y-range across all family figures so they are visually comparable
    side-by-side.
    """
    apply_style()

    # compute shared y range across all methods present (log scale, floor 1e-4).
    y_max = 0.0
    for m, d in results.items():
        if 'mae_mean' not in d: continue
        upper = np.nanmax(d['mae_mean'] + d['mae_se'])
        if np.isfinite(upper):
            y_max = max(y_max, float(upper))
    y_max *= 1.2 if y_max > 0 else 1.0
    y_floor = 1e-4

    os.makedirs(out_dir, exist_ok=True)
    for group, members in METHOD_GROUPS.items():
        present = [m for m in members if m in results and 'mae_mean' in results[m]]
        if not present:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in present:
            data = results[method]
            mae_mean = data['mae_mean']
            mae_se = data['mae_se']
            if np.all(np.isnan(mae_mean)):
                continue
            mae_floored = np.maximum(mae_mean, y_floor)
            kw = style_for(method)
            ax.plot(k1_values, mae_floored, label=method, **kw)
            if not np.all(np.isnan(mae_se)):
                lo = np.maximum(mae_floored - mae_se, y_floor)
                hi = mae_floored + mae_se
                ax.fill_between(k1_values, lo, hi,
                                color=kw["color"], alpha=ERROR_BAND_ALPHA, linewidth=0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('K1')
        ax.set_ylabel('Pointwise LDR MAE')
        ax.set_title(f'{GROUP_LABEL[group]} (beta = {beta_value})')
        ax.set_ylim(y_floor, y_max)
        ax.legend(loc='best')
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            path = os.path.join(out_dir, f'k1_vs_mae_{group}.{ext}')
            fig.savefig(path)
        print(f"saved k1_vs_mae_{group}.{{pdf,png}}")
        plt.close(fig)


def plot_eldr_err_bars(
    results: dict[str, dict[str, np.ndarray]],
    beta_value: float,
    out_dir: str
) -> None:
    """
    Plot integrated ELDR error: bar chart (method × error, with error bars).

    Emits {out_dir}/eldr_err_bars.pdf and .png

    composition:
      - x-axis: method names (sorted, rotated 45°)
      - y-axis: ELDR error (mean of eldr_err_{method} across K1)
      - error bars: SE (mean of eldr_err_se_{method} across K1)
      - title: f"Integrated ELDR Error by Method (beta = {beta_value})"
      - colors: METHOD_COLORS palette
      - dpi: 300
      - figsize: (10, 6)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # extract and aggregate per method
    methods = sorted([m for m in results.keys() if 'eldr_mean' in results[m]])
    eldr_means = []
    eldr_ses = []

    for method in methods:
        data = results[method]
        eldr_mean = data['eldr_mean']  # [G_k1]
        eldr_se = data['eldr_se']  # [G_k1]

        # check for all-NaN; warn and skip if so
        if np.all(np.isnan(eldr_mean)):
            warnings.warn(f'{method}: all-NaN eldr_err_mean; skipping from bar chart')
            continue

        # aggregate across K1 via nanmean
        eldr_agg_mean = np.nanmean(eldr_mean)  # scalar
        eldr_agg_se = np.nanmean(eldr_se)  # scalar

        eldr_means.append(eldr_agg_mean)
        eldr_ses.append(eldr_agg_se)

    # filter out methods with all-NaN
    methods = [m for i, m in enumerate(methods) if i < len(eldr_means)]

    # create bar plot
    x_pos = np.arange(len(methods))
    for i, method in enumerate(methods):
        color = style_for(method)["color"]
        ax.bar(x_pos[i], eldr_means[i], yerr=eldr_ses[i],
               color=color, capsize=5, alpha=0.8, edgecolor='black', linewidth=1)

    # configure axes
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('ELDR Error', fontsize=12)
    ax.set_title(f'Integrated ELDR Error by Method (beta = {beta_value})', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    # save figures
    pdf_path = os.path.join(out_dir, 'eldr_err_bars.pdf')
    png_path = os.path.join(out_dir, 'eldr_err_bars.png')

    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Figure saved to: {pdf_path}')
    print(f'PNG saved to: {png_path}')


def main() -> None:
    """
    Orchestrate: parse args → load config → load h5 → plot two figures.

    process:
      1. parse command-line args
      2. load config (yaml)
      3. extract processed_results_dir and figures_dir
      4. ensure figures_dir exists
      5. load HDF5 summary
      6. extract beta_value from attrs (warn if beta_count > 1)
      7. emit both figures
      8. print summary
    """
    args = parse_args()

    # load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    processed_results_dir = config['processed_results_dir']
    figures_dir = config['figures_dir']

    # ensure output directory exists
    os.makedirs(figures_dir, exist_ok=True)

    # load HDF5
    h5_path = os.path.join(processed_results_dir, 'mae_summary.h5')
    k1_values, results, attrs = load_summary_h5(h5_path)

    # extract beta_value
    beta_value = attrs.get('beta_value', 'unknown')
    beta_count = attrs.get('beta_count', 1)

    if beta_count > 1:
        warnings.warn('future extension: multi-K2 not supported; using first K2')

    # emit figures
    plot_k1_vs_mae(k1_values, results, beta_value, figures_dir)
    plot_eldr_err_bars(results, beta_value, figures_dir)

    print(f'Both figures saved to {figures_dir}')


if __name__ == '__main__':
    main()
