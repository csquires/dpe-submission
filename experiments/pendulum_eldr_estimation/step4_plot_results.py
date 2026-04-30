"""
Step 4: Plot Results for Pendulum ELDR Estimation

Plots alpha vs MAE with translucent error bands for all methods.
Loads results from mae_summary.h5 and creates PDF and PNG outputs.
"""
import yaml
import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns


# method colors: canonical palette for pendulum continuous-control methods
# renamed (V1 variants of historical names) + core unchanged + new methods
METHODS = {
    # four core methods unchanged
    'TriangularMDRE': 'blue',
    'MultiHeadTriangularTDRE': 'orange',
    'TSM': 'red',
    'CTSM': 'purple',

    # renamed entries (old names with _V1 suffix)
    'TriangularCTSM_V1': 'brown',
    'TriangularVFM_V1': 'green',
    'MDRE_15': 'cyan',

    # new methods (distinct colors from extended palette)
    'TriangularTSM': 'magenta',
    'FMDRE': 'teal',
    'FMDRE_S2': 'olive',
    'TriangularFMDRE': 'maroon',
    'VFM': 'navy',
    'TDRE_5': 'lime',
    'TriangularCTSM_V2': 'aquamarine',
    'TriangularCTSM_V3': 'coral',
    'TriangularVFM_V2': 'darkgreen',
    'TriangularVFM_V3': 'goldenrod',
}

FIGURE_SIZE = (8, 5)
FONT_SIZE = 12
ERROR_BAND_ALPHA = 0.2


def parse_args():
    """
    parse command-line arguments for config path.

    returns: argparse.Namespace with config attribute.
    """
    parser = argparse.ArgumentParser(description='plot pendulum eldr estimation results')
    parser.add_argument(
        '--config',
        default='experiments/pendulum_eldr_estimation/config.yaml',
        help='path to config yaml',
    )
    return parser.parse_args()


def main():
    """
    Load results from HDF5 and create alpha vs MAE plot.

    Process:
    - load config (yaml)
    - ensure output directory exists
    - load results (h5py, parse alphas and method results)
    - configure plot style (seaborn, rcParams)
    - create figure
    - plot all methods (line + fill_between for error bands)
    - configure axes (log scale, labels, legend, grid)
    - save figures (pdf and png)
    """
    args = parse_args()

    # load config
    with open(args.config, 'r') as f:
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


if __name__ == '__main__':
    main()
