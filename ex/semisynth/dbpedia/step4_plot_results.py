"""step4 plotting for dbpedia_eldr.

family-grouped box plots (hue = method/variant, lightness = alpha hardness) for
the three uniform metrics, via ex.utils.family_boxplot. reads the flat per-cell
summary.h5 from step3 and reshapes [n_cells] -> [n_alpha, n_pairs] (cell order is
alpha-major: cell i -> alpha = i // n_pairs).
"""
import argparse
import os

import h5py
import yaml

from ex.utils.family_boxplot import plot_family_boxplot

CONFIG_PATH = 'ex/semisynth/dbpedia/config.yaml'

# metric -> (ylabel, yscale). regret is normalized in [0,1] -> linear.
METRICS = {
    'eldr_abs_err': ('ELDR abs error', 'log'),
    'mae_train':    ('MAE (train p*)', 'log'),
    'regret':       ('ELDR regret', 'linear'),
}


def load_stratified(h5_path, metric, methods, n_hard):
    """flat [n_cells] per method -> [n_hard, n_pairs] via hardness-major reshape."""
    data = {}
    with h5py.File(h5_path, 'r') as f:
        for m in methods:
            key = f'{metric}_{m}'
            if key not in f:
                continue
            arr = f[key][:]
            if arr.shape[0] % n_hard != 0:
                raise ValueError(f'{key}: n_cells {arr.shape[0]} not divisible by n_hard {n_hard}')
            data[m] = arr.reshape(n_hard, -1)
    return data


def main():
    p = argparse.ArgumentParser(description='plot dbpedia eldr estimation results')
    p.add_argument('--config', default=CONFIG_PATH)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    figures_dir = config['figures_dir']
    processed = config['processed_results_dir']
    alphas = config['alphas']
    n_hard = len(alphas)
    os.makedirs(figures_dir, exist_ok=True)

    h5_path = os.path.join(processed, 'summary.h5')
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f'summary.h5 not found at {h5_path}; run step3 first.')
    with h5py.File(h5_path, 'r') as f:
        methods = [m.decode() if isinstance(m, bytes) else m for m in f.attrs['methods']]

    for metric, (ylabel, yscale) in METRICS.items():
        data = load_stratified(h5_path, metric, methods, n_hard)
        plot_family_boxplot(
            data, alphas, sweep_name='alpha', ylabel=ylabel,
            out_dir=figures_dir, prefix=f'dbpedia_{metric}', yscale=yscale,
        )

    print(f'dbpedia step4: family box plots for {list(METRICS)} saved to {figures_dir}')


if __name__ == '__main__':
    main()
