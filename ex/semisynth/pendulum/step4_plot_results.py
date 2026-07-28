"""
Step 4: Plot Results for Pendulum ELDR Estimation

Family-grouped box plots (hue = method/variant, lightness = K1 hardness) for the
three uniform metrics, via ex.utils.family_boxplot. Reads the flat per-cell
summary.h5 written by step3 and reshapes each metric [n_cells] -> [n_k1, n_seeds]
before plotting. cell order is hardness-major (cell i -> k1 = i // n_seeds), so a
plain reshape recovers the stratification.
"""
import matplotlib
matplotlib.use('Agg')
import argparse
import os

import h5py
import numpy as np
import yaml

from ex.utils.family_boxplot import plot_family_boxplot


# pendulum uses the classifier base name "MDRE" (not "MDRE_15" as mnist/dbpedia).
FAMILIES = [
    ("BDRE",          []),
    ("MDRE",          ["TriangularMDRE"]),
    ("MultiHeadTDRE", ["MultiHeadTriangularTDRE"]),
    ("TSM",           ["TriangularTSM"]),
    ("CTSM",          ["TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3"]),
    ("VFM",           ["TriangularVFM_V1", "TriangularVFM_V2", "TriangularVFM_V3"]),
    ("FMDRE",         ["TriangularFMDRE"]),
]

# metric -> (ylabel, yscale). regret is cross-method normalized in [0,1] -> linear.
METRICS = {
    'eldr_abs_err': ('ELDR abs error', 'log'),
    'mae_train':    ('MAE (train p*)', 'log'),
    'regret':       ('ELDR regret',    'linear'),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='plot pendulum eldr estimation results')
    p.add_argument('--config', default='ex/semisynth/pendulum/config.yaml')
    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    figures_dir = os.path.expandvars(config['figures_dir'])
    processed = os.path.expandvars(config['processed_results_dir'])
    k1_values = config['kl_targets']['k1_values']
    n_hard = len(k1_values)
    os.makedirs(figures_dir, exist_ok=True)

    h5_path = os.path.join(processed, 'summary.h5')
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f'summary.h5 not found at {h5_path}; run step3 first.')
    with h5py.File(h5_path, 'r') as f:
        methods = [m.decode() if isinstance(m, bytes) else m for m in f.attrs['methods']]

    for metric, (ylabel, yscale) in METRICS.items():
        data = load_stratified(h5_path, metric, methods, n_hard)
        plot_family_boxplot(
            data, k1_values, sweep_name='K1', ylabel=ylabel,
            out_dir=figures_dir, prefix=f'pendulum_{metric}',
            families=FAMILIES, yscale=yscale,
        )

    print(f'pendulum step4: family box plots for {list(METRICS)} saved to {figures_dir}')


if __name__ == '__main__':
    main()
