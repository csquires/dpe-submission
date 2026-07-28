"""
Step 4: Plot Results for occupancy SMODICE ELDR estimation.

Family-grouped box plots (hue = method/variant, lightness = K1 hardness) for the
three metrics (pointwise_mae, eldr_err, regret), via ex.utils.family_boxplot.
reads /{metric}_{method}_seed_values [n_k1, n_seeds] (nan-padded) written by step3.
"""
import matplotlib
matplotlib.use('Agg')
import argparse
import os

import h5py
import numpy as np
import yaml

from ex.utils.family_boxplot import plot_family_boxplot


# occupancy uses the classifier base name "MDRE". VFMOrthros is dropped (not a
# family here); any method absent from FAMILIES is simply not drawn.
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
    'pointwise_mae': ('Pointwise LDR MAE', 'log'),
    'eldr_err':      ('ELDR Error',        'log'),
    'regret':        ('ELDR regret',       'linear'),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='plot occupancy eldr estimation results')
    p.add_argument('--config', default='ex/synth/occupancy/config.yaml')
    return p.parse_args()


def _encoding_subdir(config) -> str:
    """mirror step2_adapter/step3: onehot -> sigma_na, else sigma_{sigma:.3f}."""
    enc = config['encoding']['type']
    if enc.startswith('onehot'):
        return os.path.join(enc, 'sigma_na')
    return os.path.join(enc, f"sigma_{config['encoding']['sigma']:.3f}")


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    figures_dir = config['figures_dir']
    k1_values = config['kl_targets']['k1_values']
    os.makedirs(figures_dir, exist_ok=True)

    h5_path = os.path.join(config['processed_results_dir'], _encoding_subdir(config), 'summary.h5')
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f'summary.h5 not found: {h5_path}; run step3 first.')

    for metric, (ylabel, yscale) in METRICS.items():
        suffix = '_seed_values'
        with h5py.File(h5_path, 'r') as f:
            data = {
                k[len(metric) + 1:-len(suffix)]: np.atleast_2d(f[k][:])
                for k in f.keys()
                if k.startswith(f'{metric}_') and k.endswith(suffix)
            }
        plot_family_boxplot(
            data, k1_values, sweep_name='K1', ylabel=ylabel,
            out_dir=figures_dir, prefix=f'occupancy_{metric}',
            families=FAMILIES, yscale=yscale,
        )

    print(f'occupancy step4: family box plots for {list(METRICS)} saved to {figures_dir}')


if __name__ == '__main__':
    main()
