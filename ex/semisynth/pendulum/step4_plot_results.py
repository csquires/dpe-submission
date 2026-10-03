"""
Step 4: Plot Results for Pendulum ELDR Estimation

Family-grouped box plots (hue = method/variant, lightness = K1 hardness) for the
three uniform metrics, via ex.utils.family_boxplot. Reads the flat per-cell
summary.h5 written by step3 and reshapes each metric [n_cells] -> [n_k1, n_seeds]
before plotting. cell order is hardness-major (cell i -> k1 = i // n_seeds), so a
plain reshape recovers the stratification.

    summary.h5 [n_cells] --> drop methods --> recompute regret
                         --> reshape [n_k1, n_seeds] --> plot_family_boxplot

K1 axis labels are the realized values measured by step0d, read from the datagen
manifest next to the data. seeds reserved for hpo train/holdout are nan in the
summary and drop out of each box.
"""
import matplotlib
matplotlib.use('Agg')
import argparse
import os

import h5py
import numpy as np
import yaml

from ex.utils.family_boxplot import plot_family_boxplot


# base method names as the summary spells them; display_name maps MDRE_15 -> MDRE
# and MultiHeadTDRE -> TDRE for every figure label and table row.
FAMILIES = [
    ("BDRE",          []),
    ("MDRE_15",       ["TriangularMDRE"]),
    ("MultiHeadTDRE", ["MultiHeadTriangularTDRE"]),
    ("TSM",           ["TriangularTSM"]),
    ("CTSM",          ["TriangularCTSM_V1", "TriangularCTSM_V2", "TriangularCTSM_V3"]),
    ("VFM",           ["TriangularVFM_V1", "TriangularVFM_V2", "TriangularVFM_V3"]),
    ("FMDRE",         ["TriangularFMDRE"]),
]

# metric -> plot options. regret is cross-method normalized in [0,1] -> linear.
# both log panels are clipped: a handful of TriangularTSM fliers otherwise claim
# most of the height (66% of the MAE panel, 30% of the eldr panel). eldr keeps a
# ~5 decade span after the clip, so it also gets a taller canvas.
METRICS = {
    'eldr_abs_err': dict(ylabel='Abs. ELDR err.', yscale='log',
                         clip_to_whiskers=True, height=5.4),
    'mae_train':    dict(ylabel='Pointwise LDR MAE', yscale='log',
                         clip_to_whiskers=True),
    'regret':       dict(ylabel='Rel. ELDR regret', yscale='linear'),
}

# TSM is degenerate on most cells and VFMOrthros has no winner, so neither is
# drawn, and neither joins the regret pool. the per-method metrics still report
# them in their tables, since those values stand on their own. regret has no
# such row: it is relative, so a method outside the pool has no regret to quote.
# pass --drop '' to put every method back.
DROP_DEFAULT = 'TSM,VFMOrthros'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='plot pendulum eldr estimation results')
    p.add_argument('--config', default='ex/semisynth/pendulum/config.yaml')
    p.add_argument('--drop', default=DROP_DEFAULT,
                   help='comma-separated methods to leave out of the panels and '
                        'the regret pool; per-method tables still report them')
    return p.parse_args()


def k1_labels(config: dict) -> list[float]:
    """realized K1 per stratum, ordered by stratum_label = hardness-major order."""
    manifest = os.path.join(os.path.expandvars(config['data_dir']), 'alphas_chosen.yaml')
    with open(manifest) as f:
        strata = yaml.safe_load(f)['strata']
    strata.sort(key=lambda s: s['stratum_label'])
    return [round(float(s['K1_realized_flow']), 1) for s in strata]


def regret(err: dict) -> dict:
    """per-cell regret over the given method pool, as step3 defines it.

    (err - best) / (worst - best) per cell, best and worst over the pool. an
    exact tie gives 0. a cell with less than 2 finite methods gives nan. regret
    is relative, so it must be recomputed whenever the pool changes.
    """
    names = list(err)
    mat = np.stack([err[m] for m in names], axis=0).astype(np.float64)
    out = np.full_like(mat, np.nan)
    for i in range(mat.shape[1]):
        col = mat[:, i]
        finite = np.isfinite(col)
        if finite.sum() < 2:
            continue
        lo, hi = np.nanmin(col), np.nanmax(col)
        span = hi - lo
        out[:, i] = np.where(finite, 0.0, np.nan) if span == 0 \
            else np.where(finite, (col - lo) / span, np.nan)
    return {m: out[j] for j, m in enumerate(names)}


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    figures_dir = os.path.expandvars(config['figures_dir'])
    processed = os.path.expandvars(config['processed_results_dir'])
    k1_values = k1_labels(config)
    n_hard = len(k1_values)
    os.makedirs(figures_dir, exist_ok=True)

    drop = {m for m in args.drop.split(',') if m}
    h5_path = os.path.join(processed, 'summary.h5')
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f'summary.h5 not found at {h5_path}; run step3 first.')

    with h5py.File(h5_path, 'r') as f:
        methods = [m.decode() if isinstance(m, bytes) else m for m in f.attrs['methods']]
        flat = {k: {m: f[f'{k}_{m}'][:] for m in methods} for k in METRICS}
    shown = [m for m in methods if m not in drop]
    # regret is relative: only the drawn methods set the per-cell best and worst.
    flat['regret'] = regret({m: flat['eldr_abs_err'][m] for m in shown})

    for metric, opts in METRICS.items():
        # per-method metrics keep a table row for every method; regret cannot,
        # so its table follows its pool.
        full = {m: v.reshape(n_hard, -1) for m, v in flat[metric].items()}
        plot_family_boxplot(
            {m: full[m] for m in shown}, k1_values, sweep_name='K1',
            out_dir=figures_dir, prefix=f'pendulum_{metric}',
            families=FAMILIES, table_data=full, **opts,
        )

    print(f'pendulum step4: family box plots for {list(METRICS)} saved to {figures_dir}')
    print(f'  drawn={len(shown)} of {len(methods)}  K1={k1_values}  '
          f'excluded={sorted(drop & set(methods)) or "none"} '
          f'(kept in the per-method tables)')


if __name__ == '__main__':
    main()
