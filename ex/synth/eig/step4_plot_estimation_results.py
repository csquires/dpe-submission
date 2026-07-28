"""plot eig estimation results: one figure per metric, stratifications side by side.

each figure is a single row of method-group panels (vfm_fmdre / tsm_ctsm / cls)
via ex.utils.group_panels, plus sibling {stem}.md/.tex tables of the plotted
values. metrics:
  regret   -- normalized EIG regret, median-of-medians point + bootstrap IQR band
  eldr_err -- absolute |est - true| EIG error, mean +/- SE band
pointwise LDR MAE is not available for eig: the raw campaign stored only the
integrated est_eigs per cell, not per-sample LDR estimates.
"""
import argparse
import os

import h5py
import yaml

from ex.utils.group_panels import plot_group_row
from ex.utils.tables import fmt_pm, fmt_iqr, write_tables


_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument('--config', default='ex/synth/eig/config1.yaml')
config = yaml.load(open(_p.parse_args().config, 'r'), Loader=yaml.FullLoader)
PROCESSED_RESULTS_DIR = config['processed_results_dir']
FIGURES_DIR = config['figures_dir']
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']

XLABEL = r'Design optimality $\beta = \mathrm{EIG}(\xi) / \mathrm{EIG}_{\max}$'


def load(f, prefix):
    """dict method -> (B,) for every '{prefix}_<m>' dataset in f."""
    return {k[len(prefix) + 1:]: f[k][:] for k in f.keys() if k.startswith(f'{prefix}_')}


def table_rows(methods, cols, cell_fn):
    return [[m] + [cell_fn(m, i) for i in range(len(cols))] for m in methods]


def main():
    path = f'{PROCESSED_RESULTS_DIR}/regret_by_beta_d={DATA_DIM},nsamples={NSAMPLES}.h5'
    with h5py.File(path, 'r') as f:
        betas = f['design_eig_percentages'][:]
        reg = load(f, 'regret_by_beta')
        reg_lo = load(f, 'regret_lo_by_beta')
        reg_hi = load(f, 'regret_hi_by_beta')
        reg_bstd = load(f, 'regret_bstd_by_beta')
        err = load(f, 'eldr_err_by_beta')
        err_se = load(f, 'eldr_err_se_by_beta')
        err_med = load(f, 'eldr_err_med_by_beta')
        err_q1 = load(f, 'eldr_err_q1_by_beta')
        err_q3 = load(f, 'eldr_err_q3_by_beta')

    os.makedirs(FIGURES_DIR, exist_ok=True)
    beta_cols = [f'beta={b:g}' for b in betas]

    drawn = plot_group_row(
        betas, reg, reg_lo, reg_hi,
        xlabel=XLABEL, ylabel='Rel. EIG regret (MoM, IQR band)',
        out_dir=FIGURES_DIR, prefix='eig_regret_mom', yscale='linear',
    )
    regret_sections = [(
        'EIG regret -- MoM [bootstrap IQR] per beta',
        ['Method'] + beta_cols,
        table_rows(drawn, betas, lambda m, i: fmt_iqr(reg[m][i], reg_lo[m][i], reg_hi[m][i])),
    )]
    if reg_bstd:
        regret_sections.append((
            'EIG regret -- MoM +/- bootstrap std per beta',
            ['Method'] + beta_cols,
            table_rows(drawn, betas, lambda m, i: fmt_pm(reg[m][i], reg_bstd[m][i])),
        ))
    write_tables(os.path.join(FIGURES_DIR, 'eig_regret_mom_table'), regret_sections)

    err_lo = {m: err[m] - err_se[m] for m in err}
    err_hi = {m: err[m] + err_se[m] for m in err}
    drawn = plot_group_row(
        betas, err, err_lo, err_hi,
        xlabel=XLABEL, ylabel='ELDR error (abs)',
        out_dir=FIGURES_DIR, prefix='eig_eldr_err', yscale='log',
    )
    err_sections = [(
        'EIG absolute ELDR error -- mean +/- SE per beta',
        ['Method'] + beta_cols,
        table_rows(drawn, betas, lambda m, i: fmt_pm(err[m][i], err_se[m][i])),
    )]
    if err_med:
        err_sections.append((
            'EIG absolute ELDR error -- median [q1, q3] per beta',
            ['Method'] + beta_cols,
            table_rows(drawn, betas, lambda m, i: fmt_iqr(err_med[m][i], err_q1[m][i], err_q3[m][i])),
        ))
    write_tables(os.path.join(FIGURES_DIR, 'eig_eldr_err_table'), err_sections)

    print('note: pointwise LDR MAE unavailable for eig (raw results hold integrated '
          'est_eigs only); plotted regret + eldr_err.')
    print(f'done. figures in: {FIGURES_DIR}')


if __name__ == '__main__':
    main()
