"""step3_process_results.py

thin driver: load gathered results, map cells to data files, call compute_metrics
for 3 uniform metrics (eldr_abs_err, mae_train, regret).
"""

import os
import argparse
import logging
import h5py
import numpy as np
import yaml

from ex.utils import semisynth_metrics


def parse_args():
    """parse command-line arguments; return args.config."""
    parser = argparse.ArgumentParser(description='pendulum step3: compute metrics via semisynth_metrics')
    parser.add_argument('--config', default='ex/semisynth/pendulum/config.yaml',
                       help='path to config yaml')
    return parser.parse_args()


def main():
    """
    orchestrate: load config, build cell_data_paths (full-grid decode),
    call compute_metrics, print summary table.

    the gathered file is indexed by flat cell_idx = (k1_idx*G_beta + beta_idx)*seeds_default + seed.
    cell_data_paths must match this order exactly.
    """
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # expand environment variables
    for key in ('data_dir', 'raw_results_dir', 'processed_results_dir'):
        if key in config:
            config[key] = os.path.expandvars(config[key])

    data_dir = config['data_dir']
    raw_results_dir = config['raw_results_dir']
    processed_results_dir = config['processed_results_dir']
    config_seed = config.get('seed', 0)

    G_k1 = len(config['kl_targets']['k1_values'])

    # locate gathered results: config raw_results_dir first, else the in-repo
    # raw_results (where the pendulum gather actually wrote).
    gathered_h5 = os.path.join(raw_results_dir, 'results_all_cells.h5')
    if not os.path.exists(gathered_h5):
        gathered_h5 = 'ex/semisynth/pendulum/raw_results/results_all_cells.h5'
    if not os.path.exists(gathered_h5):
        raise FileNotFoundError(f"gathered results not found: {gathered_h5} (run step2_runner.gather)")

    # n_cells and the decode come from the GATHERED file, not the live config: the
    # gather was built under seeds_per_k1=40, but config seeds_default was later
    # bumped to 100, which would misdecode this file (see pendulum salvage probe).
    # gathered[i] == fragment cell_i, dense k1-major: k1 = i//seeds_per_k1, seed = i%.
    with h5py.File(gathered_h5, 'r') as f:
        est_keys = [k for k in f.keys() if k.startswith('est_ldrs_')]
        if not est_keys:
            raise ValueError(f'no est_ldrs_* datasets in {gathered_h5}')
        n_cells = f[est_keys[0]].shape[0]
    seeds_per_k1 = n_cells // G_k1

    cell_data_paths = [
        f"{data_dir}/k1_{i // seeds_per_k1}_beta_0_seed_{i % seeds_per_k1}.h5"
        for i in range(n_cells)
    ]

    # prepare output directory and path
    os.makedirs(processed_results_dir, exist_ok=True)
    output_h5 = os.path.join(processed_results_dir, 'summary.h5')

    # call compute_metrics
    methods = semisynth_metrics.compute_metrics(
        gathered_h5=gathered_h5,
        cell_data_paths=cell_data_paths,
        out_h5=output_h5,
        seed=config_seed
    )

    # load output and print summary
    with h5py.File(output_h5, 'r') as f:
        methods_attr = list(f.attrs.get('methods', methods))

        eldr_abs_err = {m: f[f'eldr_abs_err_{m}'][:] for m in methods_attr}
        mae_train = {m: f[f'mae_train_{m}'][:] for m in methods_attr}
        regret = {m: f[f'regret_{m}'][:] for m in methods_attr}

    # print summary table
    print("\n" + "="*75)
    print("Pendulum Step 3: Uniform Metrics via Compute")
    print("="*75)
    print(f"Output:   {output_h5}")
    print(f"Cells:    {n_cells} ({G_k1} k1 x {seeds_per_k1} seeds, gather-era)")
    print(f"Methods:  {len(methods_attr)}")
    print("="*75)

    for method in methods_attr:
        e_mean = np.nanmean(eldr_abs_err[method])
        e_std = np.nanstd(eldr_abs_err[method])
        m_mean = np.nanmean(mae_train[method])
        m_std = np.nanstd(mae_train[method])
        r_mean = np.nanmean(regret[method])
        r_std = np.nanstd(regret[method])

        print(f"  {method:20s} | "
              f"eldr_abs_err: {e_mean:.4f}±{e_std:.4f} | "
              f"mae_train: {m_mean:.4f}±{m_std:.4f} | "
              f"regret: {r_mean:.4f}±{r_std:.4f}")

    print("="*75 + "\n")


if __name__ == '__main__':
    main()
