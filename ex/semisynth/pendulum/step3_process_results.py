"""step3_process_results.py

aggregate per-cell raw results across seeds into per-K1 summary.
computes pointwise MAE and integral ELDR error per method per K1 bin.
"""

import os
import argparse
import logging
import h5py
import numpy as np
import yaml


def agg_metric(vals):
    """
    aggregate a list of metric values into (mean, se, n).

    vals: list of float, possibly containing NaN
    returns: (mean, se, n) where
      - mean: np.mean(valid) or NaN if empty
      - se: std(valid) / sqrt(n) if n >= 2, else NaN
      - n: count of non-NaN values
    """
    valid = np.array([v for v in vals if not np.isnan(v)])
    if len(valid) == 0:
        return np.nan, np.nan, 0
    mean = np.mean(valid)
    n = len(valid)
    if n >= 2:
        se = np.std(valid, ddof=1) / np.sqrt(n)
    else:
        se = np.nan
    return mean, se, n


def parse_args():
    """
    parse command-line arguments.

    returns: Namespace with config path
    """
    parser = argparse.ArgumentParser(description='process pendulum eldr estimation results')
    parser.add_argument('--config', default='ex/semisynth/pendulum/config.yaml',
                       help='path to config yaml')
    return parser.parse_args()


def compute_metrics_for_seed(data_path, est_ldrs, method):
    """
    given a per-cell data file and the est_ldrs row for one (method, cell),
    return (pointwise_mae, eldr_err) or None.

    ported 2026-06-15 to read est_ldrs from the step2_runner gathered
    results_all_cells.h5 (one row per cell) instead of per-seed result files.
    true_ldrs / integrated_eldr still come from the per-cell data file.
    """
    if not os.path.exists(data_path) or est_ldrs is None:
        return None
    try:
        with h5py.File(data_path, 'r') as f:
            # use the canonical true_ldrs field: it matches the sign convention of
            # est_ldrs (predict_ldr) and integrated_eldr. log_p_pstar[:,2]-[:,1] is
            # the NEGATED convention (bug in the old per-seed reader).
            true_ldrs = f['true_ldrs'][:]
            integrated_eldr = float(f.attrs['integrated_eldr'])

        est_ldrs = np.asarray(est_ldrs)
        if est_ldrs.shape[0] != true_ldrs.shape[0] or not np.isfinite(est_ldrs).all():
            return None
        pointwise_mae = np.mean(np.abs(est_ldrs - true_ldrs))
        eldr_err = np.abs(np.mean(est_ldrs) - integrated_eldr)
        return (pointwise_mae, eldr_err)
    except Exception as e:
        logging.warning(f'error processing {data_path} [{method}]: {e}')
        return None


def aggregate_cells(config):
    """
    loop over all (k1_idx, beta_idx, seed) and collect metrics for each method.

    logic:
      1. extract from config: data_dir, raw_results_dir, k1_values, beta_values,
         algorithms, seeds_default
      2. initialize results[k1_idx][beta_idx] = {method: [list of (mae, eldr_err) tuples]}
      3. nested loop: for k1_idx, beta_idx, seed:
         - construct paths
         - call compute_metrics_for_seed for each method
         - if not None, append to results[k1_idx][beta_idx][method]
      4. return results (3-level nested dict)
    """
    data_dir = config['data_dir']
    raw_results_dir = config['raw_results_dir']

    kl_targets = config['kl_targets']
    k1_values = np.array(kl_targets['k1_values'], dtype=np.float32)
    beta_values = np.array(kl_targets['beta_values'], dtype=np.float32)
    seeds_default = kl_targets['seeds_default']

    algorithms = config['algorithms']

    G_k1 = len(k1_values)
    G_beta = len(beta_values)

    # results[i][j] = {method: [list of (mae, eldr_err) tuples]}
    results = [[{method: [] for method in algorithms} for _ in range(G_beta)] for _ in range(G_k1)]

    missing_data_files = set()
    missing_method_keys = set()

    # read the step2_runner gathered file (one est_ldrs_<method> row per cell).
    # cell_idx = (k1_idx*G_beta + beta_idx)*seeds_default + seed (matches the
    # adapter's _decode_cell). this replaces the old per-seed result files.
    gathered = os.path.join(raw_results_dir, "results_all_cells.h5")
    if not os.path.exists(gathered):
        raise FileNotFoundError(f"gathered results not found: {gathered} (run step2_runner.gather)")
    with h5py.File(gathered, 'r') as gf:
        est_all = {m: gf[f'est_ldrs_{m}'][:] for m in algorithms if f'est_ldrs_{m}' in gf}
    for m in algorithms:
        if m not in est_all:
            missing_method_keys.add(m)

    for k1_idx in range(G_k1):
        for beta_idx in range(G_beta):
            for seed in range(seeds_default):
                data_path = f"{data_dir}/k1_{k1_idx}_beta_{beta_idx}_seed_{seed}.h5"
                if not os.path.exists(data_path):
                    missing_data_files.add(data_path)
                    continue
                cell_idx = (k1_idx * G_beta + beta_idx) * seeds_default + seed
                for method in algorithms:
                    arr = est_all.get(method)
                    row = arr[cell_idx] if (arr is not None and cell_idx < arr.shape[0]) else None
                    metrics = compute_metrics_for_seed(data_path, row, method)
                    if metrics is not None:
                        results[k1_idx][beta_idx][method].append(metrics)

    if missing_data_files:
        logging.warning(f'missing {len(missing_data_files)} data files (skipped)')
    if missing_method_keys:
        logging.warning(f'methods absent from gathered file (skipped): {sorted(missing_method_keys)}')

    return results


def write_summary_h5(out_path, k1_values, beta_value, beta_count, per_method):
    """
    write 1D HDF5 summary file.

    per_method[method] is dict with pre-aggregated 1D arrays of shape [G_k1]:
      keys: mae_mean, mae_se, mae_n, eldr_mean, eldr_se, eldr_n
    plus per-seed arrays of shape [G_k1, seeds_default], NaN-padded over the
    seed axis (so the step4 boxplot can split boxes by K1):
      keys: mae_seed_values, eldr_seed_values

    logic:
      1. create output directory if needed
      2. open h5 file for writing
      3. write dataset k1_values (float32)
      4. set root attrs: beta_value (float), beta_count (int)
      5. for each method in per_method:
         - write datasets: pointwise_mae_{method}, pointwise_mae_se_{method}, pointwise_mae_n_{method},
           eldr_err_{method}, eldr_err_se_{method}, eldr_err_n_{method}
         - all 1D, shape [G_k1]. mae/eldr means and SEs are float32; n is int32.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('k1_values', data=k1_values, dtype=np.float32)
        f.attrs['beta_value'] = float(beta_value)
        f.attrs['beta_count'] = int(beta_count)

        for method in per_method:
            method_data = per_method[method]

            f.create_dataset(f'pointwise_mae_{method}',
                           data=method_data['mae_mean'], dtype=np.float32)
            f.create_dataset(f'pointwise_mae_se_{method}',
                           data=method_data['mae_se'], dtype=np.float32)
            f.create_dataset(f'pointwise_mae_n_{method}',
                           data=method_data['mae_n'], dtype=np.int32)

            f.create_dataset(f'eldr_err_{method}',
                           data=method_data['eldr_mean'], dtype=np.float32)
            f.create_dataset(f'eldr_err_se_{method}',
                           data=method_data['eldr_se'], dtype=np.float32)
            f.create_dataset(f'eldr_err_n_{method}',
                           data=method_data['eldr_n'], dtype=np.int32)

            # flattened per-seed values for the step4 boxplot overlay
            f.create_dataset(f'pointwise_mae_{method}_seed_values',
                           data=method_data['mae_seed_values'], dtype=np.float32)
            f.create_dataset(f'eldr_err_{method}_seed_values',
                           data=method_data['eldr_seed_values'], dtype=np.float32)


def main():
    """
    main control flow.

    1. parse args, load config, setup logging
    2. extract paths, grid, algorithms
    3. call aggregate_cells -> nested results dict
    4. loop over k1_idx: aggregate metrics per method
    5. call write_summary_h5 with pre-aggregated 1D arrays
    6. print summary table
    """
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # expand ${DPE_DATA_ROOT} etc. (yaml.safe_load leaves them literal)
    for _k in ('data_dir', 'raw_results_dir', 'processed_results_dir'):
        if _k in config:
            config[_k] = os.path.expandvars(config[_k])

    data_dir = config['data_dir']
    raw_results_dir = config['raw_results_dir']
    processed_results_dir = config['processed_results_dir']
    algorithms = config['algorithms']

    kl_targets = config['kl_targets']
    k1_values = np.array(kl_targets['k1_values'], dtype=np.float32)
    beta_values = np.array(kl_targets['beta_values'], dtype=np.float32)
    beta_value = beta_values[0]

    G_k1 = len(k1_values)

    # aggregate raw results
    results = aggregate_cells(config)

    # initialize output arrays: means, ses, ns per method
    means = {method: np.full((G_k1,), np.nan, dtype=np.float32) for method in algorithms}
    ses = {method: np.full((G_k1,), np.nan, dtype=np.float32) for method in algorithms}
    ns = {method: np.zeros((G_k1,), dtype=np.int32) for method in algorithms}

    eldr_means = {method: np.full((G_k1,), np.nan, dtype=np.float32) for method in algorithms}
    eldr_ses = {method: np.full((G_k1,), np.nan, dtype=np.float32) for method in algorithms}
    eldr_ns = {method: np.zeros((G_k1,), dtype=np.int32) for method in algorithms}

    # aggregate per k1_idx
    for k1_idx in range(G_k1):
        for method in algorithms:
            # beta_idx is always 0 (singleton beta)
            vals_list = results[k1_idx][0][method]

            if len(vals_list) > 0:
                mae_vals = [v[0] for v in vals_list]
                eldr_vals = [v[1] for v in vals_list]

                # aggregate pointwise_mae
                mae_mean, mae_se, mae_n = agg_metric(mae_vals)
                means[method][k1_idx] = mae_mean
                ses[method][k1_idx] = mae_se
                ns[method][k1_idx] = mae_n

                # aggregate eldr_err
                eldr_mean, eldr_se, eldr_n = agg_metric(eldr_vals)
                eldr_means[method][k1_idx] = eldr_mean
                eldr_ses[method][k1_idx] = eldr_se
                eldr_ns[method][k1_idx] = eldr_n

    # per-seed arrays shaped [G_k1, seeds_default], NaN-padded over the seed
    # axis (beta is singleton, idx 0). lets step4 split boxes by K1.
    seeds_default = kl_targets['seeds_default']
    seed_mae = {method: np.full((G_k1, seeds_default), np.nan, dtype=np.float32)
                for method in algorithms}
    seed_eldr = {method: np.full((G_k1, seeds_default), np.nan, dtype=np.float32)
                 for method in algorithms}
    for method in algorithms:
        for k1_idx in range(G_k1):
            vals = results[k1_idx][0][method]
            for s, (mae, eldr) in enumerate(vals[:seeds_default]):
                seed_mae[method][k1_idx, s] = mae
                seed_eldr[method][k1_idx, s] = eldr

    # construct per_method dict for write_summary_h5
    per_method = {}
    for method in algorithms:
        per_method[method] = {
            'mae_mean': means[method],
            'mae_se': ses[method],
            'mae_n': ns[method],
            'eldr_mean': eldr_means[method],
            'eldr_se': eldr_ses[method],
            'eldr_n': eldr_ns[method],
            'mae_seed_values': seed_mae[method],
            'eldr_seed_values': seed_eldr[method],
        }

    # write output
    os.makedirs(processed_results_dir, exist_ok=True)
    output_path = f'{processed_results_dir}/mae_summary.h5'
    write_summary_h5(output_path, k1_values, beta_value, 1, per_method)

    # print summary
    print("\n" + "="*100)
    print("Pendulum ELDR Estimation: Step 3 Results Processing Complete")
    print("="*100)
    print(f"Processed results saved to: {output_path}")
    print(f"\nK1 values: {list(k1_values)}")
    print(f"beta value: {beta_value} (singleton; K2 is derived per-k1)")
    print(f"Algorithms: {algorithms}")
    print(f"\nMetrics per K1:")

    for k1_idx in range(G_k1):
        print(f"\n  K1={k1_values[k1_idx]:.2f}:")
        for method in algorithms:
            mae_mean = means[method][k1_idx]
            mae_se = ses[method][k1_idx]
            mae_n = ns[method][k1_idx]
            eldr_mean = eldr_means[method][k1_idx]
            eldr_se = eldr_ses[method][k1_idx]

            if mae_n == 0:
                print(f"    {method:25s}: NaN (0 seeds)")
            else:
                print(f"    {method:25s}: MAE={mae_mean:.4f}±{mae_se:.4f} "
                      f"| ELDR={eldr_mean:.4f}±{eldr_se:.4f} (n={mae_n})")

    print("="*100 + "\n")


if __name__ == '__main__':
    main()
