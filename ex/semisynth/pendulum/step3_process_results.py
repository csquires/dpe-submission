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


def compute_metrics_for_seed(data_path, results_path, method):
    """
    load one per-cell file pair and return (pointwise_mae, eldr_err) or None.

    logic:
      1. check both files exist; if not, return None
      2. load data_path:
         - read log_p_pstar (shape [N, 3])
         - compute true_ldrs = log_p_pstar[:, 2] - log_p_pstar[:, 1]
         - read attr integrated_eldr
      3. load results_path:
         - read est_ldrs_{method} (shape [N])
         - compute pointwise_mae = mean(abs(est_ldrs - true_ldrs))
         - compute eldr_err = abs(mean(est_ldrs) - integrated_eldr)
      4. return (pointwise_mae, eldr_err)
      5. on exception: log warning, return None
    """
    if not os.path.exists(data_path) or not os.path.exists(results_path):
        return None

    try:
        with h5py.File(data_path, 'r') as f:
            log_p_pstar = f['log_p_pstar'][:]  # shape [N, 3]
            true_ldrs = log_p_pstar[:, 2] - log_p_pstar[:, 1]  # E - O
            integrated_eldr = float(f.attrs['integrated_eldr'])

        with h5py.File(results_path, 'r') as f:
            key = f'est_ldrs_{method}'
            if key not in f:
                return None
            est_ldrs = f[key][:]  # shape [N]

        pointwise_mae = np.mean(np.abs(est_ldrs - true_ldrs))
        eldr_est = np.mean(est_ldrs)
        eldr_err = np.abs(eldr_est - integrated_eldr)

        return (pointwise_mae, eldr_err)

    except Exception as e:
        logging.warning(f'error processing {data_path}/{results_path}: {e}')
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
    missing_results_files = set()
    missing_method_keys = set()

    for k1_idx in range(G_k1):
        for beta_idx in range(G_beta):
            for seed in range(seeds_default):
                data_path = f"{data_dir}/k1_{k1_idx}_beta_{beta_idx}_seed_{seed}.h5"
                results_path = f"{raw_results_dir}/k1_{k1_idx}_beta_{beta_idx}_seed_{seed}.h5"

                # check files exist
                if not os.path.exists(data_path):
                    missing_data_files.add(data_path)
                    continue
                if not os.path.exists(results_path):
                    missing_results_files.add(results_path)
                    continue

                for method in algorithms:
                    metrics = compute_metrics_for_seed(data_path, results_path, method)
                    if metrics is not None:
                        results[k1_idx][beta_idx][method].append(metrics)
                    else:
                        # track missing method keys
                        try:
                            with h5py.File(results_path, 'r') as f:
                                if f'est_ldrs_{method}' not in f:
                                    missing_method_keys.add((method, results_path))
                        except:
                            pass

    if missing_data_files:
        logging.warning(f'missing {len(missing_data_files)} data files (skipped)')
    if missing_results_files:
        logging.warning(f'missing {len(missing_results_files)} results files (skipped)')
    if missing_method_keys:
        logging.warning(f'missing {len(missing_method_keys)} method keys in results files (skipped)')

    return results


def write_summary_h5(out_path, k1_values, beta_value, beta_count, per_method):
    """
    write 1D HDF5 summary file.

    per_method[method] is dict with pre-aggregated 1D arrays of shape [G_k1]:
      keys: mae_mean, mae_se, mae_n, eldr_mean, eldr_se, eldr_n

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
