"""
step3_process_results.py

aggregate per-cell raw results across all (k1_idx, k2_idx, seed) into per-(k1, k2)
means and standard errors. compute pointwise ldr mae and integral eldr error per
method per cell.
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
      - mean: np.nanmean(vals) or NaN if empty
      - se: std(vals) / sqrt(n) if n >= 2, else NaN
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


def main():
    parser = argparse.ArgumentParser(description='process smodice eldr estimation results')
    parser.add_argument('--config', default='experiments/smodice_eldr_estimation/config.yaml',
                       help='path to config yaml')
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    # load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # extract config fields
    encoding_type = config['encoding']['type']
    is_onehot = encoding_type in {'onehot_joint', 'onehot_concat'}

    data_dir = config['data_dir']
    raw_results_dir = config['raw_results_dir']
    processed_results_dir = config['processed_results_dir']
    algorithms = config['algorithms']

    kl_targets = config['kl_targets']
    k1_values = np.array(kl_targets['k1_values'], dtype=np.float32)
    k2_values = np.array(kl_targets['k2_values'], dtype=np.float32)
    threshold = kl_targets['hard_corner_threshold']
    seeds_default = kl_targets['seeds_default']
    seeds_hard = kl_targets['seeds_hard']

    sigma_sweep = config['sigma_sweep']
    if sigma_sweep['enabled']:
        sigmas = sigma_sweep['sigmas']
    else:
        sigmas = [config['encoding']['sigma']]

    # grid dimensions
    G_k1 = len(k1_values)
    G_k2 = len(k2_values)

    # initialize result structures: results[i][j] = {method: [list of (mae, eldr_err)]}
    results = [[{method: [] for method in algorithms} for _ in range(G_k2)] for _ in range(G_k1)]
    results_discrete = [[[] for _ in range(G_k2)] for _ in range(G_k1)]
    feasibility = np.zeros((G_k1, G_k2), dtype=np.int8)

    missing_data_files = set()
    missing_results_files = set()
    missing_method_keys = set()

    # nested loop: sigma, (k1_idx, k2_idx), seed
    for sigma in sigmas:
        sigma_dir = "sigma_na" if is_onehot else f"sigma_{sigma:.3f}"

        for k1_idx in range(G_k1):
            for k2_idx in range(G_k2):
                k1_val = k1_values[k1_idx]
                k2_val = k2_values[k2_idx]

                # determine seed count for this cell
                if k1_val >= threshold and k2_val >= threshold:
                    n_seeds = seeds_hard
                else:
                    n_seeds = seeds_default

                for seed in range(n_seeds):
                    data_path = f"{data_dir}/{encoding_type}/{sigma_dir}/kl1_{k1_idx}_kl2_{k2_idx}_seed_{seed}.h5"
                    results_path = f"{raw_results_dir}/{encoding_type}/{sigma_dir}/kl1_{k1_idx}_kl2_{k2_idx}_seed_{seed}.h5"

                    # check data file exists
                    if not os.path.exists(data_path):
                        missing_data_files.add(data_path)
                        continue

                    # check results file exists
                    if not os.path.exists(results_path):
                        missing_results_files.add(results_path)
                        continue

                    # load ground truth
                    try:
                        with h5py.File(data_path, 'r') as f:
                            if is_onehot:
                                true_ldrs = f["true_ldrs_discrete"][:]
                            else:
                                true_ldrs = f["true_ldrs_smoothed"][:]
                            integrated_eldr = float(f.attrs["integrated_eldr"])
                    except Exception as e:
                        logging.warning(f"error reading data file {data_path}: {e}")
                        continue

                    # load and process results
                    try:
                        with h5py.File(results_path, 'r') as f:
                            # load discrete ground truth for special case
                            if not is_onehot:
                                try:
                                    with h5py.File(data_path, 'r') as f_data:
                                        true_ldrs_discrete = f_data["true_ldrs_discrete"][:]
                                except Exception as e:
                                    logging.warning(f"error reading discrete ldrs from {data_path}: {e}")
                                    true_ldrs_discrete = None

                            for method in algorithms:
                                key = f"est_ldrs_{method}"
                                if key not in f:
                                    missing_method_keys.add((method, results_path))
                                    continue

                                est_ldrs = f[key][:]
                                pointwise_mae = np.mean(np.abs(est_ldrs - true_ldrs))
                                eldr_est = np.mean(est_ldrs)
                                eldr_err = np.abs(eldr_est - integrated_eldr)

                                results[k1_idx][k2_idx][method].append((pointwise_mae, eldr_err))

                                # special: TabularPluginDRE on blob/flow stores discrete MAE
                                if method == "TabularPluginDRE" and not is_onehot and true_ldrs_discrete is not None:
                                    discrete_mae = np.mean(np.abs(est_ldrs - true_ldrs_discrete))
                                    results_discrete[k1_idx][k2_idx].append(discrete_mae)
                    except Exception as e:
                        logging.warning(f"error processing results file {results_path}: {e}")
                        continue

    # log warnings for missing files
    if missing_data_files:
        logging.warning(f"missing {len(missing_data_files)} data files (skipped)")
    if missing_results_files:
        logging.warning(f"missing {len(missing_results_files)} results files (skipped)")
    if missing_method_keys:
        logging.warning(f"missing {len(missing_method_keys)} method keys in results files (skipped)")

    # set feasibility: 1 if any method has >=1 seed
    for k1_idx in range(G_k1):
        for k2_idx in range(G_k2):
            for method in algorithms:
                if len(results[k1_idx][k2_idx][method]) > 0:
                    feasibility[k1_idx, k2_idx] = 1
                    break

    # initialize output arrays
    mean_arrays = {metric: {method: np.full((G_k1, G_k2), np.nan, dtype=np.float32)
                           for method in algorithms}
                   for metric in ['pointwise_mae', 'eldr_err']}
    se_arrays = {metric: {method: np.full((G_k1, G_k2), np.nan, dtype=np.float32)
                         for method in algorithms}
                 for metric in ['pointwise_mae', 'eldr_err']}
    n_arrays = {metric: {method: np.zeros((G_k1, G_k2), dtype=np.int32)
                        for method in algorithms}
                for metric in ['pointwise_mae', 'eldr_err']}

    discrete_mean = np.full((G_k1, G_k2), np.nan, dtype=np.float32)
    discrete_se = np.full((G_k1, G_k2), np.nan, dtype=np.float32)
    discrete_n = np.zeros((G_k1, G_k2), dtype=np.int32)

    # aggregate per cell, per method, per metric
    for k1_idx in range(G_k1):
        for k2_idx in range(G_k2):
            for method in algorithms:
                vals_list = results[k1_idx][k2_idx][method]

                if len(vals_list) > 0:
                    # extract pointwise_mae (index 0) and eldr_err (index 1)
                    mae_vals = [v[0] for v in vals_list]
                    eldr_vals = [v[1] for v in vals_list]

                    # aggregate pointwise_mae
                    mae_mean, mae_se, mae_n = agg_metric(mae_vals)
                    mean_arrays['pointwise_mae'][method][k1_idx, k2_idx] = mae_mean
                    se_arrays['pointwise_mae'][method][k1_idx, k2_idx] = mae_se
                    n_arrays['pointwise_mae'][method][k1_idx, k2_idx] = mae_n

                    # aggregate eldr_err
                    eldr_mean, eldr_se, eldr_n = agg_metric(eldr_vals)
                    mean_arrays['eldr_err'][method][k1_idx, k2_idx] = eldr_mean
                    se_arrays['eldr_err'][method][k1_idx, k2_idx] = eldr_se
                    n_arrays['eldr_err'][method][k1_idx, k2_idx] = eldr_n

            # special: discrete MAE for TabularPluginDRE on blob/flow
            if not is_onehot and "TabularPluginDRE" in algorithms:
                discrete_vals = results_discrete[k1_idx][k2_idx]
                if len(discrete_vals) > 0:
                    d_mean, d_se, d_n = agg_metric(discrete_vals)
                    discrete_mean[k1_idx, k2_idx] = d_mean
                    discrete_se[k1_idx, k2_idx] = d_se
                    discrete_n[k1_idx, k2_idx] = d_n

    # create output directory
    os.makedirs(processed_results_dir, exist_ok=True)

    # determine sigma dir for output
    sigma_dir = "sigma_na" if is_onehot else f"sigma_{sigmas[0]:.3f}"
    output_subdir = f"{processed_results_dir}/{encoding_type}/{sigma_dir}"
    os.makedirs(output_subdir, exist_ok=True)

    # write summary HDF5
    output_path = f"{output_subdir}/summary.h5"
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('k1_values', data=k1_values, dtype=np.float32)
        f.create_dataset('k2_values', data=k2_values, dtype=np.float32)
        f.create_dataset('feasibility', data=feasibility, dtype=np.int8)

        for method in algorithms:
            for metric in ['pointwise_mae', 'eldr_err']:
                f.create_dataset(f'{metric}_{method}_mean',
                               data=mean_arrays[metric][method], dtype=np.float32)
                f.create_dataset(f'{metric}_{method}_se',
                               data=se_arrays[metric][method], dtype=np.float32)
                f.create_dataset(f'{metric}_{method}_n',
                               data=n_arrays[metric][method], dtype=np.int32)

        # special: discrete MAE for TabularPluginDRE on blob/flow
        if not is_onehot and "TabularPluginDRE" in algorithms:
            f.create_dataset('pointwise_mae_discrete_TabularPluginDRE_mean',
                           data=discrete_mean, dtype=np.float32)
            f.create_dataset('pointwise_mae_discrete_TabularPluginDRE_se',
                           data=discrete_se, dtype=np.float32)
            f.create_dataset('pointwise_mae_discrete_TabularPluginDRE_n',
                           data=discrete_n, dtype=np.int32)

    # print summary table
    print("\n" + "="*120)
    print("ELDR Estimation Results Summary")
    print("="*120)

    # header row: k1/k2 cell labels
    header = "Method".ljust(25)
    for k1_idx in range(G_k1):
        for k2_idx in range(G_k2):
            label = f"K1={k1_values[k1_idx]:.1f}_K2={k2_values[k2_idx]:.1f}"
            header += label.rjust(50)
    print(header)
    print("-"*120)

    # data rows: per method
    for method in algorithms:
        row = method.ljust(25)
        for k1_idx in range(G_k1):
            for k2_idx in range(G_k2):
                mae_mean = mean_arrays['pointwise_mae'][method][k1_idx, k2_idx]
                mae_se = se_arrays['pointwise_mae'][method][k1_idx, k2_idx]
                mae_n = n_arrays['pointwise_mae'][method][k1_idx, k2_idx]
                eldr_mean = mean_arrays['eldr_err'][method][k1_idx, k2_idx]
                eldr_se = se_arrays['eldr_err'][method][k1_idx, k2_idx]
                eldr_n = n_arrays['eldr_err'][method][k1_idx, k2_idx]

                if mae_n == 0:
                    cell_str = "NaN (0)"
                else:
                    cell_str = f"{mae_mean:.4f}±{mae_se:.4f} ({mae_n}) | "
                    cell_str += f"{eldr_mean:.4f}±{eldr_se:.4f} ({eldr_n})"
                row += cell_str.rjust(50)
        print(row)

    print("="*120)
    print(f"Processed results saved to: {output_path}")
    print(f"Encoding: {encoding_type}, Sigma dir: {sigma_dir}")
    print(f"Grid: {G_k1} x {G_k2}, Feasible cells: {np.sum(feasibility)}")
    print("="*120 + "\n")


if __name__ == '__main__':
    main()
