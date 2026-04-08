"""
step3_process_results.py

aggregate raw eldr estimation results into per-alpha mae and std statistics.
loads ground truth ldrs and estimated ldrs, computes per-pair mae, aggregates
to per-alpha mean/std ready for plotting.
"""

import os
import argparse
import h5py
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description='process mnist eldr estimation results')
    parser.add_argument('--config', default='experiments/mnist_eldr_estimation/config.yaml',
                       help='path to config yaml')
    args = parser.parse_args()

    # load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    alphas = config['alphas']
    data_dir = config['data_dir']
    raw_results_dir = config['raw_results_dir']
    processed_results_dir = config['processed_results_dir']
    algorithms = config['algorithms']

    n_alphas = len(alphas)
    n_pairs = config['num_pairs_per_alpha']

    # initialize output data structures
    mae_by_method = {method: np.zeros(n_alphas, dtype=np.float32) for method in algorithms}
    std_by_method = {method: np.zeros(n_alphas, dtype=np.float32) for method in algorithms}
    per_pair_by_method = {method: np.zeros((n_alphas, n_pairs), dtype=np.float32) for method in algorithms}

    # compute mae for each (alpha, pair) combination
    for alpha_idx in range(n_alphas):
        for pair_idx in range(n_pairs):
            data_file = f'{data_dir}/alpha_{alpha_idx}_pair_{pair_idx}.h5'
            raw_results_file = f'{raw_results_dir}/alpha_{alpha_idx}_pair_{pair_idx}.h5'

            # validate files exist
            if not os.path.exists(data_file):
                print(f'warning: data file not found {data_file}')
                for method in algorithms:
                    per_pair_by_method[method][alpha_idx, pair_idx] = np.nan
                continue
            if not os.path.exists(raw_results_file):
                print(f'warning: raw results file not found {raw_results_file}')
                for method in algorithms:
                    per_pair_by_method[method][alpha_idx, pair_idx] = np.nan
                continue

            # load ground truth
            try:
                with h5py.File(data_file, 'r') as f:
                    true_ldrs = f['true_ldrs'][:]
            except Exception as e:
                print(f'warning: error reading {data_file}: {e}')
                for method in algorithms:
                    per_pair_by_method[method][alpha_idx, pair_idx] = np.nan
                continue

            # load and compute mae for each method
            try:
                with h5py.File(raw_results_file, 'r') as results_file:
                    for method in algorithms:
                        dataset_name = f'est_ldrs_{method}'
                        if dataset_name not in results_file:
                            print(f'warning: {dataset_name} not found in {raw_results_file}')
                            mae_val = np.nan
                        else:
                            est_ldrs = results_file[dataset_name][:]
                            abs_errors = np.abs(est_ldrs - true_ldrs)
                            mae_val = np.mean(abs_errors)

                        per_pair_by_method[method][alpha_idx, pair_idx] = mae_val
            except Exception as e:
                print(f'warning: error processing {raw_results_file}: {e}')
                for method in algorithms:
                    per_pair_by_method[method][alpha_idx, pair_idx] = np.nan

    # aggregate per-pair mae to per-alpha statistics
    for method in algorithms:
        per_pair_arr = per_pair_by_method[method]
        mae_list = []
        std_list = []

        for alpha_idx in range(n_alphas):
            mae_values = per_pair_arr[alpha_idx, :]
            mae_mean = np.nanmean(mae_values)
            mae_std = np.nanstd(mae_values, ddof=1)

            mae_list.append(mae_mean)
            std_list.append(mae_std)

        mae_by_method[method] = np.array(mae_list, dtype=np.float32)
        std_by_method[method] = np.array(std_list, dtype=np.float32)

    # validation before save
    for method in algorithms:
        mae_arr = mae_by_method[method]
        assert mae_arr.shape == (n_alphas,), f'mae array shape mismatch for {method}'
        assert len(std_by_method[method]) == n_alphas, f'std array shape mismatch for {method}'

        if np.all(np.isnan(mae_arr)):
            print(f'warning: {method} has all-nan mae values')

        if per_pair_by_method[method].shape != (n_alphas, n_pairs):
            print(f'warning: per_pair shape mismatch for {method}')

    # create output directory and save results
    os.makedirs(processed_results_dir, exist_ok=True)
    processed_results_file = f'{processed_results_dir}/mae_summary.h5'

    with h5py.File(processed_results_file, 'w') as out_file:
        # store alpha values
        out_file.create_dataset('alphas', data=np.array(alphas, dtype=np.float32))

        # store metrics for each algorithm
        for method in algorithms:
            out_file.create_dataset(f'mae_{method}', data=mae_by_method[method])
            out_file.create_dataset(f'std_{method}', data=std_by_method[method])
            out_file.create_dataset(f'per_pair_{method}', data=per_pair_by_method[method])

    # print summary report
    print("\n" + "="*80)
    print("MNIST ELDR Estimation: Step 3 Results Processing Complete")
    print("="*80)
    print(f"Processed results saved to: {processed_results_file}")
    print(f"\nAlphas: {alphas}")
    print(f"Pairs per alpha: {n_pairs}")
    print(f"Algorithms: {algorithms}")
    print(f"\nMetrics computed per algorithm:")
    for method in algorithms:
        mae_arr = mae_by_method[method]
        std_arr = std_by_method[method]
        mae_str = ', '.join([f'{x:.4f}' for x in mae_arr])
        std_str = ', '.join([f'{x:.4f}' for x in std_arr])
        print(f"  {method:25s}: MAE=[{mae_str}], Std=[{std_str}]")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
