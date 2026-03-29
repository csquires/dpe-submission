"""
Filter density ratio estimation data to select rows with specific EIG percentage.

Loads data from eig_estimation experiment (600 rows, 6 datasets) and filters to
rows where design_eig_percentage == target value, writing filtered subset to output file.

Algorithm:
1. Load config from experiments/hidden_dim_scaling/config.yaml
2. Validate source file exists
3. Load all 6 datasets from source HDF5
4. Identify rows matching target EIG percentage using numpy.isclose()
5. Filter all datasets to matched indices
6. Validate shapes and dtypes on filtered data
7. Write to temp file, then atomic rename to final location
8. Print summary with counts and shapes
"""

import os
import h5py
import numpy as np
import yaml


def main():
    # step 1: load configuration
    config_path = "experiments/hidden_dim_scaling/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    source_data_dir = config["source_data_dir"]
    data_dir = config["data_dir"]
    target_eig_percentage = config.get("target_design_eig_percentage", 0.7)

    source_file = f"{source_data_dir}/dataset_d=3,nsamples=10000.h5"

    # step 2: validate source file exists
    if not os.path.exists(source_file):
        raise FileNotFoundError(
            f"Source file not found: {source_file}. Check config['source_data_dir']."
        )

    # step 3: load source data
    with h5py.File(source_file, "r") as f:
        prior_mean = f["prior_mean_arr"][:]
        prior_cov = f["prior_covariance_arr"][:]
        design_eig_pct = f["design_eig_percentage_arr"][:]
        design = f["design_arr"][:]
        theta_samples = f["theta_samples_arr"][:]
        y_samples = f["y_samples_arr"][:]

    nrows = prior_mean.shape[0]
    assert nrows > 0, f"Source file has no rows"

    # step 4: identify matching rows
    mask = np.isclose(design_eig_pct, target_eig_percentage, atol=1e-6)
    mask = mask.flatten()  # convert (600,1) to (600,) for indexing
    indices = np.where(mask)[0]
    num_matches = len(indices)

    if num_matches == 0:
        unique_values = np.unique(design_eig_pct)
        raise ValueError(
            f"No rows found with design_eig_percentage == {target_eig_percentage}. "
            f"Available values in source: {unique_values}"
        )

    if num_matches != 100:
        print(
            f"Warning: Expected 100 rows (5 priors × 20 designs), "
            f"found {num_matches}"
        )

    # step 5: filter all datasets
    filtered_prior_mean = prior_mean[indices]
    filtered_prior_cov = prior_cov[indices]
    filtered_design_eig_pct = design_eig_pct[indices]
    filtered_design = design[indices]
    filtered_theta_samples = theta_samples[indices]
    filtered_y_samples = y_samples[indices]

    # validate filter shapes
    assert filtered_prior_mean.shape == (num_matches, 3), \
        f"prior_mean shape mismatch: {filtered_prior_mean.shape} vs ({num_matches}, 3)"
    assert filtered_prior_cov.shape == (num_matches, 3, 3), \
        f"prior_cov shape mismatch: {filtered_prior_cov.shape} vs ({num_matches}, 3, 3)"
    assert filtered_design_eig_pct.shape == (num_matches, 1), \
        f"design_eig_pct shape mismatch: {filtered_design_eig_pct.shape} vs ({num_matches}, 1)"
    assert filtered_design.shape == (num_matches, 3, 1), \
        f"design shape mismatch: {filtered_design.shape} vs ({num_matches}, 3, 1)"
    assert filtered_theta_samples.shape == (num_matches, 10000, 3), \
        f"theta_samples shape mismatch: {filtered_theta_samples.shape} vs ({num_matches}, 10000, 3)"
    assert filtered_y_samples.shape == (num_matches, 10000, 1), \
        f"y_samples shape mismatch: {filtered_y_samples.shape} vs ({num_matches}, 10000, 1)"

    # validate data types
    assert filtered_prior_mean.dtype == np.float32, \
        f"prior_mean dtype is {filtered_prior_mean.dtype}, expected float32"
    assert filtered_theta_samples.dtype == np.float32, \
        f"theta_samples dtype is {filtered_theta_samples.dtype}, expected float32"

    # validate EIG values in filtered subset
    assert np.allclose(filtered_design_eig_pct, target_eig_percentage, atol=1e-6), \
        f"filtered EIG values don't match target {target_eig_percentage}"

    # step 6: atomic write with temp file
    os.makedirs(data_dir, exist_ok=True)
    output_file = f"{data_dir}/dataset_filtered.h5"
    temp_file = f"{output_file}.tmp"

    with h5py.File(temp_file, "w") as f:
        f.create_dataset("prior_mean_arr", data=filtered_prior_mean)
        f.create_dataset("prior_covariance_arr", data=filtered_prior_cov)
        f.create_dataset("design_eig_percentage_arr", data=filtered_design_eig_pct)
        f.create_dataset("design_arr", data=filtered_design)
        f.create_dataset("theta_samples_arr", data=filtered_theta_samples)
        f.create_dataset("y_samples_arr", data=filtered_y_samples)

    os.replace(temp_file, output_file)

    # step 7: summary report
    print(f"Source file: {source_file}")
    print(f"Total rows in source: {nrows}")
    print(f"Target EIG percentage: {target_eig_percentage}")
    print(f"Filtered rows: {num_matches}")
    print(f"Output file: {output_file}")
    print(f"Output shapes: prior_mean {filtered_prior_mean.shape}, "
          f"prior_cov {filtered_prior_cov.shape}, "
          f"design_eig_pct {filtered_design_eig_pct.shape}, "
          f"design {filtered_design.shape}, "
          f"theta_samples {filtered_theta_samples.shape}, "
          f"y_samples {filtered_y_samples.shape}")


if __name__ == "__main__":
    main()
