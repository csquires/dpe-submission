import os

import h5py
import numpy as np
import yaml


config = yaml.load(open('experiments/ising_eldr_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
# directories
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
PROCESSED_RESULTS_DIR = config['processed_results_dir']
# dataset parameters
GRID_SIZE = config['grid_size']
DATA_DIM = GRID_SIZE ** 2
NSAMPLES = config['nsamples']
DESIGNS = config['designs']

dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM},nsamples={NSAMPLES}.h5'
raw_results_filename = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM},nsamples={NSAMPLES}.h5'
processed_results_filename = f'{PROCESSED_RESULTS_DIR}/mae_by_beta_d={DATA_DIM},nsamples={NSAMPLES}.h5'

# load design values from dataset
with h5py.File(dataset_filename, 'r') as dataset_file:
    design_betas_arr = dataset_file['design_beta_arr'][:]

# load true and estimated eldrs
with h5py.File(raw_results_filename, 'r') as results_file:
    true_eldrs_arr = results_file['true_eldrs_arr'][:]
    est_keys = [key for key in results_file.keys() if key.startswith('est_eldrs_arr_')]
    est_eldrs_by_alg = {key.replace('est_eldrs_arr_', ''): results_file[key][:] for key in est_keys}

# compute metrics per algorithm
mae_by_design = {}
mean_by_design = {}
std_by_design = {}
for alg_name, est_eldrs in est_eldrs_by_alg.items():
    abs_errors = np.abs(est_eldrs - true_eldrs_arr)

    mae_vals = []
    mean_vals = []
    std_vals = []
    for beta in DESIGNS:
        mask = np.isclose(design_betas_arr, beta)
        mae_vals.append(abs_errors[mask].mean())
        mean_vals.append(est_eldrs[mask].mean())
        std_vals.append(est_eldrs[mask].std())

    mae_by_design[alg_name] = np.array(mae_vals, dtype=np.float32)
    mean_by_design[alg_name] = np.array(mean_vals, dtype=np.float32)
    std_by_design[alg_name] = np.array(std_vals, dtype=np.float32)

# save to hdf5
os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)
with h5py.File(processed_results_filename, 'w') as out_file:
    out_file.create_dataset('design_betas', data=np.array(DESIGNS, dtype=np.float32))
    for alg_name in est_eldrs_by_alg.keys():
        out_file.create_dataset(f'mae_by_design_{alg_name}', data=mae_by_design[alg_name])
        out_file.create_dataset(f'mean_by_design_{alg_name}', data=mean_by_design[alg_name])
        out_file.create_dataset(f'std_by_design_{alg_name}', data=std_by_design[alg_name])

print(f"Processed {len(est_eldrs_by_alg)} algorithms: {list(est_eldrs_by_alg.keys())}")
print(f"Processed results saved to {processed_results_filename}")
