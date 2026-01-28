import os

import h5py
import numpy as np
import yaml


config = yaml.load(open('experiments/eig_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
# directories
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
PROCESSED_RESULTS_DIR = config['processed_results_dir']
# dataset parameters
DATA_DIM = config['data_dim']
DESIGN_EIG_PERCENTAGES = config['design_eig_percentages']

dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM}.h5'
# raw_results_filename = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM}.h5'
raw_results_filename = f'{RAW_RESULTS_DIR}/direct.h5'
# processed_results_filename = f'{PROCESSED_RESULTS_DIR}/mae_by_beta_d={DATA_DIM}.h5'
processed_results_filename = f'{PROCESSED_RESULTS_DIR}/direct.h5'

with h5py.File(dataset_filename, 'r') as dataset_file:
    design_eig_percentages = dataset_file['design_eig_percentage_arr'][:].squeeze()

with h5py.File(raw_results_filename, 'r') as results_file:
    true_eigs = results_file['true_eigs_arr'][:]
    est_keys = [key for key in results_file.keys() if key.startswith('est_eigs_arr_')]
    est_eigs_by_alg = {key.replace('est_eigs_arr_', ''): results_file[key][:] for key in est_keys}

mae_by_beta = {}
for alg_name, est_eigs in est_eigs_by_alg.items():
    abs_errors = np.abs(est_eigs - true_eigs)
    beta_maes = []
    for beta in DESIGN_EIG_PERCENTAGES:
        mask = np.isclose(design_eig_percentages, beta)
        beta_maes.append(abs_errors[mask].mean())
    mae_by_beta[alg_name] = np.array(beta_maes, dtype=np.float32)

os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)
with h5py.File(processed_results_filename, 'w') as out_file:
    out_file.create_dataset('design_eig_percentages', data=np.array(DESIGN_EIG_PERCENTAGES, dtype=np.float32))
    for alg_name, maes in mae_by_beta.items():
        out_file.create_dataset(f'mae_by_beta_{alg_name}', data=maes)
