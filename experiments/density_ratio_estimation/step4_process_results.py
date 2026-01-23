import os
import pickle

import yaml
import numpy as np
import torch


config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)

RAW_RESULTS_DIR = config['raw_results_dir']
PROCESSED_RESULTS_DIR = config['processed_results_dir']
DATA_DIM = config['data_dim']
KL_DISTANCES = config['kl_distances']
NSAMPLES_TRAIN = config['nsamples_train']
NSAMPLES_TEST = config['nsamples_test']
SEED = config['seed']
NUM_INSTANCES = config['num_instances']

os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)


for kl_distance in KL_DISTANCES:
    bdre_results = pickle.load(open(f'{RAW_RESULTS_DIR}/bdre_results_d={DATA_DIM},k={kl_distance}.pkl', 'rb'))
    tdre_results = pickle.load(open(f'{RAW_RESULTS_DIR}/tdre_results_d={DATA_DIM},k={kl_distance}.pkl', 'rb'))
    true_ldrs = pickle.load(open(f'{RAW_RESULTS_DIR}/true_ldrs_d={DATA_DIM},k={kl_distance}.pkl', 'rb'))
    
    bdre_maes = np.zeros((NUM_INSTANCES, 3))
    # tdre_maes = np.zeros(NUM_INSTANCES)
    for instance_idx in range(NUM_INSTANCES):
        true_ldrs_pstar1 = true_ldrs[instance_idx][0]
        true_ldrs_pstar2 = true_ldrs[instance_idx][1]
        true_ldrs_pstar3 = true_ldrs[instance_idx][2]

        # load BDRE results
        bdre_ldrs_pstar1 = bdre_results[instance_idx][0]
        bdre_ldrs_pstar2 = bdre_results[instance_idx][1]
        bdre_ldrs_pstar3 = bdre_results[instance_idx][2]
        
        # compute the mean absolute error for each test set
        bdre_maes[instance_idx, 0] = torch.mean(torch.abs(bdre_ldrs_pstar1 - true_ldrs_pstar1))
        bdre_maes[instance_idx, 1] = torch.mean(torch.abs(bdre_ldrs_pstar2 - true_ldrs_pstar2))
        bdre_maes[instance_idx, 2] = torch.mean(torch.abs(bdre_ldrs_pstar3 - true_ldrs_pstar3))

    pickle.dump(bdre_maes, open(f'{PROCESSED_RESULTS_DIR}/bdre_maes_d={DATA_DIM},k={kl_distance}.pkl', 'wb'))