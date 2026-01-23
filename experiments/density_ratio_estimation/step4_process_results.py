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


# set up results dictionary structure
results = dict()
test_sets = ['pstar1', 'pstar2', 'pstar3']
algorithms = ['bdre', 'tdre', 'mdre', 'tsm']
for test_set in test_sets:
    results[test_set] = dict()
    for algorithm in algorithms:
        results[test_set][algorithm] = np.zeros(len(KL_DISTANCES))

# fill results dictionary
for kl_idx, kl_distance in enumerate(KL_DISTANCES):
    bdre_results = pickle.load(open(f'{RAW_RESULTS_DIR}/bdre_results_d={DATA_DIM},k={kl_distance}.pkl', 'rb'))
    tdre_results = pickle.load(open(f'{RAW_RESULTS_DIR}/tdre_results_d={DATA_DIM},k={kl_distance}.pkl', 'rb'))
    true_ldrs = pickle.load(open(f'{RAW_RESULTS_DIR}/true_ldrs_d={DATA_DIM},k={kl_distance}.pkl', 'rb'))
    
    bdre_maes = np.zeros((NUM_INSTANCES, 3))
    tdre_maes = np.zeros((NUM_INSTANCES, 3))
    for instance_idx in range(NUM_INSTANCES):
        for pstar_idx in range(3):
            true_ldrs_pstar = true_ldrs[instance_idx][pstar_idx]

            # BDRE
            bdre_ldrs_pstar = bdre_results[instance_idx][pstar_idx]
            bdre_maes[instance_idx, pstar_idx] = torch.mean(torch.abs(bdre_ldrs_pstar - true_ldrs_pstar))

    # BDRE
    results["pstar1"]["bdre"][kl_idx] = bdre_maes[:, 0].mean()
    results["pstar2"]["bdre"][kl_idx] = bdre_maes[:, 1].mean()
    results["pstar3"]["bdre"][kl_idx] = bdre_maes[:, 2].mean()

pickle.dump(results, open(f'{PROCESSED_RESULTS_DIR}/results_d={DATA_DIM}.pkl', 'wb'))