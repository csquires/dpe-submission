import pickle

import yaml
import numpy as np
import torch


config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)

RAW_RESULTS_DIR = config['raw_results_dir']
DATA_DIM = config['data_dim']
KL_DISTANCES = config['kl_distances']
NSAMPLES_TRAIN = config['nsamples_train']
NSAMPLES_TEST = config['nsamples_test']
SEED = config['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)


for kl_distance in KL_DISTANCES:
    raw_results = pickle.load(open(f'{RAW_RESULTS_DIR}/bdre_results_d={DATA_DIM},k={kl_distance}.pkl', 'rb'))
    for dataset in raw_results:
        ldrs_pstar1 = dataset[0]
        ldrs_pstar2 = dataset[1]
        ldrs_pstar3 = dataset[2]