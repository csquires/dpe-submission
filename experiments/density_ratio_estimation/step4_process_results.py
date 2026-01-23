import os
import pickle

import yaml
import numpy as np
import pandas as pd
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


raw_results = pickle.load(open(f'{RAW_RESULTS_DIR}/results.pkl', 'rb'))

results = []
for kl_distance in KL_DISTANCES:
    true_ldrs = pickle.load(open(f'{RAW_RESULTS_DIR}/true_ldrs_d={DATA_DIM},k={kl_distance}.pkl', 'rb'))
    subset = raw_results[raw_results["kl_distance"] == kl_distance]
    for _, row in subset.iterrows():
        instance_idx = int(row["instance_idx"])
        test_set_idx = int(row["test_set_idx"])
        est_ldrs = row["est_ldrs"]
        true_ldrs_pstar = true_ldrs[instance_idx][test_set_idx]
        bdre_mae = torch.mean(torch.abs(est_ldrs - true_ldrs_pstar)).item()
        results.append({
            "kl_distance": kl_distance,
            "test_set_idx": test_set_idx,
            "algorithm": row["algorithm"],
            "mae": bdre_mae
        })

results_df = pd.DataFrame.from_dict(results)
os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)
pickle.dump(results_df, open(f'{PROCESSED_RESULTS_DIR}/results_d={DATA_DIM}.pkl', 'wb'))
