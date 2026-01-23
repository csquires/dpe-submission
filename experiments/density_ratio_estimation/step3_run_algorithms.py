import os
import pickle

import yaml
from tqdm import tqdm
import pandas as pd

from src.density_ratio_estimation.bdre import BDRE
from src.density_ratio_estimation.tdre import TDRE



config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']

DATA_DIM = config['data_dim']
KL_DISTANCES = config['kl_distances']
NSAMPLES_TRAIN = config['nsamples_train']
NSAMPLES_TEST = config['nsamples_test']
SEED = config['seed']


os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
bdre = BDRE(DATA_DIM)
tdre = TDRE(DATA_DIM)

results = []
for kl_distance in KL_DISTANCES:
    datasets = pickle.load(open(f'{DATA_DIR}/d={DATA_DIM},k={kl_distance},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.pkl', 'rb'))

    print(f'Running BDRE for kl_distance={kl_distance}')
    for instance_idx, dataset in enumerate(tqdm(datasets)):
        samples_p0 = dataset['samples_p0']
        samples_p1 = dataset['samples_p1']
        all_test_samples = [dataset['samples_pstar1'], dataset['samples_pstar2'], dataset['samples_pstar3']]
        for test_set_idx in range(3):
            bdre.fit(samples_p0, samples_p1)
            est_ldrs = bdre.predict_ldr(all_test_samples[test_set_idx])
            results.append({
                "kl_distance": kl_distance,
                "test_set_idx": test_set_idx,
                "instance_idx": instance_idx,
                "algorithm": "bdre",
                "est_ldrs": est_ldrs
            })
    
    print(f'Running TDRE for kl_distance={kl_distance}')
    for instance_idx, dataset in enumerate(tqdm(datasets)):
        samples_p0 = dataset['samples_p0']
        samples_p1 = dataset['samples_p1']
        all_test_samples = [dataset['samples_pstar1'], dataset['samples_pstar2'], dataset['samples_pstar3']]
        for test_set_idx in range(3):
            tdre.fit(samples_p0, samples_p1)
            est_ldrs = tdre.predict_ldr(all_test_samples[test_set_idx])
            results.append({
                "kl_distance": kl_distance,
                "test_set_idx": test_set_idx,
                "instance_idx": instance_idx,
                "algorithm": "tdre",
                "est_ldrs": est_ldrs
            })
    
results_df = pd.DataFrame(results)
pickle.dump(results_df, open(f'{RAW_RESULTS_DIR}/results.pkl', 'wb'))
