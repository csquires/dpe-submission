import os
import pickle

import yaml
from tqdm import tqdm

from src.density_ratio_estimation.bdre import BDRE
from src.density_ratio_estimation.tdre import TDRE

from experiments.utils.dre_algorithm_runner import DREAlgorithmRunner


config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']

DATA_DIM = config['data_dim']
KL_DISTANCES = config['kl_distances']
NSAMPLES_TRAIN = config['nsamples_train']
NSAMPLES_TEST = config['nsamples_test']
SEED = config['seed']


os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
alg_runner = DREAlgorithmRunner()


for kl_distance in tqdm(KL_DISTANCES):
    datasets = pickle.load(open(f'{DATA_DIR}/d={DATA_DIM},k={kl_distance},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.pkl', 'rb'))

    print(f'Running BDRE for kl_distance={kl_distance}')
    bdre_results_all = []
    for dataset in datasets:
        samples_p0 = dataset['samples_p0']
        samples_p1 = dataset['samples_p1']
        all_test_samples = [dataset['samples_pstar1'], dataset['samples_pstar2'], dataset['samples_pstar3']]
        bdre_results = alg_runner.run(samples_p0, samples_p1, all_test_samples, BDRE(DATA_DIM))
        bdre_results_all.append(bdre_results)
    
    print(f'Running TDRE for kl_distance={kl_distance}')
    tdre_results_all = []
    for dataset in datasets:
        samples_p0 = dataset['samples_p0']
        samples_p1 = dataset['samples_p1']
        all_test_samples = [dataset['samples_pstar1'], dataset['samples_pstar2'], dataset['samples_pstar3']]
        tdre_results = alg_runner.run(samples_p0, samples_p1, all_test_samples, TDRE(DATA_DIM))
        tdre_results_all.append(tdre_results)

    pickle.dump(bdre_results_all, open(f'{RAW_RESULTS_DIR}/bdre_results_d={DATA_DIM},k={kl_distance}.pkl', 'wb'))
    pickle.dump(tdre_results_all, open(f'{RAW_RESULTS_DIR}/tdre_results_d={DATA_DIM},k={kl_distance}.pkl', 'wb'))
