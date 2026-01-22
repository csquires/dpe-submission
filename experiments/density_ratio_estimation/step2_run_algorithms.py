import os
import pickle

from tqdm import tqdm

from src.density_ratio_estimation.bdre import BDRE
from src.density_ratio_estimation.tdre import TDRE

from experiments.utils.dre_algorithm_runner import DREAlgorithmRunner


DATA_DIM = 3
DATA_DIR = 'experiments/density_ratio_estimation/data'
RAW_RESULTS_DIR = 'experiments/density_ratio_estimation/raw_results'
os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
kl_distances = pickle.load(open(f'{DATA_DIR}/kl_distances.pkl', 'rb'))
NSAMPLES_TRAIN = 1000
NSAMPLES_TEST = 1000
alg_runner = DREAlgorithmRunner()


for kl_distance in tqdm(kl_distances):
    datasets = pickle.load(open(f'{DATA_DIR}/d={DATA_DIM},k={kl_distance},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.pkl', 'rb'))

    bdre_results_all = []
    for dataset in datasets:
        samples_p0 = dataset['samples_p0']
        samples_p1 = dataset['samples_p1']
        all_test_samples = [dataset['samples_pstar1'], dataset['samples_pstar2'], dataset['samples_pstar3']]
        bdre_results = alg_runner.run(samples_p0, samples_p1, all_test_samples, BDRE(DATA_DIM))
        bdre_results_all.append(bdre_results)
    
    tdre_results_all = []
    for dataset in datasets:
        samples_p0 = dataset['samples_p0']
        samples_p1 = dataset['samples_p1']
        all_test_samples = [dataset['samples_pstar1'], dataset['samples_pstar2'], dataset['samples_pstar3']]
        tdre_results = alg_runner.run(samples_p0, samples_p1, all_test_samples, TDRE(DATA_DIM))
        tdre_results_all.append(tdre_results)

    pickle.dump(bdre_results_all, open(f'{RAW_RESULTS_DIR}/bdre_results_d={DATA_DIM},k={kl_distance}.pkl', 'wb'))
    pickle.dump(tdre_results_all, open(f'{RAW_RESULTS_DIR}/tdre_results_d={DATA_DIM},k={kl_distance}.pkl', 'wb'))
