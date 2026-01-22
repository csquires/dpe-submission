import pickle

from src.density_ratio_estimation.bdre import BDRE
from src.density_ratio_estimation.tdre import TDRE
from src.density_ratio_estimation.mdre import MDRE
from src.density_ratio_estimation.tsm import TSM

from experiments.utils.dre_algorithm_runner import DREAlgorithmRunner


DATA_DIR = 'experiments/density_ratio_estimation/data'
kl_distances = pickle.load(open(f'{DATA_DIR}/kl_distances.pkl', 'rb'))
NSAMPLES = 1000
alg_runner = DREAlgorithmRunner()

for kl_distance in kl_distances:
    datasets = pickle.load(open(f'{DATA_DIR}/k={kl_distance},n={NSAMPLES}.pkl', 'rb'))
    for dataset in datasets:
        samples_p0 = dataset['samples_p0']
        samples_p1 = dataset['samples_p1']
        all_test_samples = [dataset['samples_pstar1'], dataset['samples_pstar2'], dataset['samples_pstar3']]
        bdre_results = alg_runner.run(samples_p0, samples_p1, all_test_samples, BDRE())
        tdre_results = alg_runner.run(samples_p0, samples_p1, all_test_samples, TDRE())
        mdre_results = alg_runner.run(samples_p0, samples_p1, all_test_samples, MDRE())
        tsm_results = alg_runner.run(samples_p0, samples_p1, all_test_samples, TSM())




