import pickle

from src.density_ratio_estimation.bdre import BDRE
from src.density_ratio_estimation.tdre import TDRE
from src.density_ratio_estimation.mdre import MDRE
from src.density_ratio_estimation.tsm import TSM

from experiments.utils.dre_algorithm_runner import DREAlgorithmRunner


DATA_DIR = 'experiments/density_ratio_estimation/data'
kl_distances = pickle.load(open(f'{DATA_DIR}/kl_distances.pkl', 'rb'))
NSAMPLES = 1000
# alg_runner = DREAlgorithmRunner(algorithms=[BDRE(), TDRE(), MDRE(), TSM()])

for kl_distance in kl_distances:
    datasets = pickle.load(open(f'{DATA_DIR}/k={kl_distance},n={NSAMPLES}.pkl', 'rb'))
    for dataset in datasets:
        samples_p0 = dataset['samples_p0']
        samples_p1 = dataset['samples_p1']
        # alg_runner.run(samples_p0, samples_p1)




