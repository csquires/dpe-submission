import os
import pickle

import yaml
from tqdm import tqdm
from torch.distributions import MultivariateNormal


config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']

DATA_DIM = config['data_dim']
KL_DISTANCES = config['kl_distances']
NSAMPLES_TRAIN = config['nsamples_train']
NSAMPLES_TEST = config['nsamples_test']
SEED = config['seed']

os.makedirs(RAW_RESULTS_DIR, exist_ok=True)


for kl_distance in KL_DISTANCES:
    datasets = pickle.load(open(f'{DATA_DIR}/d={DATA_DIM},k={kl_distance},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.pkl', 'rb'))

    true_ldrs_all = []
    for dataset in tqdm(datasets):
        mu0, Sigma0, mu1, Sigma1 = dataset['mu0'], dataset['Sigma0'], dataset['mu1'], dataset['Sigma1']
        p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
        p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)
        true_ldrs1 = p0.log_prob(dataset['samples_pstar1']) - p1.log_prob(dataset['samples_pstar1'])
        true_ldrs2 = p0.log_prob(dataset['samples_pstar2']) - p1.log_prob(dataset['samples_pstar2'])
        true_ldrs3 = p0.log_prob(dataset['samples_pstar3']) - p1.log_prob(dataset['samples_pstar3'])
        true_ldrs_all.append([true_ldrs1, true_ldrs2, true_ldrs3])

    pickle.dump(true_ldrs_all, open(f'{RAW_RESULTS_DIR}/true_ldrs_d={DATA_DIM},k={kl_distance}.pkl', 'wb'))
