import os

import h5py
import numpy as np
import torch
from tqdm import trange
import yaml

from src.models.binary_classification import make_binary_classifier, make_pairwise_binary_classifiers
from src.models.multiclass_classification import make_multiclass_classifier
from src.density_ratio_estimation import BDRE, MDRE, TDRE, TSM, TriangularTSM
from src.eig_estimation.plugin import EIGPlugin


config = yaml.load(open('experiments/eig_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
DEVICE = config['device']
# directories
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
# dataset parameters
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']
# random seed
SEED = config['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)

dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM}.h5'
results_filename = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM}.h5'

# instantiate bdre plugin
bdre_classifier = make_binary_classifier(name="default", input_dim=DATA_DIM+1)
bdre = BDRE(bdre_classifier, device=DEVICE)
bdre_plugin = EIGPlugin(density_ratio_estimator=bdre)

algorithms = [
    ("BDREPlugin", bdre_plugin),
]


os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
with h5py.File(dataset_filename, 'r') as dataset_file:
    nrows = dataset_file['design_arr'].shape[0]

    for alg_name, alg in algorithms:
        est_eigs_arr = np.zeros(nrows)
        for idx in trange(nrows):
            design = torch.from_numpy(dataset_file['design_arr'][idx]).to(DEVICE)  # (DATA_DIM, 1)
            theta_samples = torch.from_numpy(dataset_file['theta_samples_arr'][idx]).to(DEVICE)  # (NSAMPLES, DATA_DIM)
            y_samples = torch.from_numpy(dataset_file['y_samples_arr'][idx]).to(DEVICE)  # (NSAMPLES, 1)

            alg.estimate_eig(theta_samples, y_samples)
