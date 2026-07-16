import os

import h5py  # HDF5
from tqdm import tqdm
import numpy as np
import torch
from torch.distributions import MultivariateNormal

from ex.utils.prescribed_kls import create_two_gaussians_kl_range
from ex.synth.model_selection.dists import test_dist_params, true_eldr_arr
from ex.synth.model_selection.variants import resolve


def main(variant=None):
    tag, config = resolve(variant)
    # directories
    DATA_DIR = config['data_dir']
    # dataset parameters
    DATA_DIM = config['data_dim']
    GAMMA = config['gamma']
    KL_DISTANCES = config['kl_distances']
    NUM_INSTANCES_PER_KL = config['num_instances_per_kl']
    NSAMPLES_TRAIN = config['nsamples_train']
    NSAMPLES_TEST = config['nsamples_test']
    NTEST_SETS = config['ntest_sets']
    # random seed
    SEED = config['seed']
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # dataset storage
    nrows = len(KL_DISTANCES) * NUM_INSTANCES_PER_KL
    kl_distance_arr = np.zeros(nrows, dtype=np.float32)
    # true parameters (metadata)
    mu0_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
    mu1_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
    Sigma0_arr = np.zeros((nrows, DATA_DIM, DATA_DIM), dtype=np.float32)
    Sigma1_arr = np.zeros((nrows, DATA_DIM, DATA_DIM), dtype=np.float32)
    # data itself
    samples_p0_arr = np.zeros((nrows, NSAMPLES_TRAIN, DATA_DIM), dtype=np.float32)
    samples_p1_arr = np.zeros((nrows, NSAMPLES_TRAIN, DATA_DIM), dtype=np.float32)
    samples_pstar_arr = np.zeros((nrows, NTEST_SETS, NSAMPLES_TEST, DATA_DIM), dtype=np.float32)
    samples_pstar_train_arr = np.zeros((nrows, NSAMPLES_TRAIN, DATA_DIM), dtype=np.float32)
    # true density ratios
    true_ldrs_arr = np.zeros((nrows, NTEST_SETS, NSAMPLES_TEST), dtype=np.float32)

    idx = 0
    for kl_distance in tqdm(KL_DISTANCES):
        gaussian_pairs = create_two_gaussians_kl_range(dim=DATA_DIM, k=kl_distance, beta_min=0.3, beta_max=0.7, npairs=NUM_INSTANCES_PER_KL)
        for gaussian_pair in gaussian_pairs:
            mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
            mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']
            # define the two distributions whose density ratios we want to estimate
            p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
            p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)
            # define test distributions via the shared recipe (kept in sync with the
            # closed-form ELDR in dists.py): [p0, p1, midpoint, distant gaussian]
            pstar1, pstar2, pstar3, pstar4 = [
                MultivariateNormal(m, covariance_matrix=S)
                for m, S in test_dist_params(mu0, Sigma0, mu1, Sigma1, torch.sqrt)
            ]

            # store parameters
            kl_distance_arr[idx] = kl_distance
            mu0_arr[idx] = mu0.numpy()
            mu1_arr[idx] = mu1.numpy()
            Sigma0_arr[idx] = Sigma0.numpy()
            Sigma1_arr[idx] = Sigma1.numpy()

            # draw and store samples from training distributions
            samples_p0_arr[idx] = p0.sample((NSAMPLES_TRAIN,)).numpy()
            samples_p1_arr[idx] = p1.sample((NSAMPLES_TRAIN,)).numpy()
            # draw and store samples from test distributions
            samples_pstar1 = pstar1.sample((NSAMPLES_TEST,))
            samples_pstar2 = pstar2.sample((NSAMPLES_TEST,))
            samples_pstar3 = pstar3.sample((NSAMPLES_TEST,))
            samples_pstar4 = pstar4.sample((NSAMPLES_TEST,))
            samples_pstar_arr[idx] = torch.stack([samples_pstar1, samples_pstar2, samples_pstar3, samples_pstar4], dim=0).numpy()

            samples_pstar_train = pstar4.sample((NSAMPLES_TRAIN,))
            samples_pstar_train_arr[idx] = samples_pstar_train.numpy()

            # compute true ldrs
            true_ldrs1 = p0.log_prob(samples_pstar1) - p1.log_prob(samples_pstar1)
            true_ldrs2 = p0.log_prob(samples_pstar2) - p1.log_prob(samples_pstar2)
            true_ldrs3 = p0.log_prob(samples_pstar3) - p1.log_prob(samples_pstar3)
            true_ldrs4 = p0.log_prob(samples_pstar4) - p1.log_prob(samples_pstar4)
            true_ldrs_arr[idx] = torch.stack([true_ldrs1, true_ldrs2, true_ldrs3, true_ldrs4], dim=0).numpy()

            idx += 1

    # closed-form population ELDR per (row, test_set): exact ground truth for step3,
    # free of the MC noise in true_ldrs_arr.mean(axis=2).
    true_eldr_analytic_arr = true_eldr_arr(
        mu0_arr, Sigma0_arr, mu1_arr, Sigma1_arr).astype(np.float32)

    os.makedirs(DATA_DIR, exist_ok=True)
    with h5py.File(f'{DATA_DIR}/dataset_newpstar.h5', 'w') as f:
    # with h5py.File(f'{DATA_DIR}/dataset_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST},ntestsets={NTEST_SETS}.h5', 'w') as f:
        f.create_dataset('kl_distance_arr', data=kl_distance_arr)
        f.create_dataset('mu0_arr', data=mu0_arr)
        f.create_dataset('mu1_arr', data=mu1_arr)
        f.create_dataset('Sigma0_arr', data=Sigma0_arr)
        f.create_dataset('Sigma1_arr', data=Sigma1_arr)
        f.create_dataset('samples_p0_arr', data=samples_p0_arr)
        f.create_dataset('samples_p1_arr', data=samples_p1_arr)
        f.create_dataset('samples_pstar_arr', data=samples_pstar_arr)
        f.create_dataset('samples_pstar_train_arr', data=samples_pstar_train_arr)
        f.create_dataset('true_ldrs_arr', data=true_ldrs_arr)
        f.create_dataset('true_eldr_analytic_arr', data=true_eldr_analytic_arr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Step 1: Create dataset with variant selection.')
    parser.add_argument('--variant', type=str, default=None,
                        help='Variant tag (precedence: --variant > $DPE_MS_VARIANT > DEFAULT_TAG)')
    args = parser.parse_args()
    main(args.variant)
