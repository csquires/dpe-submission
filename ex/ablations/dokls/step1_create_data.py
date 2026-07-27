"""dokls step1: create dataset with prescribed KL gaussians and true ELDR labels.

generates dataset.h5 with 70 rows (7 kl × 10 instances, dim 3), p0/p1 samples,
two p* anchors (q0=midpoint, q1=distant) at 8192 samples each, per-sample
log-ratios, and true ELDR values. seeding is reproducible per (row, p*_idx).
"""
import os

# import from ex first to ensure yaml env-var patch is applied
from ex.utils.prescribed_kls import create_two_gaussians_kl_range
from ex.synth.model_selection.dists import test_dist_params, analytic_eldr

import h5py
import yaml
import numpy as np
import torch
from tqdm import tqdm
from torch.distributions import MultivariateNormal


def main(variant=None):
    """load config, generate dataset, write to hdf5."""
    # load config (assume dokls/config.yaml exists in same directory)
    config_dir = os.path.dirname(__file__)
    config_path = os.path.join(config_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # constants
    SEED = config['seed']
    KLS_TARGET = config['kl_distances']  # 7 values
    N_INSTANCES_PER_KL = config['num_instances_per_kl']
    DATA_DIM = config.get('data_dim', 3)
    MAX_N = config.get('nsamples_max', 8192)
    DATA_DIR = config['data_dir']

    # init global RNG
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # allocate arrays
    N_ROWS = len(KLS_TARGET) * N_INSTANCES_PER_KL
    kl_arr = np.zeros(N_ROWS, dtype=np.float32)
    mu0_arr = np.zeros((N_ROWS, DATA_DIM), dtype=np.float32)
    mu1_arr = np.zeros((N_ROWS, DATA_DIM), dtype=np.float32)
    Sigma0_arr = np.zeros((N_ROWS, DATA_DIM, DATA_DIM), dtype=np.float32)
    Sigma1_arr = np.zeros((N_ROWS, DATA_DIM, DATA_DIM), dtype=np.float32)

    samples_p0_arr = np.zeros((N_ROWS, MAX_N, DATA_DIM), dtype=np.float32)
    samples_p1_arr = np.zeros((N_ROWS, MAX_N, DATA_DIM), dtype=np.float32)
    pstar_arr = np.zeros((N_ROWS, 2, MAX_N, DATA_DIM), dtype=np.float32)
    true_ldrs_arr = np.zeros((N_ROWS, 2, MAX_N), dtype=np.float32)
    true_eldr_arr = np.zeros((N_ROWS, 2), dtype=np.float32)

    # main loop
    idx = 0
    for kl_idx, kl in enumerate(tqdm(KLS_TARGET, desc='kl')):
        pairs = create_two_gaussians_kl_range(
            dim=DATA_DIM, k=kl, beta_min=0.3, beta_max=0.7, npairs=N_INSTANCES_PER_KL
        )

        for instance_idx, pair in enumerate(pairs):
            row = kl_idx * N_INSTANCES_PER_KL + instance_idx

            # unpack pair
            mu0, Sigma0 = pair['mu0'], pair['Sigma0']
            mu1, Sigma1 = pair['mu1'], pair['Sigma1']

            # convert to numpy [batch dims]
            mu0_np = mu0.numpy()
            mu1_np = mu1.numpy()
            Sigma0_np = Sigma0.numpy()
            Sigma1_np = Sigma1.numpy()

            # store metadata
            kl_arr[row] = kl
            mu0_arr[row] = mu0_np
            mu1_arr[row] = mu1_np
            Sigma0_arr[row] = Sigma0_np
            Sigma1_arr[row] = Sigma1_np

            # create torch distributions
            p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
            p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

            # sample p0, p1 (seeded per row; use offset strategy)
            torch.manual_seed(SEED + row)
            samples_p0_arr[row] = p0.sample((MAX_N,)).numpy()

            torch.manual_seed(SEED + 1000 + row)
            samples_p1_arr[row] = p1.sample((MAX_N,)).numpy()

            # construct p* anchors: q0=pstar3 (midpoint), q1=pstar4 (distant)
            pstar_dists = test_dist_params(mu0, Sigma0, mu1, Sigma1, torch.sqrt)
            q0_dist = MultivariateNormal(pstar_dists[2][0], covariance_matrix=pstar_dists[2][1])
            q1_dist = MultivariateNormal(pstar_dists[3][0], covariance_matrix=pstar_dists[3][1])

            # sample q0 (seeded per row)
            torch.manual_seed(SEED + 2000 + row)
            pstar_arr[row, 0] = q0_dist.sample((MAX_N,)).numpy()

            # sample q1 (seeded per row)
            torch.manual_seed(SEED + 3000 + row)
            pstar_arr[row, 1] = q1_dist.sample((MAX_N,)).numpy()

            # compute log-ratios and ELDR for both anchors
            for p_idx, (q_dist, q_mu, q_Sigma) in enumerate([
                (q0_dist, pstar_dists[2][0], pstar_dists[2][1]),
                (q1_dist, pstar_dists[3][0], pstar_dists[3][1]),
            ]):
                samples_q = pstar_arr[row, p_idx]  # [MAX_N, 3]

                # per-sample log-ratios
                samples_q_tensor = torch.from_numpy(samples_q).to(torch.float32)
                log_p0 = p0.log_prob(samples_q_tensor)  # [MAX_N]
                log_p1 = p1.log_prob(samples_q_tensor)  # [MAX_N]
                true_ldrs_arr[row, p_idx] = (log_p0 - log_p1).numpy().astype(np.float32)

                # analytic ELDR (compute in float64, cast to float32)
                q_mu_np = q_mu.numpy().astype(np.float64)
                q_Sigma_np = q_Sigma.numpy().astype(np.float64)
                true_eldr_arr[row, p_idx] = analytic_eldr(
                    mu0_np.astype(np.float64),
                    Sigma0_np.astype(np.float64),
                    mu1_np.astype(np.float64),
                    Sigma1_np.astype(np.float64),
                    q_mu_np,
                    q_Sigma_np,
                ).astype(np.float32)

    # write to hdf5
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, 'dataset.h5')

    with h5py.File(output_path, 'w') as f:
        # data arrays
        f.create_dataset('samples_p0_arr', data=samples_p0_arr)
        f.create_dataset('samples_p1_arr', data=samples_p1_arr)
        f.create_dataset('pstar_arr', data=pstar_arr)
        f.create_dataset('true_ldrs_arr', data=true_ldrs_arr)
        f.create_dataset('true_eldr_arr', data=true_eldr_arr)

        # metadata arrays
        f.create_dataset('kl_arr', data=kl_arr)
        f.create_dataset('mu0_arr', data=mu0_arr)
        f.create_dataset('mu1_arr', data=mu1_arr)
        f.create_dataset('Sigma0_arr', data=Sigma0_arr)
        f.create_dataset('Sigma1_arr', data=Sigma1_arr)

        # file attributes
        f.attrs['data_dim'] = DATA_DIM
        f.attrs['max_n'] = MAX_N
        f.attrs['n_rows'] = N_ROWS
        f.attrs['n_kls'] = len(KLS_TARGET)
        f.attrs['seed'] = SEED

    # validate
    assert samples_p0_arr.shape == (N_ROWS, MAX_N, DATA_DIM), f"p0 {samples_p0_arr.shape}"
    assert samples_p1_arr.shape == (N_ROWS, MAX_N, DATA_DIM), f"p1 {samples_p1_arr.shape}"
    assert pstar_arr.shape == (N_ROWS, 2, MAX_N, DATA_DIM), f"pstar {pstar_arr.shape}"
    assert true_ldrs_arr.shape == (N_ROWS, 2, MAX_N), f"ldrs {true_ldrs_arr.shape}"
    assert true_eldr_arr.shape == (N_ROWS, 2), f"eldr {true_eldr_arr.shape}"
    assert kl_arr.shape == (N_ROWS,), f"kl {kl_arr.shape}"
    assert mu0_arr.shape == (N_ROWS, DATA_DIM), f"mu0 {mu0_arr.shape}"
    assert Sigma0_arr.shape == (N_ROWS, DATA_DIM, DATA_DIM), f"Sigma0 {Sigma0_arr.shape}"

    assert samples_p0_arr.dtype == np.float32
    assert true_ldrs_arr.dtype == np.float32
    assert true_eldr_arr.dtype == np.float32

    # kl ordering
    expected_kls = np.array(KLS_TARGET, dtype=np.float32)
    actual_kls = kl_arr[::N_INSTANCES_PER_KL]
    assert np.allclose(actual_kls, expected_kls), f"kl mismatch: {actual_kls} vs {expected_kls}"

    # sanity: no nan/inf
    assert not np.any(np.isnan(true_ldrs_arr)), "nan in true_ldrs_arr"
    assert not np.any(np.isinf(true_ldrs_arr)), "inf in true_ldrs_arr"
    assert not np.any(np.isnan(true_eldr_arr)), "nan in true_eldr_arr"
    assert not np.any(np.isinf(true_eldr_arr)), "inf in true_eldr_arr"

    # covariance symmetry
    for i in range(N_ROWS):
        assert np.allclose(Sigma0_arr[i], Sigma0_arr[i].T), f"Sigma0[{i}] not symmetric"
        assert np.allclose(Sigma1_arr[i], Sigma1_arr[i].T), f"Sigma1[{i}] not symmetric"

    # positive definiteness
    for i in range(N_ROWS):
        eigs0 = np.linalg.eigvalsh(Sigma0_arr[i])
        eigs1 = np.linalg.eigvalsh(Sigma1_arr[i])
        assert np.all(eigs0 > -1e-6), f"Sigma0[{i}] not positive definite: {eigs0}"
        assert np.all(eigs1 > -1e-6), f"Sigma1[{i}] not positive definite: {eigs1}"

    print(f"dataset written to {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='dokls step1: create dataset')
    parser.add_argument('--variant', type=str, default=None, help='unused; for compat')
    args = parser.parse_args()
    main(args.variant)
