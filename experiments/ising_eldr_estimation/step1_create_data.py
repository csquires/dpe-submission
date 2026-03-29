"""data generation for ising model eldr estimation experiments

generates (theta, y) sample pairs where theta ~ n(0, i) and y are ising model
samples conditioned on theta. computes ground truth log density ratios using
analytical formula and stores in hierarchical hdf5 structure organized by
design parameter.

structure:
- load config from config1.yaml
- build ising lattice once
- for each design parameter:
  - for each prior sample:
    - sample theta ~ n(0, i_dim)
    - initialize ising gibbs sampler with h = theta * design
    - collect nsamples of y via mcmc
    - compute ground truth ldrs: (theta^t y) / design - 128 / design^2
    - write to hdf5 incrementally per prior

hdf5 structure:
  design_k/
    theta_samples_arr: [num_priors, nsamples, dim]
    y_samples_arr: [num_priors, nsamples, dim]
    true_ldrs_arr: [num_priors, nsamples]
    design_value: scalar attribute
"""
import os
import h5py
import yaml
from tqdm import trange, tqdm
import numpy as np
import torch

from src.utils.ising_lattice import build_ising_adjacency
from src.sampling.ising_gibbs import IsingGibbsSampler


def main():
    # load config
    config_path = 'experiments/ising_eldr_estimation/config1.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # extract config parameters
    data_dir = config['data_dir']
    num_priors = config['num_priors']
    designs = config['designs']
    nsamples = config['nsamples']
    burnin = config['gibbs_burnin']
    seed = config['seed']
    grid_size = config['grid_size']

    # set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # constants
    dim = grid_size ** 2

    # build ising adjacency once
    adj = build_ising_adjacency(grid_size)

    # estimate disk space
    estimated_size_gb = (
        len(designs) * num_priors * nsamples * dim
        * (4 + 4 + 4) / (1024**3)  # float32 + int32 + float32
    )
    print(f"estimated hdf5 size: {estimated_size_gb:.2f} gb")
    if estimated_size_gb > 50:
        print(f"warning: estimated size exceeds 50 gb")

    # create data directory
    os.makedirs(data_dir, exist_ok=True)
    dataset_filename = f'{data_dir}/ising_eldr_d={dim},designs={len(designs)}.h5'

    # main generation loop with hierarchical structure
    try:
        with h5py.File(dataset_filename, 'a') as f:
            for design in designs:
                group_name = f'design_{design}'
                print(f"\nprocessing design {design}")

                # create group and resizable datasets
                if group_name not in f:
                    grp = f.create_group(group_name)
                    grp.attrs['design_value'] = design

                    # create resizable datasets
                    grp.create_dataset(
                        'theta_samples_arr',
                        shape=(0, nsamples, dim),
                        maxshape=(None, nsamples, dim),
                        dtype=np.float32
                    )
                    grp.create_dataset(
                        'y_samples_arr',
                        shape=(0, nsamples, dim),
                        maxshape=(None, nsamples, dim),
                        dtype=np.int32
                    )
                    grp.create_dataset(
                        'true_ldrs_arr',
                        shape=(0, nsamples),
                        maxshape=(None, nsamples),
                        dtype=np.float32
                    )
                else:
                    grp = f[group_name]

                # populate with priors
                for prior_idx in trange(num_priors, desc=f"design {design}: priors", leave=False):
                    # sample theta ~ n(0, i_dim)
                    theta = torch.randn(dim)  # [dim]

                    # initialize sampler
                    h = theta.numpy() * design  # [dim]
                    sampler = IsingGibbsSampler(
                        adjacency=adj,
                        h=h,
                        temperature=1.0
                    )

                    # collect y samples
                    y_samples = np.zeros((nsamples, dim), dtype=np.int32)  # [nsamples, dim]

                    for sample_idx in trange(nsamples, desc="sampling y", leave=False):
                        if sample_idx == 0:
                            # burnin on first iteration
                            sampler.step(burnin_steps=burnin)
                        y_samples[sample_idx] = sampler.step(burnin_steps=0)  # [dim]

                    # compute ground truth ldrs
                    # formula: (theta^t y) / design - 128 / design^2
                    dot_products = y_samples @ theta.numpy()  # [nsamples]
                    ldrs = (dot_products / design) - (128.0 / (design ** 2))  # [nsamples]

                    # convert to float32
                    ldrs = ldrs.astype(np.float32)
                    theta_np = theta.numpy().astype(np.float32)

                    # resize datasets and write incrementally
                    current_size = grp['theta_samples_arr'].shape[0]
                    grp['theta_samples_arr'].resize((current_size + 1, nsamples, dim))
                    grp['y_samples_arr'].resize((current_size + 1, nsamples, dim))
                    grp['true_ldrs_arr'].resize((current_size + 1, nsamples))

                    # write data for this prior
                    # replicate theta across samples
                    theta_replicated = np.tile(theta_np, (nsamples, 1))  # [nsamples, dim]
                    grp['theta_samples_arr'][current_size] = theta_replicated  # [nsamples, dim]
                    grp['y_samples_arr'][current_size] = y_samples  # [nsamples, dim]
                    grp['true_ldrs_arr'][current_size] = ldrs  # [nsamples]

        print(f"\ndataset saved to {dataset_filename}")
        print(f"structure: {len(designs)} designs, {num_priors} priors each, {nsamples} samples each")

    except Exception as e:
        print(f"\nerror during data generation: {e}")
        print(f"progress: check {dataset_filename} for partial results")
        raise


if __name__ == '__main__':
    main()
