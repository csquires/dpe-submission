import argparse
import os

import h5py
import numpy as np
import torch
from tqdm import trange
import yaml

from src.models.binary_classification import make_binary_classifier, make_pairwise_binary_classifiers
from src.models.multiclass_classification import make_multiclass_classifier
from src.density_ratio_estimation import BDRE, MDRE, TDRE, TSM
from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
from src.density_ratio_estimation.triangular_tsm import TriangularTSM
from src.density_ratio_estimation.spatial_adapters import make_spatial_velo_denoiser


class TriangularMDREELDRAdapter:
    """
    adapter for triangular mdre for eldr estimation.

    triangular mdre expects three distributions for fitting. for eldr:
    - samples_p0: joint samples (theta, y)
    - samples_p1: marginal samples (shuffled theta, shuffled y)
    - samples_pstar: same as p0 (use joint as reference)
    """
    def __init__(self, triangular_mdre):
        self.triangular_mdre = triangular_mdre

    def fit(self, samples_p0, samples_p1):
        # use samples_p0 as pstar (joint distribution)
        self.triangular_mdre.fit(samples_p0, samples_p1, samples_p0)

    def predict_ldr(self, xs):
        return self.triangular_mdre.predict_ldr(xs)


class TriangularTSMELDRAdapter:
    """
    adapter for triangular tsm for eldr estimation.

    triangular tsm expects three distributions for fitting. for eldr:
    - samples_p0: joint samples (theta, y)
    - samples_p1: marginal samples (shuffled theta, shuffled y)
    - samples_pstar: same as p0 (use joint as reference)
    """
    def __init__(self, triangular_tsm):
        self.triangular_tsm = triangular_tsm

    def fit(self, samples_p0, samples_p1):
        # use samples_p0 as pstar (joint distribution)
        self.triangular_tsm.fit(samples_p0, samples_p1, samples_p0)

    def predict_ldr(self, xs):
        return self.triangular_tsm.predict_ldr(xs)


class ELDRPlugin:
    """
    wraps a density ratio estimator for eldr estimation.

    eldr = e_{p0}[log(p0/p1)] where:
    - p0 is the joint distribution p(theta, y) = p(theta) p(y | theta)
    - p1 is the product of marginals p(theta) p(y)

    estimation procedure:
    1. fit dre on (samples_p0, samples_p1) pair
    2. predict ldr at p0 samples
    3. average ldrs to get eldr
    """
    def __init__(self, density_ratio_estimator):
        self.dre = density_ratio_estimator

    def _create_marginal_samples(self, samples_theta, samples_y):
        """
        create product-of-marginals samples by shuffling each component independently.

        args:
            samples_theta: [nsamples, theta_dim]
            samples_y: [nsamples, 1]

        returns:
            [nsamples, theta_dim + 1]
        """
        shuffled_theta = samples_theta[torch.randperm(samples_theta.shape[0])]
        shuffled_y = samples_y[torch.randperm(samples_y.shape[0])]
        return torch.cat([shuffled_theta, shuffled_y], dim=1)

    def estimate_eldr(self, samples_theta, samples_y):
        """
        estimate eldr for a (theta, y) pair.

        args:
            samples_theta: [nsamples, theta_dim] samples from p(theta)
            samples_y: [nsamples, 1] samples from p(y | theta)

        returns:
            float: estimated eldr
        """
        # p0: joint distribution
        samples_p0 = torch.cat([samples_theta, samples_y], dim=1)  # [nsamples, theta_dim+1]
        # p1: product of marginals
        samples_p1 = self._create_marginal_samples(samples_theta, samples_y)  # [nsamples, theta_dim+1]
        # fit and predict
        self.dre.fit(samples_p0, samples_p1)
        est_ldrs = self.dre.predict_ldr(samples_p0)  # [nsamples]
        # return mean ldr
        return torch.mean(est_ldrs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='force re-run of all algorithms, overwriting existing results')
    args = parser.parse_args()

    config = yaml.load(open('experiments/ising_eldr_estimation/config1.yaml', 'r'),
                       Loader=yaml.FullLoader)
    DEVICE = config['device']
    DATA_DIR = config['data_dir']
    RAW_RESULTS_DIR = config['raw_results_dir']
    THETA_DIM = config['grid_size'] ** 2  # grid_size x grid_size = theta_dim
    NSAMPLES = config['nsamples']
    SEED = config['seed']

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_filename = f'{DATA_DIR}/dataset_theta_dim={THETA_DIM},nsamples={NSAMPLES}.h5'
    results_filename = f'{RAW_RESULTS_DIR}/results_theta_dim={THETA_DIM},nsamples={NSAMPLES}.h5'

    # check input file exists
    if not os.path.exists(data_filename):
        raise FileNotFoundError(f"input data file not found: {data_filename}")

    # create output directory
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    # instantiate algorithms
    DATA_DIM = THETA_DIM + 1  # theta + y

    # bdre
    bdre_classifier = make_binary_classifier(name="default", input_dim=DATA_DIM)
    bdre = BDRE(bdre_classifier, device=DEVICE)
    bdre_plugin = ELDRPlugin(density_ratio_estimator=bdre)

    # tdre (5 waypoints)
    TDRE_WAYPOINTS = 5
    tdre_classifiers = make_pairwise_binary_classifiers(
        name="default",
        num_classes=TDRE_WAYPOINTS,
        input_dim=DATA_DIM,
    )
    tdre = TDRE(tdre_classifiers, num_waypoints=TDRE_WAYPOINTS, device=DEVICE)
    tdre_plugin = ELDRPlugin(density_ratio_estimator=tdre)

    # mdre (15 waypoints)
    MDRE_WAYPOINTS = 15
    mdre_classifier = make_multiclass_classifier(
        name="default",
        input_dim=DATA_DIM,
        num_classes=MDRE_WAYPOINTS,
    )
    mdre = MDRE(mdre_classifier, device=DEVICE)
    mdre_plugin = ELDRPlugin(density_ratio_estimator=mdre)

    # triangular mdre (15 waypoints)
    triangular_mdre_classifier = make_multiclass_classifier(
        name="default",
        input_dim=DATA_DIM,
        num_classes=MDRE_WAYPOINTS,
    )
    triangular_mdre = TriangularMDRE(triangular_mdre_classifier, device=DEVICE)
    triangular_mdre_adapter = TriangularMDREELDRAdapter(triangular_mdre)
    triangular_mdre_plugin = ELDRPlugin(density_ratio_estimator=triangular_mdre_adapter)

    # tsm
    tsm = TSM(input_dim=DATA_DIM, device=DEVICE)
    tsm_plugin = ELDRPlugin(density_ratio_estimator=tsm)

    # vfm (spatial denoiser)
    spatial_denoiser = make_spatial_velo_denoiser(input_dim=DATA_DIM, device=DEVICE)
    spatial_denoiser_plugin = ELDRPlugin(density_ratio_estimator=spatial_denoiser)

    algorithms = [
        ("BDRE", bdre_plugin),
        ("TDRE", tdre_plugin),
        ("MDRE", mdre_plugin),
        ("TriangularMDRE", triangular_mdre_plugin),
        ("TSM", tsm_plugin),
        ("VFM", spatial_denoiser_plugin),
    ]

    # main processing loop
    with h5py.File(data_filename, 'r') as data_file:
        # get list of design groups (sorted)
        design_groups = sorted([k for k in data_file.keys() if k.startswith('design_')])

        with h5py.File(results_filename, 'a') as results_file:
            for design_idx in trange(len(design_groups), desc='designs'):
                design_group_name = design_groups[design_idx]
                design_group = data_file[design_group_name]

                # create output design group if not exists
                if design_group_name not in results_file:
                    results_file.create_group(design_group_name)
                result_design_group = results_file[design_group_name]

                # get list of prior subgroups (sorted)
                prior_subgroups = sorted([k for k in design_group.keys() if k.startswith('prior_')])

                for prior_idx in trange(
                    len(prior_subgroups),
                    desc=f'{design_group_name}',
                    leave=False
                ):
                    prior_subgroup_name = prior_subgroups[prior_idx]
                    prior_subgroup = design_group[prior_subgroup_name]

                    # load data
                    theta_samples = torch.from_numpy(
                        prior_subgroup['theta_samples_arr'][:]
                    ).to(DEVICE)  # [nsamples, theta_dim]
                    y_samples = torch.from_numpy(
                        prior_subgroup['y_samples_arr'][:]
                    ).to(DEVICE)  # [nsamples, 1]
                    true_ldrs = prior_subgroup['true_ldrs_arr'][:]  # [nsamples]

                    # create output prior subgroup if not exists
                    if prior_subgroup_name not in result_design_group:
                        result_design_group.create_group(prior_subgroup_name)
                    result_prior_subgroup = result_design_group[prior_subgroup_name]

                    # compute and save ground truth eldr
                    if 'true_eldr' not in result_prior_subgroup or args.force:
                        true_eldr = np.mean(true_ldrs).astype(np.float32)
                        if 'true_eldr' in result_prior_subgroup:
                            del result_prior_subgroup['true_eldr']
                        result_prior_subgroup.create_dataset('true_eldr', data=true_eldr)

                    # run each algorithm
                    for alg_name, alg_plugin in algorithms:
                        dataset_name = f'est_eldr_{alg_name}'

                        # check if result exists
                        if dataset_name in result_prior_subgroup and not args.force:
                            print(f"  skipping {alg_name} (exists, use --force to overwrite)")
                            continue

                        print(f"  running {alg_name}...")

                        # estimate eldr
                        est_eldr = alg_plugin.estimate_eldr(theta_samples, y_samples)
                        est_eldr_value = est_eldr.item() if hasattr(est_eldr, 'item') else est_eldr

                        # save result
                        if dataset_name in result_prior_subgroup:
                            del result_prior_subgroup[dataset_name]
                        result_prior_subgroup.create_dataset(
                            dataset_name,
                            data=np.float32(est_eldr_value)
                        )

    print(f"results saved to {results_filename}")
