import argparse
import math
import os
import sys
import time

# add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import h5py
import numpy as np
import torch
from tqdm import trange
import yaml

from src.models.multiclass_classification import make_multiclass_classifier
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D
from src.models.binary_classification.default_binary_classifier import DefaultBinaryClassifier
from src.models.binary_classification.multi_head_binary_classifier import MultiHeadBinaryClassifier
from src.density_ratio_estimation.tsm import TSM
from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
from src.density_ratio_estimation.triangular_tdre import TriangularTDRE
from src.density_ratio_estimation.mh_triangular_tdre import MultiHeadTriangularTDRE
from src.eig_estimation.plugin import EIGPlugin


def compute_true_eig(Sigma_pi: torch.Tensor, xi: torch.Tensor, sigma2: float = 1.0) -> torch.Tensor:
    """
    Compute true EIG for Gaussian linear model.

    args:
        Sigma_pi: prior covariance, shape [data_dim, data_dim]
        xi: design vector, shape [data_dim, 1]
        sigma2: observation noise variance

    returns:
        scalar torch.Tensor with true EIG value
    """
    quad = (xi.T @ Sigma_pi @ xi).squeeze()
    return 0.5 * torch.log1p(quad / sigma2)


class TriangularDREAdapter:
    """
    Adapter bridging triangular DRE methods to EIGPlugin interface.

    Triangular DRE methods require fit(samples_p0, samples_p1, samples_pstar),
    while EIGPlugin expects fit(samples_p0, samples_p1).
    This adapter uses samples_p0 (joint distribution) as pstar.
    """

    def __init__(self, dre):
        self.dre = dre

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        self.dre.fit(samples_p0, samples_p1, samples_p0)

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        return self.dre.predict_ldr(xs)


def validate_subset(user_values, config_values, param_name):
    """
    Return filtered config list preserving order, or raise ValueError.

    If user_values is None/empty, return all config_values.
    Otherwise, validate that all user_values exist in config_values
    and return only those from config_values that appear in user_values,
    preserving config_values' order.
    """
    if not user_values:
        return config_values
    invalid = [v for v in user_values if v not in set(config_values)]
    if invalid:
        raise ValueError(f"invalid {param_name}: {invalid}\navailable: {config_values}")
    return [v for v in config_values if v in set(user_values)]


def main():
    # === 1. COMMAND-LINE ARGUMENT PARSING ===
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of all algorithms, overwriting existing results"
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="algorithms to run (default: all from config)"
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="*",
        type=int,
        default=None,
        help="hidden dimensions to run (default: all from config)"
    )
    args = parser.parse_args()

    # === 1.5 VALIDATE CLI ARGUMENTS ===
    config_path = "experiments/hidden_dim_scaling/config.yaml"
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    try:
        # validate methods if provided
        config_methods = ["TriangularMDRE", "TriangularTDRE", "MultiHeadTriangularTDRE", "TSM"]
        methods = validate_subset(args.methods, config_methods, "methods")

        # validate hidden_dims if provided
        hidden_dims = validate_subset(args.hidden_dims, config['hidden_dims'], "hidden_dims")
    except ValueError as e:
        parser.error(str(e))

    # === 2. CONFIG LOADING AND PARAMETER EXTRACTION ===
    HIDDEN_DIMS = hidden_dims
    NUM_WAYPOINTS = config['num_waypoints']
    NUM_LAYERS = config['num_layers']
    TSM_N_EPOCHS = config['tsm_n_epochs']
    TSM_BATCH_SIZE = config['tsm_batch_size']
    TSM_LR = config['tsm_lr']
    DEVICE = config['device']
    SEED = config['seed']
    DATA_DIR = config['data_dir']
    RAW_RESULTS_DIR = config['raw_results_dir']
    DATA_DIM = config['source_data_dim']
    NSAMPLES = config['source_nsamples']

    # === 2.3 VALIDATE CONFIGURATION ===
    assert isinstance(HIDDEN_DIMS, list) and len(HIDDEN_DIMS) > 0, "hidden_dims must be non-empty list"
    assert NUM_WAYPOINTS > 0, "num_waypoints must be > 0"
    assert NUM_LAYERS > 0, "num_layers must be > 0"
    assert TSM_N_EPOCHS > 0, "tsm_n_epochs must be > 0"
    assert TSM_BATCH_SIZE > 0, "tsm_batch_size must be > 0"
    if DEVICE == "cuda":
        assert torch.cuda.is_available(), "CUDA device requested but not available"

    print(f"Config loaded from {config_path}")
    print(f"  device: {DEVICE}")
    print(f"  seed: {SEED}")
    print(f"  hidden_dims: {HIDDEN_DIMS}")
    print(f"  num_waypoints: {NUM_WAYPOINTS}")
    print(f"  num_layers: {NUM_LAYERS}")
    print(f"  data_dim: {DATA_DIM}, nsamples: {NSAMPLES}")

    # === 3. FILE PATHS AND DATASET INITIALIZATION ===
    dataset_filename = f'{DATA_DIR}/dataset_filtered.h5'
    true_eigs_filename = f'{RAW_RESULTS_DIR}/true_eigs.h5'

    # 3.2 CREATE OUTPUT DIRECTORY
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    # === 4. DATASET LOADING ===
    with h5py.File(dataset_filename, 'r') as dataset_file:
        nrows = dataset_file['design_arr'].shape[0]
        print(f"\nOpening dataset: {dataset_filename}")
        print(f"  - Number of designs: {nrows}")
        print(f"  - theta_samples_arr shape: {dataset_file['theta_samples_arr'].shape}")
        print(f"  - y_samples_arr shape: {dataset_file['y_samples_arr'].shape}")

        # === 5. TRUE EIG COMPUTATION (ONE-TIME, BEFORE ALGORITHM LOOP) ===
        true_eigs_arr = np.zeros(nrows, dtype=np.float32)

        if os.path.exists(true_eigs_filename) and not args.force:
            print("Loading existing true EIGs...")
            with h5py.File(true_eigs_filename, 'r') as f:
                true_eigs_arr = f['true_eigs'][:]
        else:
            print("Computing true EIGs...")
            for idx in trange(nrows):
                Sigma_pi = torch.from_numpy(dataset_file['prior_covariance_arr'][idx]).to(DEVICE)
                design = torch.from_numpy(dataset_file['design_arr'][idx]).to(DEVICE)  # already [data_dim, 1]
                true_eigs_arr[idx] = compute_true_eig(Sigma_pi, design).item()

            # write true_eigs_arr atomically
            with h5py.File(true_eigs_filename, 'w') as f:
                f.create_dataset('true_eigs', data=true_eigs_arr)

        # === 7. MAIN LOOP STRUCTURE: PER HIDDEN DIMENSION ===
        for hidden_dim in HIDDEN_DIMS:
            print(f"\nProcessing hidden_dim={hidden_dim}...")

            # 7.1 RESET RANDOM SEEDS
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            # 7.1.5 COMPUTE EPOCH MULTIPLIER
            epochs_multiplier = int(math.log2(hidden_dim // 16)) + 1
            tsm_epochs = TSM_N_EPOCHS * epochs_multiplier
            mdre_epochs = 1000 * epochs_multiplier
            tdre_epochs = 1000 * epochs_multiplier
            print(f"  epochs_multiplier={epochs_multiplier} (tsm={tsm_epochs}, mdre={mdre_epochs}, tdre={tdre_epochs})")

            # 7.2 INSTANTIATE TSM
            tsm = TSM(
                input_dim=DATA_DIM + 1,
                hidden_dim=hidden_dim,
                n_epochs=tsm_epochs,
                batch_size=TSM_BATCH_SIZE,
                lr=TSM_LR,
                device=DEVICE,
            )
            tsm_plugin = EIGPlugin(density_ratio_estimator=tsm)
            temp_tsm_model = TimeScoreNetwork1D(DATA_DIM + 1, hidden_dim)
            tsm_param_count = sum(p.numel() for p in temp_tsm_model.parameters())
            del temp_tsm_model

            # 7.3 INSTANTIATE TRIANGULAR MDRE
            mdre_classifier = make_multiclass_classifier(
                name="default",
                input_dim=DATA_DIM + 1,
                num_classes=NUM_WAYPOINTS,
                latent_dim=hidden_dim,
                num_layers=NUM_LAYERS,
                num_epochs=mdre_epochs,
            )
            triangular_mdre = TriangularMDRE(mdre_classifier, device=DEVICE)
            mdre_adapter = TriangularDREAdapter(triangular_mdre)
            mdre_plugin = EIGPlugin(density_ratio_estimator=mdre_adapter)
            mdre_param_count = sum(p.numel() for p in mdre_classifier.parameters())

            # 7.4 INSTANTIATE TRIANGULAR TDRE
            tdre_classifiers = [
                DefaultBinaryClassifier(
                    input_dim=DATA_DIM + 1,
                    latent_dim=hidden_dim,
                    num_layers=NUM_LAYERS,
                    num_epochs=tdre_epochs,
                ).to(DEVICE)
                for _ in range(NUM_WAYPOINTS - 1)
            ]
            triangular_tdre = TriangularTDRE(
                classifiers=tdre_classifiers,
                num_waypoints=NUM_WAYPOINTS,
                device=DEVICE,
            )
            tdre_adapter = TriangularDREAdapter(triangular_tdre)
            tdre_plugin = EIGPlugin(density_ratio_estimator=tdre_adapter)
            tdre_param_count = sum(
                sum(p.numel() for p in c.parameters()) for c in tdre_classifiers
            )

            # 7.5 INSTANTIATE MULTIHEAD TRIANGULAR TDRE
            mh_classifier = MultiHeadBinaryClassifier(
                input_dim=DATA_DIM + 1,
                num_heads=NUM_WAYPOINTS - 1,
                hidden_dim=hidden_dim,
                head_dim=hidden_dim,
                num_shared_layers=NUM_LAYERS - 2,  # heads add 2 layers
                num_epochs=tdre_epochs,
            ).to(DEVICE)
            mh_tdre = MultiHeadTriangularTDRE(
                classifier=mh_classifier,
                num_waypoints=NUM_WAYPOINTS,
                device=DEVICE,
            )
            mh_tdre_adapter = TriangularDREAdapter(mh_tdre)
            mh_tdre_plugin = EIGPlugin(density_ratio_estimator=mh_tdre_adapter)
            mh_tdre_param_count = sum(p.numel() for p in mh_classifier.parameters())

            # 7.6 BUILD ALGORITHM TUPLES
            all_algorithms = [
                ("TriangularMDRE", mdre_plugin, mdre_param_count),
                ("TriangularTDRE", tdre_plugin, tdre_param_count),
                ("MultiHeadTriangularTDRE", mh_tdre_plugin, mh_tdre_param_count),
                ("TSM", tsm_plugin, tsm_param_count),
            ]
            # filter by --methods CLI arg
            algorithms = [(n, p, c) for n, p, c in all_algorithms if n in methods]

            # === 8. INNER LOOP: PER ALGORITHM ===
            for alg_name, alg_plugin, param_count in algorithms:
                # 8.1 CONSTRUCT FILENAME FOR THIS METHOD-DIM PAIR
                results_filename = f'{RAW_RESULTS_DIR}/{alg_name}_hidden_dim_{hidden_dim}.h5'

                # 8.2 CHECK FOR EXISTING RESULTS
                if os.path.exists(results_filename) and not args.force:
                    print(f"  Skipping {alg_name} hidden_dim={hidden_dim} (exists)")
                    continue

                print(f"  Starting {alg_name} hidden_dim={hidden_dim}.")

                # 8.3 INITIALIZE RESULT ARRAYS
                est_eigs_arr = np.zeros(nrows, dtype=np.float32)
                timing_arr = np.zeros(nrows, dtype=np.float32)
                peak_memory = 0.0

                # 8.4 INNERMOST LOOP: PER DATA POINT
                for idx in trange(nrows):
                    # 8.4.1 LOAD SAMPLE DATA
                    theta_samples = torch.from_numpy(dataset_file['theta_samples_arr'][idx]).to(DEVICE)
                    y_samples = torch.from_numpy(dataset_file['y_samples_arr'][idx]).to(DEVICE)

                    # 8.4.2 RESET GPU MEMORY STATS
                    if DEVICE == 'cuda':
                        torch.cuda.reset_peak_memory_stats()

                    # 8.4.3 MEASURE TIMING AND RUN ALGORITHM
                    t0 = time.perf_counter()
                    try:
                        result = alg_plugin.estimate_eig(theta_samples, y_samples)
                        est_eigs_arr[idx] = result.item() if hasattr(result, 'item') else result
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print(f"OOM at hidden_dim={hidden_dim}, {alg_name}, idx={idx}")
                            est_eigs_arr[idx] = np.nan
                            torch.cuda.empty_cache()
                        else:
                            raise
                    finally:
                        t1 = time.perf_counter()
                        timing_arr[idx] = t1 - t0

                    # 8.4.4 CAPTURE PEAK MEMORY
                    if DEVICE == 'cuda':
                        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())

                # 8.5 SAVE RESULTS TO HDF5 (ATOMIC)
                temp_file = results_filename + '.tmp'
                with h5py.File(temp_file, 'w') as f:
                    f.create_dataset('est_eigs_arr', data=est_eigs_arr)
                    f.create_dataset('timing_arr', data=timing_arr)
                    f.create_dataset('peak_memory', data=np.array(peak_memory, dtype=np.float32))
                    f.create_dataset('param_count', data=np.array(param_count, dtype=np.int64))
                os.replace(temp_file, results_filename)

    # === 9. POST-LOOP SUMMARY AND EXIT ===
    print("=" * 60)
    print("Experiment completed successfully")
    print(f"Results saved to: {RAW_RESULTS_DIR}/")
    print(f"  - True EIGs: {true_eigs_filename}")
    print(f"  - Per-method results: {RAW_RESULTS_DIR}/<method>_hidden_dim_<dim>.h5")
    print(f"Total hidden_dims tested: {len(HIDDEN_DIMS)}")
    print(f"Total algorithms: {len(algorithms)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
