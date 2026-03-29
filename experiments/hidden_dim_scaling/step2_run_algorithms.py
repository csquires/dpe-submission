import argparse
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
from src.density_ratio_estimation.tsm import TSM
from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
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


class TriangularMDREEIGAdapter:
    """
    Adapter bridging TriangularMDRE to EIGPlugin interface.

    TriangularMDRE.fit() requires three arguments (samples_p0, samples_p1, samples_pstar),
    while EIGPlugin expects a DensityRatioEstimator with fit(samples_p0, samples_p1).
    This adapter uses samples_p0 (joint distribution) as pstar.
    """

    def __init__(self, triangular_mdre: TriangularMDRE):
        """
        args:
            triangular_mdre: instantiated TriangularMDRE object
        """
        self.triangular_mdre = triangular_mdre

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """
        Fit TriangularMDRE with p0 samples as pstar.

        args:
            samples_p0: joint distribution samples, shape [batch_size, dim]
            samples_p1: marginal distribution samples, shape [batch_size, dim]
        """
        self.triangular_mdre.fit(samples_p0, samples_p1, samples_p0)

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict log density ratio at query points.

        args:
            xs: query points, shape [batch_size, dim]

        returns:
            log density ratios, shape [batch_size]
        """
        return self.triangular_mdre.predict_ldr(xs)


def main():
    # === 1. COMMAND-LINE ARGUMENT PARSING ===
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of all algorithms, overwriting existing results"
    )
    args = parser.parse_args()

    # === 2. CONFIG LOADING AND PARAMETER EXTRACTION ===
    config_path = "experiments/hidden_dim_scaling/config.yaml"
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    HIDDEN_DIMS = config['hidden_dims']
    MDRE_NUM_WAYPOINTS = config['mdre_num_waypoints']
    MDRE_NUM_LAYERS = config['mdre_num_layers']
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
    assert MDRE_NUM_WAYPOINTS > 0, "mdre_num_waypoints must be > 0"
    assert MDRE_NUM_LAYERS > 0, "mdre_num_layers must be > 0"
    assert TSM_N_EPOCHS > 0, "tsm_n_epochs must be > 0"
    assert TSM_BATCH_SIZE > 0, "tsm_batch_size must be > 0"
    if DEVICE == "cuda":
        assert torch.cuda.is_available(), "CUDA device requested but not available"

    print(f"Config loaded from {config_path}")
    print(f"  device: {DEVICE}")
    print(f"  seed: {SEED}")
    print(f"  hidden_dims: {HIDDEN_DIMS}")
    print(f"  mdre_num_waypoints: {MDRE_NUM_WAYPOINTS}")
    print(f"  mdre_num_layers: {MDRE_NUM_LAYERS}")
    print(f"  data_dim: {DATA_DIM}, nsamples: {NSAMPLES}")

    # === 3. FILE PATHS AND DATASET INITIALIZATION ===
    dataset_filename = f'{DATA_DIR}/dataset_filtered.h5'
    results_filename = f'{RAW_RESULTS_DIR}/results.h5'

    # 3.2 CREATE OUTPUT DIRECTORY
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    # 3.3 LOAD AND CACHE EXISTING RESULTS
    existing_results = set()
    if os.path.exists(results_filename):
        with h5py.File(results_filename, 'r') as f:
            existing_results = set(f.keys())
        print(f"Existing results for: {sorted(existing_results)}")
    else:
        print(f"No existing results file at {results_filename} (will create)")

    # === 4. DATASET LOADING ===
    with h5py.File(dataset_filename, 'r') as dataset_file:
        nrows = dataset_file['design_arr'].shape[0]
        print(f"\nOpening dataset: {dataset_filename}")
        print(f"  - Number of designs: {nrows}")
        print(f"  - theta_samples_arr shape: {dataset_file['theta_samples_arr'].shape}")
        print(f"  - y_samples_arr shape: {dataset_file['y_samples_arr'].shape}")

        # === 5. TRUE EIG COMPUTATION (ONE-TIME, BEFORE ALGORITHM LOOP) ===
        true_eigs_arr = np.zeros(nrows, dtype=np.float32)

        if 'true_eigs_arr' not in existing_results or args.force:
            print("\nComputing true EIGs...")
            for idx in trange(nrows):
                Sigma_pi = torch.from_numpy(dataset_file['prior_covariance_arr'][idx]).to(DEVICE)
                design = torch.from_numpy(dataset_file['design_arr'][idx]).to(DEVICE)  # already [data_dim, 1]
                true_eigs_arr[idx] = compute_true_eig(Sigma_pi, design).item()

            # write true_eigs_arr atomically
            with h5py.File(results_filename, 'a') as results_file:
                if 'true_eigs_arr' in results_file:
                    del results_file['true_eigs_arr']
                results_file.create_dataset('true_eigs_arr', data=true_eigs_arr)
        else:
            print("True EIGs already computed, skipping")

        # === 7. MAIN LOOP STRUCTURE: PER HIDDEN DIMENSION ===
        for hidden_dim in HIDDEN_DIMS:
            print(f"\nProcessing hidden_dim={hidden_dim}...")

            # 7.1 RESET RANDOM SEEDS
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            # 7.2 COUNT TSM PARAMETERS
            temp_tsm_model = TimeScoreNetwork1D(DATA_DIM + 1, hidden_dim)
            tsm_param_count = sum(p.numel() for p in temp_tsm_model.parameters())
            del temp_tsm_model

            # 7.3 INSTANTIATE TRIANGULAR MDRE AND ADAPTER
            mdre_classifier = make_multiclass_classifier(
                name="default",
                input_dim=DATA_DIM + 1,
                num_classes=MDRE_NUM_WAYPOINTS,
                latent_dim=hidden_dim,
                num_layers=MDRE_NUM_LAYERS,
            )
            triangular_mdre = TriangularMDRE(mdre_classifier, device=DEVICE)
            mdre_adapter = TriangularMDREEIGAdapter(triangular_mdre)
            mdre_plugin = EIGPlugin(density_ratio_estimator=mdre_adapter)
            mdre_param_count = sum(p.numel() for p in mdre_classifier.parameters())

            # 7.4 INSTANTIATE TSM AND PLUGIN
            tsm = TSM(
                input_dim=DATA_DIM + 1,
                hidden_dim=hidden_dim,
                n_epochs=TSM_N_EPOCHS,
                batch_size=TSM_BATCH_SIZE,
                lr=TSM_LR,
                device=DEVICE,
            )
            tsm_plugin = EIGPlugin(density_ratio_estimator=tsm)

            # 7.5 BUILD ALGORITHM TUPLES
            algorithms = [
                ("TriangularMDRE", mdre_plugin, mdre_param_count),
                ("TSM", tsm_plugin, tsm_param_count),
            ]

            # === 8. INNER LOOP: PER ALGORITHM ===
            for alg_name, alg_plugin, param_count in algorithms:
                # 8.1 CHECK FOR EXISTING RESULTS
                est_eigs_key = f'est_eigs_arr_{alg_name}_hidden_dim_{hidden_dim}'

                if est_eigs_key in existing_results and not args.force:
                    print(f"  Skipping {alg_name} hidden_dim={hidden_dim} (results exist, use --force to overwrite)")
                    continue

                print(f"  Starting {alg_name} hidden_dim={hidden_dim}.")

                # 8.2 INITIALIZE RESULT ARRAYS
                est_eigs_arr = np.zeros(nrows, dtype=np.float32)
                timing_arr = np.zeros(nrows, dtype=np.float32)
                peak_memory = 0.0

                # 8.3 INNERMOST LOOP: PER DATA POINT
                for idx in trange(nrows):
                    # 8.3.1 LOAD SAMPLE DATA
                    theta_samples = torch.from_numpy(dataset_file['theta_samples_arr'][idx]).to(DEVICE)
                    y_samples = torch.from_numpy(dataset_file['y_samples_arr'][idx]).to(DEVICE)

                    # 8.3.2 RESET GPU MEMORY STATS
                    if DEVICE == 'cuda':
                        torch.cuda.reset_peak_memory_stats()

                    # 8.3.3 MEASURE TIMING AND RUN ALGORITHM
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

                    # 8.3.4 CAPTURE PEAK MEMORY
                    if DEVICE == 'cuda':
                        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())

                # 8.4 SAVE RESULTS TO HDF5 (ATOMIC)
                with h5py.File(results_filename, 'a') as results_file:
                    timing_key = f'timing_arr_{alg_name}_hidden_dim_{hidden_dim}'
                    memory_key = f'peak_memory_{alg_name}_hidden_dim_{hidden_dim}'
                    param_key = f'param_count_{alg_name}_hidden_dim_{hidden_dim}'

                    # delete if exists to ensure clean write
                    for key in [est_eigs_key, timing_key, memory_key, param_key]:
                        if key in results_file:
                            del results_file[key]

                    # create fresh datasets
                    results_file.create_dataset(est_eigs_key, data=est_eigs_arr)
                    results_file.create_dataset(timing_key, data=timing_arr)
                    results_file.create_dataset(memory_key, data=np.array(peak_memory, dtype=np.float32))
                    results_file.create_dataset(param_key, data=np.array(param_count, dtype=np.int64))

    # === 9. POST-LOOP SUMMARY AND EXIT ===
    print("=" * 60)
    print("Experiment completed successfully")
    print(f"Results saved to: {results_filename}")
    print(f"Total hidden_dims tested: {len(HIDDEN_DIMS)}")
    print(f"Total algorithms: {len(algorithms)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
