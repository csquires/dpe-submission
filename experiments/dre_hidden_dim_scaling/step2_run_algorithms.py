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
    config_path = "experiments/dre_hidden_dim_scaling/config.yaml"
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
    DEVICE = config['device']
    SEED = config['seed']
    DATA_DIM = config['data_dim']
    NSAMPLES_TEST = config['nsamples_test']
    DATA_DIR = config['data_dir']
    RAW_RESULTS_DIR = config['raw_results_dir']

    # TSM hyperparameters with defaults if not in config
    TSM_N_EPOCHS = config.get('tsm_n_epochs', 1000)
    TSM_BATCH_SIZE = config.get('tsm_batch_size', 512)
    TSM_LR = config.get('tsm_lr', 1e-3)

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
    print(f"  data_dim: {DATA_DIM}, nsamples_test: {NSAMPLES_TEST}")

    # === 3. FILE PATHS AND DATASET INITIALIZATION ===
    dataset_filename = f'{DATA_DIR}/dataset.h5'

    # 3.2 CREATE OUTPUT DIRECTORY
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    # === 4. DATASET LOADING ===
    with h5py.File(dataset_filename, 'r') as dataset_file:
        nrows = dataset_file['samples_p0_arr'].shape[0]
        print(f"\nOpening dataset: {dataset_filename}")
        print(f"  - Number of instances: {nrows}")
        print(f"  - samples_p0_arr shape: {dataset_file['samples_p0_arr'].shape}")
        print(f"  - samples_p1_arr shape: {dataset_file['samples_p1_arr'].shape}")
        print(f"  - samples_pstar_arr shape: {dataset_file['samples_pstar_arr'].shape}")

        # === 5. MAIN LOOP STRUCTURE: PER HIDDEN DIMENSION ===
        for hidden_dim in HIDDEN_DIMS:
            print(f"\nProcessing hidden_dim={hidden_dim}...")

            # 5.1 RESET RANDOM SEEDS
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            # 5.1.5 COMPUTE EPOCH MULTIPLIER
            epochs_multiplier = int(math.log2(hidden_dim // 16)) + 1
            tsm_epochs = TSM_N_EPOCHS * epochs_multiplier
            mdre_epochs = 1000 * epochs_multiplier
            tdre_epochs = 1000 * epochs_multiplier
            print(f"  epochs_multiplier={epochs_multiplier} (tsm={tsm_epochs}, mdre={mdre_epochs}, tdre={tdre_epochs})")

            # 5.2 INSTANTIATE TRIANGULAR MDRE
            mdre_classifier = make_multiclass_classifier(
                name="default",
                input_dim=DATA_DIM,
                num_classes=NUM_WAYPOINTS,
                latent_dim=hidden_dim,
                num_layers=NUM_LAYERS,
                num_epochs=mdre_epochs,
            )
            triangular_mdre = TriangularMDRE(mdre_classifier, device=DEVICE)
            mdre_param_count = sum(p.numel() for p in mdre_classifier.parameters())

            # 5.3 INSTANTIATE TRIANGULAR TDRE
            tdre_classifiers = [
                DefaultBinaryClassifier(
                    input_dim=DATA_DIM,
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
            tdre_param_count = sum(
                sum(p.numel() for p in c.parameters()) for c in tdre_classifiers
            )

            # 5.4 INSTANTIATE MULTIHEAD TRIANGULAR TDRE
            mh_classifier = MultiHeadBinaryClassifier(
                input_dim=DATA_DIM,
                num_heads=NUM_WAYPOINTS - 1,
                hidden_dim=hidden_dim,
                head_dim=hidden_dim,
                num_shared_layers=NUM_LAYERS - 2,
                num_epochs=tdre_epochs,
            ).to(DEVICE)
            mh_tdre = MultiHeadTriangularTDRE(
                classifier=mh_classifier,
                num_waypoints=NUM_WAYPOINTS,
                device=DEVICE,
            )
            mh_tdre_param_count = sum(p.numel() for p in mh_classifier.parameters())

            # 5.5 INSTANTIATE TSM
            tsm = TSM(
                input_dim=DATA_DIM,
                hidden_dim=hidden_dim,
                n_epochs=tsm_epochs,
                batch_size=TSM_BATCH_SIZE,
                lr=TSM_LR,
                device=DEVICE,
            )
            temp_tsm_model = TimeScoreNetwork1D(DATA_DIM, hidden_dim)
            tsm_param_count = sum(p.numel() for p in temp_tsm_model.parameters())
            del temp_tsm_model

            # 5.6 BUILD ALGORITHM TUPLES
            all_algorithms = [
                ("TriangularMDRE", triangular_mdre, mdre_param_count),
                ("TriangularTDRE", triangular_tdre, tdre_param_count),
                ("MultiHeadTriangularTDRE", mh_tdre, mh_tdre_param_count),
                ("TSM", tsm, tsm_param_count),
            ]
            # filter by --methods CLI arg
            algorithms = [(n, alg, c) for n, alg, c in all_algorithms if n in methods]

            # === 6. INNER LOOP: PER ALGORITHM ===
            for alg_name, alg, param_count in algorithms:
                # 6.1 CONSTRUCT FILENAME FOR THIS METHOD-DIM PAIR
                results_filename = f'{RAW_RESULTS_DIR}/{alg_name}_hidden_dim_{hidden_dim}.h5'

                # 6.2 CHECK FOR EXISTING RESULTS
                if os.path.exists(results_filename) and not args.force:
                    print(f"  Skipping {alg_name} hidden_dim={hidden_dim} (exists)")
                    continue

                print(f"  Starting {alg_name} hidden_dim={hidden_dim}.")

                # 6.3 INITIALIZE RESULT ARRAYS
                est_ldrs_arr = np.zeros((nrows, NSAMPLES_TEST), dtype=np.float32)
                timing_arr = np.zeros(nrows, dtype=np.float32)
                peak_memory = 0.0

                # 6.4 INNERMOST LOOP: PER DATA POINT
                for idx in trange(nrows):
                    # 6.4.1 LOAD SAMPLE DATA
                    samples_p0 = torch.from_numpy(dataset_file['samples_p0_arr'][idx]).to(DEVICE)
                    samples_p1 = torch.from_numpy(dataset_file['samples_p1_arr'][idx]).to(DEVICE)
                    samples_pstar = torch.from_numpy(dataset_file['samples_pstar_arr'][idx]).to(DEVICE)

                    # 6.4.2 RESET GPU MEMORY STATS
                    if DEVICE == 'cuda':
                        torch.cuda.reset_peak_memory_stats()

                    # 6.4.3 MEASURE TIMING AND RUN ALGORITHM
                    t0 = time.perf_counter()
                    try:
                        # fit algorithm: triangular methods use 3-arg, TSM uses 2-arg
                        if alg_name in {"TriangularMDRE", "TriangularTDRE", "MultiHeadTriangularTDRE"}:
                            alg.fit(samples_p0, samples_p1, samples_pstar)
                        else:  # TSM
                            alg.fit(samples_p0, samples_p1)

                        # predict ldr on test samples
                        est_ldrs = alg.predict_ldr(samples_pstar)
                        est_ldrs_arr[idx] = est_ldrs.cpu().numpy()
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print(f"OOM at hidden_dim={hidden_dim}, {alg_name}, idx={idx}")
                            est_ldrs_arr[idx, :] = np.nan
                            if DEVICE == 'cuda':
                                torch.cuda.empty_cache()
                        else:
                            raise
                    finally:
                        t1 = time.perf_counter()
                        timing_arr[idx] = t1 - t0

                    # 6.4.4 CAPTURE PEAK MEMORY
                    if DEVICE == 'cuda':
                        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1e9)

                # 6.5 SAVE RESULTS TO HDF5 (ATOMIC)
                temp_file = results_filename + '.tmp'
                with h5py.File(temp_file, 'w') as f:
                    f.create_dataset('est_ldrs_arr', data=est_ldrs_arr)
                    f.create_dataset('timing_arr', data=timing_arr)
                    f.create_dataset('peak_memory', data=np.array(peak_memory, dtype=np.float32))
                    f.create_dataset('param_count', data=np.array(param_count, dtype=np.int64))
                os.replace(temp_file, results_filename)

    # === 7. POST-LOOP SUMMARY AND EXIT ===
    print("=" * 60)
    print("Experiment completed successfully")
    print(f"Results saved to: {RAW_RESULTS_DIR}/")
    print(f"  - Per-method results: {RAW_RESULTS_DIR}/<method>_hidden_dim_<dim>.h5")
    print(f"Total hidden_dims tested: {len(HIDDEN_DIMS)}")
    print(f"Total algorithms: {len(methods)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
