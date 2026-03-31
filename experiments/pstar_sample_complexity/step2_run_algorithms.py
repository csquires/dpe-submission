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
from src.models.binary_classification.default_binary_classifier import DefaultBinaryClassifier
from src.models.binary_classification.multi_head_binary_classifier import MultiHeadBinaryClassifier
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D
from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
from src.density_ratio_estimation.triangular_tdre import TriangularTDRE
from src.density_ratio_estimation.mh_triangular_tdre import MultiHeadTriangularTDRE
from src.density_ratio_estimation.tsm import TSM


def make_algorithm(
    method_name: str,
    data_dim: int,
    device: str,
    num_waypoints: int,
    latent_dim: int,
    num_layers: int,
    num_epochs: int,
    tsm_n_epochs: int = 1000,
    tsm_batch_size: int = 512,
    tsm_lr: float = 0.001,
) -> tuple:
    """
    create fresh algorithm instance with specified configuration.

    args:
        method_name: one of ["TriangularMDRE", "TriangularTDRE", "MultiHeadTriangularTDRE", "TSM"]
        data_dim: input dimension
        device: torch device string ("cuda" or "cpu")
        num_waypoints: number of waypoints in triangular path
        latent_dim: hidden dimension size
        num_layers: number of network layers
        num_epochs: training epochs for triangular methods
        tsm_n_epochs: training epochs for TSM
        tsm_batch_size: batch size for TSM
        tsm_lr: learning rate for TSM

    returns:
        tuple of (algorithm_instance, param_count) where param_count is total trainable params
    """
    if method_name == "TSM":
        algorithm = TSM(
            input_dim=data_dim,
            hidden_dim=latent_dim,
            n_epochs=tsm_n_epochs,
            batch_size=tsm_batch_size,
            lr=tsm_lr,
            device=device,
        )
        # count params via temporary model
        temp_model = TimeScoreNetwork1D(data_dim, latent_dim)
        param_count = sum(p.numel() for p in temp_model.parameters())
        del temp_model

    elif method_name == "TriangularMDRE":
        classifier = make_multiclass_classifier(
            name="default",
            input_dim=data_dim,
            num_classes=num_waypoints,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_epochs=num_epochs,
        )
        algorithm = TriangularMDRE(classifier, device=device)
        param_count = sum(p.numel() for p in classifier.parameters())

    elif method_name == "TriangularTDRE":
        classifiers = [
            DefaultBinaryClassifier(
                input_dim=data_dim,
                latent_dim=latent_dim,
                num_layers=num_layers,
                num_epochs=num_epochs,
            ).to(device)
            for _ in range(num_waypoints - 1)
        ]
        algorithm = TriangularTDRE(
            classifiers=classifiers,
            num_waypoints=num_waypoints,
            device=device,
        )
        param_count = sum(
            sum(p.numel() for p in c.parameters()) for c in classifiers
        )

    elif method_name == "MultiHeadTriangularTDRE":
        classifier = MultiHeadBinaryClassifier(
            input_dim=data_dim,
            num_heads=num_waypoints - 1,
            hidden_dim=latent_dim,
            head_dim=latent_dim,
            num_shared_layers=num_layers - 2,  # heads add 2 layers
            num_epochs=num_epochs,
            epoch_scale=1,  # not num_waypoints-1: multi-head already processes all heads per epoch
            lr_hidden_dim_scale=True,
            lr_base_dim=16,
        ).to(device)
        algorithm = MultiHeadTriangularTDRE(
            classifier=classifier,
            num_waypoints=num_waypoints,
            device=device,
        )
        param_count = sum(p.numel() for p in classifier.parameters())

    else:
        raise ValueError(f"unknown method: {method_name}")

    return algorithm, param_count


def main():
    # === 1. COMMAND-LINE ARGUMENT PARSING ===
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["TriangularMDRE", "TriangularTDRE", "MultiHeadTriangularTDRE", "TSM"],
        help="single DRE method to run"
    )
    parser.add_argument(
        "--nsamples-pstar",
        type=int,
        required=True,
        help="number of samples from pstar (must be in config's nsamples_pstar_values)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing results for this method-nsamples pair"
    )
    args = parser.parse_args()

    # === 2. CONFIG LOADING AND VALIDATION ===
    # 2.1 load config
    config_path = "experiments/pstar_sample_complexity/config.yaml"
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # 2.2 extract config parameters
    NSAMPLES_PSTAR_VALUES = config['nsamples_pstar_values']
    NUM_WAYPOINTS = config['num_waypoints']
    NUM_LAYERS = config['num_layers']
    LATENT_DIM = config['latent_dim']
    NUM_EPOCHS = config['num_epochs']
    DEVICE = config['device']
    SEED = config['seed']
    DATA_DIR = config['data_dir']
    RAW_RESULTS_DIR = config['raw_results_dir']
    DATA_DIM = config['data_dim']
    NSAMPLES_TEST = config['nsamples_test']
    NUM_INSTANCES = config['num_instances']
    # tsm-specific
    TSM_N_EPOCHS = config['tsm_n_epochs']
    TSM_BATCH_SIZE = config['tsm_batch_size']
    TSM_LR = config['tsm_lr']

    # 2.3 validate CLI against config
    if args.nsamples_pstar not in NSAMPLES_PSTAR_VALUES:
        parser.error(
            f"--nsamples-pstar {args.nsamples_pstar} not in config "
            f"(available: {NSAMPLES_PSTAR_VALUES})"
        )

    # 2.4 validate configuration
    assert NUM_WAYPOINTS > 0, "num_waypoints must be > 0"
    assert NUM_LAYERS > 0, "num_layers must be > 0"
    assert LATENT_DIM > 0, "latent_dim must be > 0"
    assert NUM_EPOCHS > 0, "num_epochs must be > 0"
    assert NSAMPLES_TEST > 0, "nsamples_test must be > 0"
    assert NUM_INSTANCES > 0, "num_instances must be > 0"
    if DEVICE == "cuda":
        assert torch.cuda.is_available(), "CUDA device requested but not available"

    # 2.5 print configuration
    print(f"Config loaded from {config_path}")
    print(f"  device: {DEVICE}")
    print(f"  seed: {SEED}")
    print(f"  method: {args.method}")
    print(f"  nsamples_pstar: {args.nsamples_pstar}")
    print(f"  num_waypoints: {NUM_WAYPOINTS}")
    print(f"  latent_dim: {LATENT_DIM}")
    print(f"  num_layers: {NUM_LAYERS}")
    print(f"  num_epochs: {NUM_EPOCHS}")
    print(f"  data_dim: {DATA_DIM}, nsamples_test: {NSAMPLES_TEST}")
    print(f"  num_instances: {NUM_INSTANCES}")

    # === 3. MAIN LOOP SETUP ===
    # 3.1 reset random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 3.2 file paths and directory setup
    dataset_filename = f'{DATA_DIR}/dataset.h5'
    results_filename = (
        f'{RAW_RESULTS_DIR}/{args.method}_nsamples_pstar_{args.nsamples_pstar}.h5'
    )
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    # 3.3 check for existing results
    if os.path.exists(results_filename) and not args.force:
        print(f"Results exist at {results_filename}")
        print("Use --force to overwrite")
        return

    # 3.4 initialize result arrays
    est_ldrs_arr = np.zeros((NUM_INSTANCES, NSAMPLES_TEST), dtype=np.float32)
    timing_arr = np.zeros(NUM_INSTANCES, dtype=np.float32)
    peak_memory = 0.0

    # === 4. PROCESS INSTANCES ===
    with h5py.File(dataset_filename, 'r') as dataset_file:
        for instance_idx in trange(NUM_INSTANCES, desc="instances"):
            # 4.1 load sample data for this instance
            samples_p0 = torch.from_numpy(dataset_file['samples_p0_arr'][instance_idx]).to(DEVICE)
            samples_p1 = torch.from_numpy(dataset_file['samples_p1_arr'][instance_idx]).to(DEVICE)
            samples_pstar_full = torch.from_numpy(dataset_file['samples_pstar_arr'][instance_idx]).to(DEVICE)
            samples_test = torch.from_numpy(dataset_file['samples_test_arr'][instance_idx]).to(DEVICE)

            # 4.2 subsample pstar to requested size
            if samples_pstar_full.shape[0] > args.nsamples_pstar:
                indices = torch.randperm(samples_pstar_full.shape[0])[:args.nsamples_pstar]
                samples_pstar = samples_pstar_full[indices]
            else:
                samples_pstar = samples_pstar_full

            # 4.3 create fresh algorithm instance
            algorithm, param_count = make_algorithm(
                method_name=args.method,
                data_dim=DATA_DIM,
                device=DEVICE,
                num_waypoints=NUM_WAYPOINTS,
                latent_dim=LATENT_DIM,
                num_layers=NUM_LAYERS,
                num_epochs=NUM_EPOCHS,
                tsm_n_epochs=TSM_N_EPOCHS,
                tsm_batch_size=TSM_BATCH_SIZE,
                tsm_lr=TSM_LR,
            )

            # 4.4 reset memory stats
            if DEVICE == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            # 4.5 measure timing and run algorithm
            t0 = time.perf_counter()
            try:
                # fit DRE: TSM uses (p0, p1), triangular methods use (p0, p1, pstar)
                if args.method == "TSM":
                    algorithm.fit(samples_p0, samples_p1)
                else:
                    algorithm.fit(samples_p0, samples_p1, samples_pstar)

                # predict on test set
                est_ldrs = algorithm.predict_ldr(samples_test)
                est_ldrs_arr[instance_idx] = est_ldrs.cpu().numpy()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"OOM at instance={instance_idx}, nsamples_pstar={args.nsamples_pstar}")
                    est_ldrs_arr[instance_idx] = np.nan
                    if DEVICE == 'cuda':
                        torch.cuda.empty_cache()
                else:
                    raise
            finally:
                t1 = time.perf_counter()
                timing_arr[instance_idx] = t1 - t0

            # 4.6 capture peak memory
            if DEVICE == 'cuda':
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())

    # === 5. ATOMIC WRITE AND COMPLETION ===
    # 5.1 save results atomically
    temp_file = results_filename + '.tmp'
    with h5py.File(temp_file, 'w') as f:
        f.create_dataset('est_ldrs_arr', data=est_ldrs_arr)
        f.create_dataset('timing_arr', data=timing_arr)
        f.create_dataset('peak_memory', data=np.array(peak_memory, dtype=np.float32))
        f.create_dataset('param_count', data=np.array(param_count, dtype=np.int64))
    os.replace(temp_file, results_filename)

    # 5.2 print summary
    print("=" * 60)
    print("Algorithm run completed successfully")
    print(f"  method: {args.method}")
    print(f"  nsamples_pstar: {args.nsamples_pstar}")
    print(f"  instances processed: {NUM_INSTANCES}")
    print(f"  results saved to: {results_filename}")
    print(f"  shape est_ldrs_arr: {est_ldrs_arr.shape}")
    print(f"  shape timing_arr: {timing_arr.shape}")
    print(f"  peak memory (GB): {peak_memory / 1e9:.3f}")
    print(f"  param_count: {param_count}")
    print("=" * 60)


if __name__ == '__main__':
    main()
