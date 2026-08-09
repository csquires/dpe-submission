"""
append test set to dbpedia cells: sample fresh p* from cond flow and compute true LDRs.

loads class-conditional flow; samples N_test balanced points from flow(.|y);
computes per-class log probs; for each existing cell, writes
samples_test_true_ldrs (per-cell, via logsumexp formula) to h5. the shared
test points are not stored (regenerable from the seed).
"""

import os
import re
import glob
import argparse
import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.models.flow import (
    ClassCondVelocityMLP,
    sample_class_cond_flow,
    log_prob_class_cond,
)


def expand_paths(config):
    """expand environment variables in config paths.

    mutates config in place; returns config with all string values
    containing '$' replaced via os.path.expandvars.
    """
    for key, value in config.items():
        if isinstance(value, str) and "$" in value:
            config[key] = os.path.expandvars(value)
    return config


def parse_alpha_pair(filename):
    """parse alpha_idx, pair_idx from filename like 'alpha_2_pair_7.h5'.

    args:
        filename: str, basename e.g. 'alpha_2_pair_7.h5'

    returns:
        (int, int): (alpha_idx, pair_idx)
    """
    m = re.match(r'alpha_(\d+)_pair_(\d+)\.h5', filename)
    if not m:
        raise ValueError(f"cannot parse {filename}")
    return int(m.group(1)), int(m.group(2))


def sample_test_points(cond_flow, n_test, num_classes, latent_dim, device, seed):
    """sample n_test balanced points from class-conditional flow.

    args:
        cond_flow: ClassCondVelocityMLP in eval mode on device
        n_test: int, total number of samples (balanced across classes)
        num_classes: int, e.g. 14 for dbpedia
        latent_dim: int, e.g. 64
        device: torch.device
        seed: int, random seed

    returns:
        torch.Tensor of shape (n_test, latent_dim), dtype float32, on CPU
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    per_class = n_test // num_classes
    remainder = n_test % num_classes

    # create balanced class labels (class-blocked order)
    y_test_list = []
    for k in range(num_classes):
        n_k = per_class + (1 if k < remainder else 0)
        y_test_list.append(torch.full((n_k,), k, dtype=torch.long))
    y_test = torch.cat(y_test_list)  # (n_test,)

    # sample latents per class
    samples_list = []
    with torch.no_grad():
        for k in range(num_classes):
            mask = (y_test == k)
            y_k = y_test[mask]  # class-k labels only
            n_k = mask.sum().item()
            if n_k == 0:
                continue

            z_k = sample_class_cond_flow(
                cond_flow,
                y_k.to(device),
                n_k,
                latent_dim,
                device=device,
                steps=100,
            )  # (n_k, latent_dim)
            samples_list.append(z_k.cpu())

    samples_test = torch.cat(samples_list, dim=0).float()  # (n_test, latent_dim)
    return samples_test


def compute_log_p_y_test(cond_flow, samples_test, num_classes, config, device):
    """compute log p(z|y) for all classes on test samples.

    args:
        cond_flow: ClassCondVelocityMLP in eval mode on device
        samples_test: (n_test, latent_dim) tensor on device
        num_classes: int, e.g. 14
        config: dict with 'log_prob_steps', 'log_prob_chunk_size'
        device: torch.device

    returns:
        (n_test, num_classes) tensor on CPU, dtype float32
    """
    log_p_y_list = []
    with torch.no_grad():
        for k in range(num_classes):
            y_k = torch.full(
                (samples_test.shape[0],), k, dtype=torch.long
            )
            log_p_k = log_prob_class_cond(
                cond_flow,
                samples_test,
                y_k.to(device),
                steps=config["log_prob_steps"],
                device=device,
                chunk_size=config["log_prob_chunk_size"],
            )  # (n_test,)
            log_p_y_list.append(log_p_k.cpu())

    log_p_y_test = torch.stack(log_p_y_list, dim=1).float()  # (n_test, num_classes)
    return log_p_y_test


def append_cell(cell_path, samples_test, log_p_y_test, num_classes):
    """append samples_test_true_ldrs to cell h5 file.

    reads w0/w1 from cell; computes per-point true LDR via logsumexp;
    writes samples_test_true_ldrs (overwriting if present) and deletes
    any stale samples_test.

    args:
        cell_path: str, path to h5 file
        samples_test: (n_test, latent_dim) tensor on CPU
        log_p_y_test: (n_test, num_classes) tensor on CPU
        num_classes: int, e.g. 14

    returns:
        (test_mean, train_mean): tuple of floats for sanity check
    """
    # read w0, w1
    with h5py.File(cell_path, 'r') as f:
        w0 = torch.from_numpy(f['w0'][:]).float()  # (num_classes,)
        w1 = torch.from_numpy(f['w1'][:]).float()  # (num_classes,)

    # log weights (clamp to avoid log(0))
    log_w0 = torch.log(torch.clamp(w0, min=1e-10))  # (num_classes,)
    log_w1 = torch.log(torch.clamp(w1, min=1e-10))  # (num_classes,)

    # compute true LDRs via logsumexp
    n_test = samples_test.shape[0]
    samples_test_true_ldrs = (
        torch.logsumexp(log_w0.unsqueeze(0) + log_p_y_test, dim=1)  # (n_test,)
        - torch.logsumexp(log_w1.unsqueeze(0) + log_p_y_test, dim=1)  # (n_test,)
    ).float()  # (n_test,)

    # write to h5 (overwrite if present). only the per-cell true ldrs are used
    # downstream (3-metric pipeline); the shared test points are not stored
    # (regenerable and deterministic from the seed). drop any stale samples_test.
    with h5py.File(cell_path, 'a') as f:
        if 'samples_test' in f:
            del f['samples_test']
        if 'samples_test_true_ldrs' in f:
            del f['samples_test_true_ldrs']
        f.create_dataset('samples_test_true_ldrs',
                         data=samples_test_true_ldrs.numpy(), dtype=np.float32)

    # sanity check: compare test vs train means
    with h5py.File(cell_path, 'r') as f:
        train_true_ldrs = torch.from_numpy(f['true_ldrs'][:]).float()

    test_mean = samples_test_true_ldrs.mean().item()
    train_mean = train_true_ldrs.mean().item()

    return test_mean, train_mean


def main():
    """orchestrate: load config, flow, sample test p*, compute log_p_y, append to cells."""
    parser = argparse.ArgumentParser(
        description="append test set (flow-sampled) to dbpedia cells"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ex/semisynth/dbpedia/config.yaml",
        help="path to config yaml",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=100000,
        help="number of test samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for test sampling (default: config seed + 10000)",
    )
    args = parser.parse_args()

    # load and expand config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = expand_paths(config)

    # set seed
    test_seed = args.seed if args.seed is not None else (config['seed'] + 10000)
    np.random.seed(test_seed)
    torch.manual_seed(test_seed)

    # device
    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        print('warning: cuda not available, falling back to cpu')
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f"using device: {device}")

    # load cond flow
    print("loading class-conditional flow...")
    cond_flow = ClassCondVelocityMLP(
        latent_dim=config['latent_dim'],
        num_classes=config['num_classes'],
        hidden_dim=config['cond_flow_hidden_dim'],
    )
    cond_flow.load_state_dict(torch.load(
        f"{config['ckpt_dir']}/cond_flow.pt",
        map_location='cpu',
        weights_only=False
    ))
    cond_flow.to(device).eval()

    # sample test p*
    print(f"sampling {args.n_test} test points from flow...")
    samples_test = sample_test_points(
        cond_flow,
        args.n_test,
        config['num_classes'],
        config['latent_dim'],
        device,
        test_seed,
    )  # (n_test, latent_dim) on CPU
    samples_test = samples_test.to(device)
    print(f"samples_test shape: {samples_test.shape}")

    # compute log_p_y_test for all classes
    print("computing log p(z|y) for all classes...")
    log_p_y_test = compute_log_p_y_test(
        cond_flow,
        samples_test,
        config['num_classes'],
        config,
        device,
    )  # (n_test, num_classes) on CPU
    print(f"log_p_y_test shape: {log_p_y_test.shape}")

    # discover and loop over cells
    data_dir = config['data_dir']
    cell_paths = sorted(glob.glob(f"{data_dir}/alpha_*_pair_*.h5"))

    print(f"\nappending to {len(cell_paths)} cells...")
    for cell_path in tqdm(cell_paths, desc="cells"):
        basename = os.path.basename(cell_path)
        alpha_idx, pair_idx = parse_alpha_pair(basename)

        test_mean, train_mean = append_cell(
            cell_path,
            samples_test,
            log_p_y_test,
            config['num_classes'],
        )

        print(f"alpha_{alpha_idx}_pair_{pair_idx}: "
              f"test_mean_ldr={test_mean:.4f}, train_mean_ldr={train_mean:.4f}")

    print(f"\nappended samples_test (n_test={args.n_test}, d={config['latent_dim']}) "
          f"+ samples_test_true_ldrs to all {len(cell_paths)} cells.")


if __name__ == '__main__':
    main()
