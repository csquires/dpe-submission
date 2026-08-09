"""append fresh test p* from flow to all MNIST cells with per-cell true LDRs.

samples fresh test set from class-conditional flow, computes exact per-cell
true LDR via logsumexp formula, appends samples_test_true_ldrs (per-cell)
to every existing cell h5. the shared test points are not stored.
"""

import os
import argparse
import glob
import numpy as np
import torch
import h5py
import yaml

from src.models.flow import (
    ClassCondVelocityMLP,
    sample_class_cond_flow,
    log_prob_class_cond
)


def load_config(config_path):
    """load config YAML; return dict."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_cond_flow(device, latent_dim, hidden_dim):
    """instantiate and load cond_flow from the DPE_DATA_ROOT ckpt path.

    the ckpt path is not config['ckpt_dir'] (which templates the cluster-local
    CKPT_ROOT); it is $DPE_DATA_ROOT/mnist_eldr_cond_flow/ckpt/cond_flow.pt.

    args:
        device: torch device
        latent_dim: flow latent dimension (14 for MNIST)
        hidden_dim: flow hidden dimension (512)

    returns:
        ClassCondVelocityMLP on device, eval mode.
    """
    data_root = os.environ.get('DPE_DATA_ROOT')
    if not data_root:
        raise RuntimeError("DPE_DATA_ROOT env var not set")

    ckpt_path = os.path.join(
        data_root, 'mnist_eldr_cond_flow', 'ckpt', 'cond_flow.pt'
    )

    flow = ClassCondVelocityMLP(
        latent_dim=latent_dim,
        num_classes=10,
        hidden_dim=hidden_dim,
    )
    flow.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    flow.to(device).eval()

    return flow


def sample_test_from_flow(cond_flow, n_test, latent_dim, device):
    """sample balanced N_test latent points from class-conditional flow.

    y ~ uniform(10 classes, balanced); z ~ flow(.|y).
    returns (n_test, latent_dim) f32 on CPU.
    """
    per_class = n_test // 10
    # y_test: balanced mixture [0,0,...,0, 1,1,...,1, ..., 9,9,...,9]
    y_test = torch.arange(10, device=device).repeat_interleave(per_class)

    with torch.no_grad():
        z_test = sample_class_cond_flow(
            cond_flow, y_test, n_test,
            latent_dim,
            device=device, steps=100
        )  # (n_test, latent_dim) on device

    return z_test.cpu().float()


def compute_log_p_y_test(cond_flow, z_test, device, steps, chunk_size):
    """compute log p(z|y=k) for k=0..9 at all z_test.

    chunk z_test to manage memory. returns (len(z_test), 10) f32 on CPU.
    """
    n_test = z_test.shape[0]
    log_p_y_list = []

    z_test_device = z_test.to(device)  # move once to device

    for k in range(10):
        y_k = torch.full((n_test,), k, dtype=torch.long, device=device)
        # log_prob_class_cond handles chunking internally
        log_p = log_prob_class_cond(
            cond_flow,
            z_test_device,
            y_k,
            steps=steps,
            device=device,
            chunk_size=chunk_size
        ).cpu()  # (n_test,) on CPU
        log_p_y_list.append(log_p)

    log_p_y_test = torch.stack(log_p_y_list, dim=1).float()  # (n_test, 10)
    return log_p_y_test


def append_test_to_cells(data_dir, z_test, log_p_y_test):
    """discover cells via glob; append per-cell samples_test_true_ldrs.

    for each cell: read w0/w1, compute true_ldrs_test via logsumexp,
    delete-then-create samples_test_true_ldrs (deleting any stale samples_test),
    print sanity line per cell.

    args:
        data_dir: directory containing alpha_*_pair_*.h5 files
        z_test: (n_test, latent_dim) f32 on CPU - shared across cells
        log_p_y_test: (n_test, 10) f32 on CPU
    """
    if not os.path.isdir(data_dir):
        print(f"warning: data_dir does not exist: {data_dir}")
        return

    # glob cells
    cell_pattern = os.path.join(data_dir, 'alpha_*_pair_*.h5')
    cell_paths = sorted(glob.glob(cell_pattern))

    if not cell_paths:
        print(f"warning: no cells found in {data_dir}")
        return

    n_test = z_test.shape[0]
    z_test_np = z_test.numpy().astype(np.float32)

    for cell_path in cell_paths:
        # read w0, w1
        try:
            with h5py.File(cell_path, 'r') as f:
                if 'w0' not in f or 'w1' not in f:
                    print(f"warning: {cell_path} missing w0 or w1, skipping")
                    continue
                w0 = torch.from_numpy(f['w0'][()]).float()  # (10,)
                w1 = torch.from_numpy(f['w1'][()]).float()  # (10,)
                true_ldrs_train = torch.from_numpy(f['true_ldrs'][()]).float()
        except Exception as e:
            print(f"warning: failed to read {cell_path}: {e}")
            continue

        # compute true_ldrs_test via logsumexp
        log_w0 = torch.log(torch.clamp(w0, min=1e-10))  # (10,)
        log_w1 = torch.log(torch.clamp(w1, min=1e-10))  # (10,)

        log_p_y_test_t = log_p_y_test.clone()  # (n_test, 10)
        true_ldrs_test = (
            torch.logsumexp(log_w0.unsqueeze(0) + log_p_y_test_t, dim=1)
            - torch.logsumexp(log_w1.unsqueeze(0) + log_p_y_test_t, dim=1)
        )  # (n_test,)
        true_ldrs_test_np = true_ldrs_test.numpy().astype(np.float32)

        # append to h5 (delete-then-create)
        try:
            with h5py.File(cell_path, 'r+') as f:
                # delete if exists
                if 'samples_test' in f:
                    del f['samples_test']
                if 'samples_test_true_ldrs' in f:
                    del f['samples_test_true_ldrs']

                # only per-cell true ldrs are used downstream; drop the shared points.
                f.create_dataset('samples_test_true_ldrs',
                               data=true_ldrs_test_np, dtype=np.float32)
        except Exception as e:
            print(f"warning: failed to write {cell_path}: {e}")
            continue

        # sanity check: print per-cell mean
        test_mean = true_ldrs_test_np.mean()
        train_mean = true_ldrs_train.numpy().mean()
        cell_name = os.path.basename(cell_path)
        print(f"{cell_name}: test_mean={test_mean:.4f}, train_mean={train_mean:.4f}")


def main():
    """CLI entry: parse args, load config/flow, sample test p*, append to all cells."""
    parser = argparse.ArgumentParser(
        description="append fresh test p* from flow to MNIST cells"
    )
    parser.add_argument('--config', default='ex/semisynth/mnist/config.yaml',
                       help='path to config YAML')
    parser.add_argument('--n-test', type=int, default=100000,
                       help='fresh test sample count')
    parser.add_argument('--seed', type=int, default=None,
                       help='RNG seed (default: from config)')
    args = parser.parse_args()

    # load config
    config = load_config(args.config)

    # resolve seed
    seed = args.seed if args.seed is not None else config['seed']

    # seed RNG
    np.random.seed(seed)
    torch.manual_seed(seed)

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load flow
    cond_flow = load_cond_flow(
        device,
        config['latent_dim'],
        config['cond_flow_hidden_dim']
    )

    # sample test p*
    z_test = sample_test_from_flow(
        cond_flow, args.n_test, config['latent_dim'], device
    )

    # compute log_p_y_test
    log_p_y_test = compute_log_p_y_test(
        cond_flow, z_test, device,
        steps=config['log_prob_steps'],
        chunk_size=config.get('log_prob_chunk_size', 500)
    )

    # shape assertions
    assert z_test.shape == (args.n_test, 14), \
        f"z_test shape mismatch: {z_test.shape}"
    assert log_p_y_test.shape == (args.n_test, 10), \
        f"log_p_y_test shape mismatch: {log_p_y_test.shape}"

    # append to cells
    append_test_to_cells(config['data_dir'], z_test, log_p_y_test)


if __name__ == '__main__':
    main()
