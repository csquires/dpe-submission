"""
Generate balanced DBpedia samples, class-conditional flow samples, and ground-truth LDRs.

loads precomputed pstar codes and class log-probs, then generates per-pair data
via analytic logsumexp mixture formulas. saves to HDF5 with schema matching MNIST
cond-flow for estimator zoo compatibility.
"""

import os
import argparse
import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.models.flow import ClassCondVelocityMLP, sample_class_cond_flow
from experiments.utils.mnist_imbalance import (
    sample_dirichlet_weights,
    invert_weights,
    weight_kl,
)
from experiments.utils.dbpedia_imbalance import flow_state_hash


def expand_paths(config):
    """expand environment variables in config paths.

    returns modified config dict with all keys matching {ckpt_dir, data_dir, ...}
    replaced via os.path.expandvars.
    """
    result = {}
    for key, value in config.items():
        if isinstance(value, str):
            result[key] = os.path.expandvars(value)
        else:
            result[key] = value
    return result


def load_pstar_codes(config):
    """load precomputed balanced pstar codes from step0 mode log_p_y.

    args:
        config: experiment config dict

    returns:
        [N, 64] float32 tensor on CPU.

    raises:
        FileNotFoundError if pstar_codes.pt does not exist.
    """
    path = f"{config['data_dir']}/pstar_codes.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run step0 --mode log_p_y first to compute {path}")
    return torch.load(path, map_location='cpu').float()


def load_log_p_y(config, flowhash):
    """load precomputed per-class log probs at pstar codes from step0.

    args:
        config: experiment config dict
        flowhash: str, 8-char hash of cond_flow.pt state dict

    returns:
        [N, 14] float32 tensor on CPU.

    raises:
        FileNotFoundError if log_p_y.<flowhash>.pt does not exist.
    """
    path = f"{config['data_dir']}/log_p_y.{flowhash}.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run step0 --mode log_p_y first to compute {path}")
    return torch.load(path, map_location='cpu').float()


def process_pair(
    config,
    alpha_idx,
    pair_idx,
    pstar_codes,
    log_p_y,
    cond_flow,
    device,
    force=False,
):
    """generate data for single (alpha_idx, pair_idx) pair and write HDF5.

    args:
        config: experiment config dict with keys: alphas, seed, data_dir, latent_dim
        alpha_idx: int, index into config['alphas']
        pair_idx: int, index into range(config['num_pairs_per_alpha'])
        pstar_codes: [N, 64] float32 tensor on CPU
        log_p_y: [N, 14] float32 tensor on CPU
        cond_flow: ClassCondVelocityMLP in eval mode on device
        device: torch device
        force: bool, if True skip existence check

    returns:
        None (writes HDF5 side-effect).

    procedure:
        compute output path. skip if exists and not force.
        set per-pair seed. extract alpha.
        sample w0 via sample_dirichlet_weights(alpha, n_draws=1, seed, K=14)[0].
        compute w1 = invert_weights(w0).
        convert to tensors and compute log-weights (clamped).
        compute true LDRs via logsumexp mixture formula.
        sample class labels from categorical(w0), categorical(w1).
        sample flow latents for both classes.
        compute weight KL.
        write HDF5 with schema: pstar_samples, p0_samples, p1_samples,
        true_ldrs, w0, w1, kl_weights.
    """
    data_dir = config['data_dir']
    output_path = f"{data_dir}/alpha_{alpha_idx}_pair_{pair_idx}.h5"

    # skip if exists
    if os.path.exists(output_path) and not force:
        print(f"skipping {output_path} (exists, use --force to override)")
        return

    # per-pair seed
    pair_seed = config['seed'] + alpha_idx * 1000 + pair_idx
    torch.manual_seed(pair_seed)
    np.random.seed(pair_seed)

    # extract alpha
    alpha = config['alphas'][alpha_idx]

    # sample Dirichlet weights
    w0 = sample_dirichlet_weights(alpha, n_draws=1, seed=pair_seed, K=14)[0]  # [14]
    w1 = invert_weights(w0)  # [14]

    # convert to tensors
    w0_t = torch.from_numpy(w0).float()  # [14]
    w1_t = torch.from_numpy(w1).float()  # [14]

    # log weights (clamp to avoid log(0))
    log_w0 = torch.log(torch.clamp(w0_t, min=1e-10))  # [14]
    log_w1 = torch.log(torch.clamp(w1_t, min=1e-10))  # [14]

    # compute true LDRs via logsumexp
    N = pstar_codes.shape[0]
    true_ldrs = (
        torch.logsumexp(log_w0.unsqueeze(0) + log_p_y, dim=1)  # [N]
        - torch.logsumexp(log_w1.unsqueeze(0) + log_p_y, dim=1)  # [N]
    )  # [N]

    # sample class labels
    y0 = torch.distributions.Categorical(probs=w0_t).sample((N,)).long()  # [N]
    y1 = torch.distributions.Categorical(probs=w1_t).sample((N,)).long()  # [N]

    # sample latent points via class-conditional flow
    with torch.no_grad():
        p0_samples = sample_class_cond_flow(
            cond_flow, y0.to(device), N,
            config['latent_dim'],
            device=device, steps=100
        ).cpu()  # [N, 64]

        p1_samples = sample_class_cond_flow(
            cond_flow, y1.to(device), N,
            config['latent_dim'],
            device=device, steps=100
        ).cpu()  # [N, 64]

    # compute weight KL (analytic)
    kl_val = weight_kl(w0, w1)  # scalar float

    # write HDF5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('pstar_samples',
                         data=pstar_codes.numpy(), dtype=np.float32)
        f.create_dataset('p0_samples',
                         data=p0_samples.numpy(), dtype=np.float32)
        f.create_dataset('p1_samples',
                         data=p1_samples.numpy(), dtype=np.float32)
        f.create_dataset('true_ldrs',
                         data=true_ldrs.numpy(), dtype=np.float32)
        f.create_dataset('w0',
                         data=w0, dtype=np.float32)
        f.create_dataset('w1',
                         data=w1, dtype=np.float32)
        f.create_dataset('kl_weights',
                         data=float(kl_val))

    print(f"saved {output_path}")


def main():
    """top-level entry point: orchestrate loading, caching, and dispatch.

    procedure:
        parse CLI args (--alpha-idx, --pair-idx, --force). load config from
        experiments/dbpedia_eldr_cond_flow/config.yaml. expand paths.
        create data directory. set device and seeds.
        load cond_flow on device in eval mode.
        compute flowhash and load pstar_codes + log_p_y.
        dispatch: if both alpha_idx and pair_idx specified, process single pair.
        else loop over all (alpha_idx, pair_idx) with tqdm.
    """
    parser = argparse.ArgumentParser(description="generate DBpedia ELDR cond-flow data")
    parser.add_argument('--alpha-idx', type=int, default=None,
                        help='alpha index (for SLURM dispatch)')
    parser.add_argument('--pair-idx', type=int, default=None,
                        help='pair index (for SLURM dispatch)')
    parser.add_argument('--force', action='store_true',
                        help='force recomputation (ignore cache)')
    args = parser.parse_args()

    # load config
    config_path = 'experiments/dbpedia_eldr_cond_flow/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # expand paths
    config = expand_paths(config)

    # create data directory
    os.makedirs(config['data_dir'], exist_ok=True)

    # set device and seeds
    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"using device: {device}")

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # load cond_flow
    print("loading class-conditional flow...")
    ckpt_dir = config['ckpt_dir']
    cond_flow = ClassCondVelocityMLP(
        latent_dim=config['latent_dim'],
        num_classes=14,
        hidden_dim=config['cond_flow_hidden_dim'],
    )
    cond_flow.load_state_dict(torch.load(
        f"{ckpt_dir}/cond_flow.pt", map_location='cpu'
    ))
    cond_flow.to(device).eval()

    # compute flowhash and load precomputed data
    print("computing flowhash and loading precomputed data...")
    flowhash = flow_state_hash(f"{ckpt_dir}/cond_flow.pt")
    pstar_codes = load_pstar_codes(config)  # [N, 64]
    log_p_y = load_log_p_y(config, flowhash)  # [N, 14]

    # dispatch
    if args.alpha_idx is not None and args.pair_idx is not None:
        # single pair mode
        process_pair(
            config, args.alpha_idx, args.pair_idx,
            pstar_codes, log_p_y, cond_flow, device, force=args.force
        )
    else:
        # grid mode
        for alpha_idx in tqdm(range(len(config['alphas'])), desc="alpha"):
            for pair_idx in tqdm(
                range(config['num_pairs_per_alpha']),
                desc="pair", leave=False
            ):
                process_pair(
                    config, alpha_idx, pair_idx,
                    pstar_codes, log_p_y, cond_flow, device, force=args.force
                )


if __name__ == '__main__':
    main()
