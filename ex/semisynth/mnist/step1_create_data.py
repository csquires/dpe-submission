"""
Generate balanced MNIST samples, class-conditional flow samples, and ground-truth LDRs.

loads global VAE and shared class-conditional flow once, then generates per-pair
data via analytic logsumexp mixture formulas. caches log_p_y globally with atomic
writes for SLURM safety. saves to HDF5 with schema matching old pipeline.
"""

import os
import argparse
import h5py
import yaml
import numpy as np
import torch
from tqdm import tqdm

from src.models.vae import MNISTVAE
from src.models.flow import (
    ClassCondVelocityMLP,
    sample_class_cond_flow,
    log_prob_class_cond
)
from ex.utils.mnist_imbalance import (
    sample_dirichlet_weights,
    invert_weights,
    weight_kl,
    subsample_mnist,
    get_mnist_dataset
)


def load_pstar_latent(config, vae_global, device):
    """encode balanced MNIST through global VAE to obtain p* latent points.

    args:
        config: experiment config dict with keys: num_samples, latent_dim
        vae_global: MNISTVAE in eval mode on device
        device: torch device

    returns:
        [N, D] tensor of float32 on CPU where N = config['num_samples'],
        D = config['latent_dim'].

    procedure:
        load balanced MNIST dataset. subsample via balanced_weights to ensure
        roughly equal class representation. encode first num_samples images in
        batches via vae_global.encode(), collect mu on CPU. stack and return.
    """
    dataset = get_mnist_dataset(root='./data', train=True, download=True)

    # balanced class weights
    balanced_weights = np.ones(10) / 10.0

    # subsample to ensure balance
    indices = subsample_mnist(
        dataset,
        balanced_weights,
        min_per_class=config['num_samples'] // 10
    )

    # collect images for first num_samples indices
    images_list = []
    for idx in indices[:config['num_samples']]:
        img, _ = dataset[idx]
        images_list.append(img)
    pstar_images = torch.stack(images_list, dim=0)  # [N, 1, 28, 28]

    # encode batched, collect on CPU
    pstar_latent_list = []
    batch_size = 1000
    with torch.no_grad():
        for i in range(0, pstar_images.shape[0], batch_size):
            batch = pstar_images[i:i + batch_size].to(device)  # [B, 1, 28, 28]
            mu, _ = vae_global.encode(batch)  # [B, D]
            pstar_latent_list.append(mu.cpu())

    pstar_latent = torch.cat(pstar_latent_list, dim=0)  # [N, D]
    return pstar_latent.float()


def load_or_compute_log_p_y(config, cond_flow, pstar_latent, device, force=False):
    """load or compute per-class log probabilities at pstar latent points.

    computes log p(z | y=k) for k in 0..9 at all points in pstar_latent.
    caches result at {data_dir}/log_p_y.pt with atomic write to handle SLURM races.

    args:
        config: experiment config dict with keys: data_dir, log_prob_steps
        cond_flow: ClassCondVelocityMLP in eval mode on device
        pstar_latent: [N, D] tensor on CPU
        device: torch device
        force: if True, ignore cache and recompute

    returns:
        [N, 10] tensor of float32 on CPU.

    procedure:
        determine cache path. if exists and not force, load and return.
        else compute per class: for each k in 0..9, evaluate
        log_prob_class_cond at all pstar points with y=k. stack results
        along dimension 1. atomic write: save to .tmp file, os.replace
        to final path. return tensor.
    """
    cache_path = f"{config['data_dir']}/log_p_y.pt"

    # load from cache if exists
    if os.path.exists(cache_path) and not force:
        return torch.load(cache_path, map_location='cpu')

    # compute per class
    N = pstar_latent.shape[0]
    log_p_y_list = []

    for k in range(10):
        y_k = torch.full((N,), k, dtype=torch.long, device=device)
        log_p = log_prob_class_cond(
            cond_flow,
            pstar_latent.to(device),
            y_k,
            steps=config['log_prob_steps'],
            device=device,
            chunk_size=500
        ).cpu()  # [N]
        log_p_y_list.append(log_p)

    log_p_y = torch.stack(log_p_y_list, dim=1)  # [N, 10]

    # atomic write (handle SLURM race)
    tmp_path = f"{cache_path}.tmp"
    torch.save(log_p_y, tmp_path)
    os.replace(tmp_path, cache_path)

    return log_p_y


def process_pair(config, alpha_idx, pair_idx, pstar_latent, log_p_y, cond_flow, device, force=False):
    """generate data for single (alpha_idx, pair_idx) pair and write to HDF5.

    args:
        config: experiment config dict
        alpha_idx: index into config['alphas']
        pair_idx: index into range(config['num_pairs_per_alpha'])
        pstar_latent: [N, D] precomputed latent points on CPU
        log_p_y: [N, 10] precomputed class log-probs on CPU
        cond_flow: ClassCondVelocityMLP in eval mode on device
        device: torch device
        force: if True, ignore existing files

    procedure:
        check if output exists; skip if yes and not force. set per-pair seed.
        extract alpha. sample Dirichlet weights, invert for w1. compute true LDRs
        via logsumexp mixture formula. sample class labels. sample flow latents.
        compute weight KL. write HDF5 with exact schema. print success message.
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
    w0 = sample_dirichlet_weights(alpha, n_draws=1, seed=pair_seed)[0]  # [10]
    w1 = invert_weights(w0)  # [10]

    # convert to tensors
    w0_t = torch.from_numpy(w0).float()  # [10]
    w1_t = torch.from_numpy(w1).float()  # [10]

    # log weights (clamp to avoid log(0))
    log_w0 = torch.log(torch.clamp(w0_t, min=1e-10))  # [10]
    log_w1 = torch.log(torch.clamp(w1_t, min=1e-10))  # [10]

    # compute true LDRs via logsumexp
    N = pstar_latent.shape[0]
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
        ).cpu()  # [N, D]

        p1_samples = sample_class_cond_flow(
            cond_flow, y1.to(device), N,
            config['latent_dim'],
            device=device, steps=100
        ).cpu()  # [N, D]

    # compute weight KL (analytic)
    kl_val = weight_kl(w0, w1)  # scalar float

    # write HDF5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('pstar_samples',
                         data=pstar_latent.numpy(), dtype=np.float32)
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
        ex/semisynth/mnist/config.yaml. create data directory.
        set device and seeds. load VAE and cond_flow on device in eval mode.
        compute or load pstar_latent. load or compute log_p_y. dispatch:
        if both alpha_idx and pair_idx specified, process single pair.
        else loop over all (alpha_idx, pair_idx) with tqdm.
    """
    parser = argparse.ArgumentParser(description="generate MNIST ELDR cond-flow data")
    parser.add_argument('--alpha-idx', type=int, default=None,
                        help='alpha index (for SLURM dispatch)')
    parser.add_argument('--pair-idx', type=int, default=None,
                        help='pair index (for SLURM dispatch)')
    parser.add_argument('--force', action='store_true',
                        help='force recomputation (ignore cache)')
    args = parser.parse_args()

    # load config
    config_path = 'ex/semisynth/mnist/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # create data directory
    os.makedirs(config['data_dir'], exist_ok=True)

    # set device and seeds
    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"using device: {device}")

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # load VAE global
    print("loading VAE global...")
    ckpt_dir = config['ckpt_dir']
    vae_global = MNISTVAE(latent_dim=config['latent_dim'])
    vae_global.load_state_dict(torch.load(
        f"{ckpt_dir}/vae_global.pt", map_location='cpu'
    ))
    vae_global.to(device).eval()

    # load cond_flow
    print("loading class-conditional flow...")
    cond_flow = ClassCondVelocityMLP(
        latent_dim=config['latent_dim'],
        num_classes=10,
        hidden_dim=config['cond_flow_hidden_dim'],
    )
    cond_flow.load_state_dict(torch.load(
        f"{ckpt_dir}/cond_flow.pt", map_location='cpu'
    ))
    cond_flow.to(device).eval()

    # compute or load pstar_latent
    print("loading pstar latent...")
    pstar_latent = load_pstar_latent(config, vae_global, device)

    # load or compute log_p_y
    print("loading or computing log_p_y...")
    log_p_y = load_or_compute_log_p_y(
        config, cond_flow, pstar_latent, device, force=args.force
    )

    # dispatch
    if args.alpha_idx is not None and args.pair_idx is not None:
        # single pair mode
        process_pair(
            config, args.alpha_idx, args.pair_idx,
            pstar_latent, log_p_y, cond_flow, device, force=args.force
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
                    pstar_latent, log_p_y, cond_flow, device, force=args.force
                )


if __name__ == '__main__':
    main()
