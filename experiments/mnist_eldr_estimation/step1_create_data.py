"""
Generate balanced MNIST samples, flow samples, and ground-truth log density ratios.

processes all (alpha_idx, pair_idx) combinations sequentially.
loads 5 pre-trained models per pair, generates samples, and computes LDRs.
saves to HDF5 files with proper batching for memory efficiency.
"""

import os
import argparse
import h5py
import yaml
import numpy as np
import torch
from tqdm import tqdm

from src.models.vae import MNISTVAE
from src.models.flow import VelocityMLP, sample_flow, log_prob
from experiments.utils.mnist_imbalance import get_mnist_dataset, subsample_mnist


def compute_log_jacobian(vae_pair, vae_global, z_global, device, chunk_size=500):
    """log |det J| of g^{-1} = vae_pair_enc o vae_global_dec.

    change-of-variables correction from per-pair flow latent space
    to global vae latent space. computes the 14x14 jacobian column
    by column via autograd, then takes slogdet.

    args:
        vae_pair: per-pair VAE whose encoder defines the target space
        vae_global: global VAE whose decoder maps back to image space
        z_global: [N, D] points in global latent space
        device: torch device
        chunk_size: samples per chunk for memory management

    returns:
        [N] log |det J_{g^{-1}}(z)| for each point
    """
    N, D = z_global.shape
    log_dets = []

    for start in range(0, N, chunk_size):
        z_chunk = z_global[start:start + chunk_size].clone().to(device)
        z_chunk.requires_grad_(True)
        B = z_chunk.shape[0]

        # forward: global_dec -> pair_enc
        img = vae_global.decode(z_chunk)
        mu, _ = vae_pair.encode(img)

        # jacobian column by column
        J = torch.zeros(B, D, D, device=device)
        for i in range(D):
            grad = torch.autograd.grad(
                mu[:, i].sum(), z_chunk,
                create_graph=False,
                retain_graph=(i < D - 1),
            )[0]  # [B, D]
            J[:, i, :] = grad

        log_det = torch.linalg.slogdet(J).logabsdet  # [B]
        log_dets.append(log_det.detach().cpu())

    return torch.cat(log_dets, dim=0)  # [N]


def load_pstar_images(config, device):
    """load balanced MNIST images for p* distribution.

    returns [N, 1, 28, 28] tensor of balanced MNIST samples.
    """
    mnist_dataset = get_mnist_dataset(root='./data', train=True, download=True)
    balanced_weights = np.ones(10) / 10.0
    balanced_indices = subsample_mnist(
        mnist_dataset, balanced_weights,
        min_per_class=config['num_samples'] // 10
    )
    pstar_images_list = []
    for idx in balanced_indices[:config['num_samples']]:
        img, _ = mnist_dataset[idx]
        pstar_images_list.append(img)
    return torch.stack(pstar_images_list, dim=0)  # [N, 1, 28, 28]


def process_pair(config, alpha_idx, pair_idx, pstar_images, device, device_str, force=False):
    """generate data for a single (alpha_idx, pair_idx).

    loads pretrained models, generates samples, computes ground-truth LDRs,
    saves to HDF5. skips if output exists and force is False.
    """
    data_dir = config['data_dir']
    ckpt_dir = config['ckpt_dir']
    output_path = f"{data_dir}/alpha_{alpha_idx}_pair_{pair_idx}.h5"

    if os.path.exists(output_path) and not force:
        print(f"skipping {output_path} (exists, use --force to override)")
        return

    # load pretrained models
    vae_global = MNISTVAE(latent_dim=config['latent_dim'])
    vae_global.load_state_dict(torch.load(
        f"{ckpt_dir}/vae_global.pt", map_location='cpu'))
    vae_global.to(device)
    vae_global.eval()

    vae_0 = MNISTVAE(latent_dim=config['latent_dim'])
    vae_0.load_state_dict(torch.load(
        f"{ckpt_dir}/vae_alpha_{alpha_idx}_pair_{pair_idx}_side0.pt", map_location='cpu'))
    vae_0.to(device)
    vae_0.eval()

    vae_1 = MNISTVAE(latent_dim=config['latent_dim'])
    vae_1.load_state_dict(torch.load(
        f"{ckpt_dir}/vae_alpha_{alpha_idx}_pair_{pair_idx}_side1.pt", map_location='cpu'))
    vae_1.to(device)
    vae_1.eval()

    flow_0 = VelocityMLP(latent_dim=config['latent_dim'])
    flow_0.load_state_dict(torch.load(
        f"{ckpt_dir}/flow_alpha_{alpha_idx}_pair_{pair_idx}_side0.pt", map_location='cpu'))
    flow_0.to(device)
    flow_0.eval()

    flow_1 = VelocityMLP(latent_dim=config['latent_dim'])
    flow_1.load_state_dict(torch.load(
        f"{ckpt_dir}/flow_alpha_{alpha_idx}_pair_{pair_idx}_side1.pt", map_location='cpu'))
    flow_1.to(device)
    flow_1.eval()

    weights = torch.load(
        f"{ckpt_dir}/weights_alpha_{alpha_idx}_pair_{pair_idx}.pt", map_location='cpu')

    batch_enc = 1000
    batch_lp = 500

    with torch.no_grad():
        # encode p* through vae_global
        pstar_latent_list = []
        for i in range(0, pstar_images.shape[0], batch_enc):
            batch = pstar_images[i:i+batch_enc].to(device)
            mu, _ = vae_global.encode(batch)
            pstar_latent_list.append(mu.cpu())
        pstar_latent = torch.cat(pstar_latent_list, dim=0)  # [N, 14]

        # generate p0 samples: flow_0 -> vae_0.decode -> vae_global.encode
        z0 = sample_flow(flow_0, config['num_samples'], config['latent_dim'], device_str)
        p0_latent_list = []
        for i in range(0, z0.shape[0], batch_enc):
            z_batch = z0[i:i+batch_enc]
            imgs = vae_0.decode(z_batch)
            mu, _ = vae_global.encode(imgs)
            p0_latent_list.append(mu.cpu())
        p0_latent = torch.cat(p0_latent_list, dim=0)  # [N, 14]

        # generate p1 samples: flow_1 -> vae_1.decode -> vae_global.encode
        z1 = sample_flow(flow_1, config['num_samples'], config['latent_dim'], device_str)
        p1_latent_list = []
        for i in range(0, z1.shape[0], batch_enc):
            z_batch = z1[i:i+batch_enc]
            imgs = vae_1.decode(z_batch)
            mu, _ = vae_global.encode(imgs)
            p1_latent_list.append(mu.cpu())
        p1_latent = torch.cat(p1_latent_list, dim=0)  # [N, 14]

        # compute ground-truth LDRs: encode p* through both per-pair VAEs
        z_in_0_list, z_in_1_list = [], []
        for i in range(0, pstar_images.shape[0], batch_enc):
            batch = pstar_images[i:i+batch_enc].to(device)
            mu_0, _ = vae_0.encode(batch)
            mu_1, _ = vae_1.encode(batch)
            z_in_0_list.append(mu_0.cpu())
            z_in_1_list.append(mu_1.cpu())
        z_in_0 = torch.cat(z_in_0_list, dim=0)  # [N, 14]
        z_in_1 = torch.cat(z_in_1_list, dim=0)  # [N, 14]

        # evaluate flow log-probs
        log_p0_list, log_p1_list = [], []
        for i in range(0, z_in_0.shape[0], batch_lp):
            z0_b = z_in_0[i:i+batch_lp].to(device)
            z1_b = z_in_1[i:i+batch_lp].to(device)
            log_p0_list.append(log_prob(flow_0, z0_b, steps=config['log_prob_steps'], device=device_str).cpu())
            log_p1_list.append(log_prob(flow_1, z1_b, steps=config['log_prob_steps'], device=device_str).cpu())
        log_p0 = torch.cat(log_p0_list, dim=0)
        log_p1 = torch.cat(log_p1_list, dim=0)

    # jacobian corrections for change of variables
    # log p0_global(z) = log p_flow0(g0^{-1}(z)) + log|det J_{g0^{-1}}(z)|
    log_jac_0 = compute_log_jacobian(vae_0, vae_global, pstar_latent, device)
    log_jac_1 = compute_log_jacobian(vae_1, vae_global, pstar_latent, device)
    true_ldrs = (log_p0 + log_jac_0) - (log_p1 + log_jac_1)  # [N,]

    # save to HDF5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('pstar_samples', data=pstar_latent.numpy(), dtype=np.float32)
        f.create_dataset('p0_samples', data=p0_latent.numpy(), dtype=np.float32)
        f.create_dataset('p1_samples', data=p1_latent.numpy(), dtype=np.float32)
        f.create_dataset('true_ldrs', data=true_ldrs.numpy(), dtype=np.float32)
        f.create_dataset('w0', data=weights['w0'].cpu().numpy(), dtype=np.float32)
        f.create_dataset('w1', data=weights['w1'].cpu().numpy(), dtype=np.float32)

    print(f"saved {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate MNIST ELDR estimation data")
    parser.add_argument('--alpha-idx', type=int, default=None, help='alpha index (for SLURM dispatch)')
    parser.add_argument('--pair-idx', type=int, default=None, help='pair index (for SLURM dispatch)')
    parser.add_argument('--force', action='store_true', help='force recomputation')
    args = parser.parse_args()

    config_path = 'experiments/mnist_eldr_estimation/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(config['data_dir'], exist_ok=True)

    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    print(f"using device: {device}")

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    print("loading balanced MNIST...")
    pstar_images = load_pstar_images(config, device)

    # dispatch: single pair or all pairs
    if args.alpha_idx is not None and args.pair_idx is not None:
        process_pair(config, args.alpha_idx, args.pair_idx, pstar_images, device, device_str, args.force)
    else:
        for alpha_idx in tqdm(range(len(config['alphas'])), desc="alpha"):
            for pair_idx in tqdm(range(config['num_pairs_per_alpha']), desc="pair", leave=False):
                process_pair(config, alpha_idx, pair_idx, pstar_images, device, device_str, args.force)
