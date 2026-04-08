import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.models.vae import MNISTVAE, train_vae
from src.models.flow import VelocityMLP, train_flow
from experiments.utils.mnist_imbalance import (
    sample_dirichlet_weights,
    subsample_mnist,
    get_mnist_dataset,
)


def validate_args(args, config=None):
    """validate mutual exclusivity and bounds on arguments.

    raises ValueError for invalid argument combinations or out-of-bounds indices.
    """
    # check mutual exclusivity
    global_set = args.global_flag
    alpha_set = args.alpha_idx is not None
    pair_set = args.pair_idx is not None

    if global_set and (alpha_set or pair_set):
        raise ValueError("--global and --alpha-idx/--pair-idx are mutually exclusive")

    if not global_set and not alpha_set and not pair_set:
        raise ValueError("must specify either --global or both --alpha-idx and --pair-idx")

    if not global_set:
        if not (alpha_set and pair_set):
            raise ValueError("per-pair mode requires both --alpha-idx and --pair-idx")

        # validate bounds if config provided
        if config is not None:
            if args.alpha_idx < 0 or args.alpha_idx >= len(config["alphas"]):
                raise ValueError(
                    f"alpha_idx {args.alpha_idx} out of bounds [0, {len(config['alphas'])})"
                )
            if args.pair_idx < 0 or args.pair_idx >= config["num_pairs_per_alpha"]:
                raise ValueError(
                    f"pair_idx {args.pair_idx} out of bounds [0, {config['num_pairs_per_alpha']})"
                )


def get_device(device_str):
    """get valid device, falling back to cpu if cuda unavailable.

    returns string device name (e.g., 'cuda', 'cpu', 'cuda:0').
    """
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"warning: cuda not available, falling back to cpu")
        return "cpu"
    return device_str


def train_global_vae(config, device, force):
    """train vae on full mnist training set.

    loads full mnist, creates dataloader, trains vae, saves checkpoint.
    """
    # create checkpoint directory
    Path(config["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

    # set seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # load dataset
    dataset = get_mnist_dataset(root="./data", train=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    # create vae
    vae = MNISTVAE(latent_dim=config["latent_dim"], beta=config["vae_beta"])

    # delete checkpoint if force flag set
    ckpt_path = f"{config['ckpt_dir']}/vae_global.pt"
    if force:
        Path(ckpt_path).unlink(missing_ok=True)

    # train vae
    vae = train_vae(
        vae,
        loader,
        epochs=config["vae_epochs"],
        lr=config["vae_lr"],
        device=device,
        ckpt_path=ckpt_path,
    )

    print(f"Trained global VAE saved to {ckpt_path}")


def train_per_pair_vae_flow(config, alpha_idx, pair_idx, device, force):
    """train vae and flow pair on imbalanced mnist subsets.

    generates dirichlet-weighted class samples, trains separate vaes and flows
    for each side, saves checkpoints.
    """
    # create checkpoint directory
    Path(config["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

    # derive seed and config for this pair
    pair_seed = config["seed"] + alpha_idx * 1000 + pair_idx
    torch.manual_seed(pair_seed)
    np.random.seed(pair_seed)

    alpha = config["alphas"][alpha_idx]
    num_pairs = config["num_pairs_per_alpha"]

    # step 1: generate and save weights
    weights_arr = sample_dirichlet_weights(alpha, n_draws=2, seed=pair_seed)
    w0 = weights_arr[0]  # [10,]
    w1 = weights_arr[1]  # [10,]

    weights_ckpt = {
        "w0": torch.from_numpy(w0).float(),
        "w1": torch.from_numpy(w1).float(),
    }
    weights_path = f"{config['ckpt_dir']}/weights_alpha_{alpha_idx}_pair_{pair_idx}.pt"
    torch.save(weights_ckpt, weights_path)

    print(f"Generated weights for alpha_{alpha_idx}_pair_{pair_idx}")
    print(f"  side 0: min={w0.min():.4f}, max={w0.max():.4f}, mean={w0.mean():.4f}")
    print(f"  side 1: min={w1.min():.4f}, max={w1.max():.4f}, mean={w1.mean():.4f}")

    # step 2: create imbalanced mnist subsets
    dataset = get_mnist_dataset(root="./data", train=True)

    indices_0 = subsample_mnist(dataset, w0, min_per_class=10)
    indices_1 = subsample_mnist(dataset, w1, min_per_class=10)

    subset_0 = Subset(dataset, indices_0)
    subset_1 = Subset(dataset, indices_1)

    loader_0 = DataLoader(subset_0, batch_size=128, shuffle=True, num_workers=0)
    loader_1 = DataLoader(subset_1, batch_size=128, shuffle=True, num_workers=0)

    num_classes_0 = len(set(dataset.targets[i] for i in indices_0))
    num_classes_1 = len(set(dataset.targets[i] for i in indices_1))

    print(f"Trained VAE pair for alpha_{alpha_idx}_pair_{pair_idx}")
    print(f"  subset_0: {len(indices_0)} samples across {num_classes_0} classes")
    print(f"  subset_1: {len(indices_1)} samples across {num_classes_1} classes")

    # step 3: train vae_0 and vae_1
    vae_0 = MNISTVAE(latent_dim=config["latent_dim"], beta=config["vae_beta"])
    vae_1 = MNISTVAE(latent_dim=config["latent_dim"], beta=config["vae_beta"])

    vae_0_path = f"{config['ckpt_dir']}/vae_alpha_{alpha_idx}_pair_{pair_idx}_side0.pt"
    vae_1_path = f"{config['ckpt_dir']}/vae_alpha_{alpha_idx}_pair_{pair_idx}_side1.pt"

    if force:
        Path(vae_0_path).unlink(missing_ok=True)
        Path(vae_1_path).unlink(missing_ok=True)

    vae_0 = train_vae(
        vae_0,
        loader_0,
        epochs=config["vae_epochs"],
        lr=config["vae_lr"],
        device=device,
        ckpt_path=vae_0_path,
    )

    vae_1 = train_vae(
        vae_1,
        loader_1,
        epochs=config["vae_epochs"],
        lr=config["vae_lr"],
        device=device,
        ckpt_path=vae_1_path,
    )

    # step 4: encode training data to latent space
    codes_0 = []
    vae_0.eval()
    with torch.no_grad():
        for x, _ in loader_0:
            x = x.to(device)
            mu, _ = vae_0.encode(x)  # [B, latent_dim]
            codes_0.append(mu.cpu())
    codes_0 = torch.cat(codes_0, dim=0)  # [N0, latent_dim]

    codes_1 = []
    vae_1.eval()
    with torch.no_grad():
        for x, _ in loader_1:
            x = x.to(device)
            mu, _ = vae_1.encode(x)  # [B, latent_dim]
            codes_1.append(mu.cpu())
    codes_1 = torch.cat(codes_1, dim=0)  # [N1, latent_dim]

    print(f"Encoded latent codes")
    print(f"  codes_0: {codes_0.shape}")
    print(f"  codes_1: {codes_1.shape}")

    # step 5: train flow_0 and flow_1
    flow_0 = VelocityMLP(latent_dim=config["latent_dim"])
    flow_1 = VelocityMLP(latent_dim=config["latent_dim"])

    flow_0_path = f"{config['ckpt_dir']}/flow_alpha_{alpha_idx}_pair_{pair_idx}_side0.pt"
    flow_1_path = f"{config['ckpt_dir']}/flow_alpha_{alpha_idx}_pair_{pair_idx}_side1.pt"

    if force:
        Path(flow_0_path).unlink(missing_ok=True)
        Path(flow_1_path).unlink(missing_ok=True)

    flow_0 = train_flow(
        flow_0,
        codes_0,
        total_steps=config["flow_total_steps"],
        batch_size=128,
        lr=config["flow_lr"],
        device=device,
        ckpt_path=flow_0_path,
    )

    flow_1 = train_flow(
        flow_1,
        codes_1,
        total_steps=config["flow_total_steps"],
        batch_size=128,
        lr=config["flow_lr"],
        device=device,
        ckpt_path=flow_1_path,
    )

    print(f"Trained flow pair for alpha_{alpha_idx}_pair_{pair_idx}")
    print(f"  flow_0 saved to {flow_0_path}")
    print(f"  flow_1 saved to {flow_1_path}")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="pretrain vaes and flows for mnist eldr estimation"
    )
    parser.add_argument(
        "--global",
        action="store_true",
        dest="global_flag",
        help="train global vae on full mnist",
    )
    parser.add_argument(
        "--alpha-idx",
        type=int,
        default=None,
        help="alpha index for per-pair training",
    )
    parser.add_argument(
        "--pair-idx", type=int, default=None, help="pair index within alpha"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to train on"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="force retraining even if checkpoint exists",
    )
    args = parser.parse_args()

    # load config
    config = yaml.safe_load(
        open("experiments/mnist_eldr_estimation/config.yaml", "r")
    )

    # validate arguments
    validate_args(args, config)

    # get device
    device = get_device(args.device)

    # dispatch to global or per-pair mode
    if args.global_flag:
        train_global_vae(config, device, args.force)
    else:
        train_per_pair_vae_flow(config, args.alpha_idx, args.pair_idx, device, args.force)

    print("done")
