import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.models.vae import MNISTVAE
from src.models.flow import ClassCondVelocityMLP, train_class_cond_flow
from experiments.utils.mnist_imbalance import get_mnist_dataset


def get_device(device_str: str) -> str:
	"""get valid device, falling back to cpu if cuda unavailable.

	returns string device name (e.g., 'cuda', 'cpu', 'cuda:0').
	"""
	if device_str.startswith("cuda") and not torch.cuda.is_available():
		print("warning: cuda not available, falling back to cpu")
		return "cpu"
	return device_str


def encode_mnist_with_vae(
	vae: MNISTVAE,
	dataset,
	device: str,
	batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""encode full mnist dataset through trained vae.

	returns:
		codes: [N, latent_dim] float tensor on cpu.
		labels: [N] long tensor on cpu, class indices 0-9.

	procedure:
		create dataloader with shuffle=False.
		set vae to eval mode.
		loop over batches, encode via vae.encode(), collect mu and labels.
		concatenate and return on cpu.
	"""
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
	vae.eval()

	codes_list = []
	labels_list = []

	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			mu, _ = vae.encode(x)  # [B, latent_dim]; discard logvar
			codes_list.append(mu.cpu())
			labels_list.append(y.cpu())

	codes = torch.cat(codes_list, dim=0)  # [N, latent_dim]
	labels = torch.cat(labels_list, dim=0)  # [N]

	return codes, labels


def train_cond_flow_pipeline(config: dict, device: str, force: bool) -> None:
	"""orchestrate training of conditional flow matching model on mnist codes.

	procedure:
		create checkpoint directory.
		apply seed.
		load global vae checkpoint (fail fast if missing).
		load mnist dataset and encode through vae.
		instantiate ClassCondVelocityMLP.
		delete existing checkpoint if force=True.
		train flow model.
		print confirmation.
	"""
	# create checkpoint directory
	Path(config["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

	# apply seed
	torch.manual_seed(config["seed"])
	np.random.seed(config["seed"])

	# load global vae
	vae_path = f"{config['ckpt_dir']}/vae_global.pt"
	if not Path(vae_path).exists():
		raise FileNotFoundError(
			f"Global VAE checkpoint not found at {vae_path}. Run --mode global first."
		)

	vae = MNISTVAE(latent_dim=config["latent_dim"])
	vae.load_state_dict(torch.load(vae_path, map_location="cpu"))
	vae = vae.to(device)

	# load mnist and encode
	dataset = get_mnist_dataset(root="./data", train=True, download=True)
	codes, labels = encode_mnist_with_vae(vae, dataset, device)

	# instantiate flow model
	model = ClassCondVelocityMLP(
		latent_dim=config["latent_dim"],
		num_classes=10,
		hidden_dim=config["cond_flow_hidden_dim"],
		embed_dim=config["cond_flow_embed_dim"],
	)

	# checkpoint path
	ckpt_path = f"{config['ckpt_dir']}/cond_flow.pt"

	# force deletion
	if force:
		Path(ckpt_path).unlink(missing_ok=True)

	# train
	train_class_cond_flow(
		model=model,
		codes=codes,
		labels=labels,
		total_steps=config["cond_flow_total_steps"],
		batch_size=config["cond_flow_batch_size"],
		lr=config["cond_flow_lr"],
		device=device,
		ckpt_path=ckpt_path,
		ema_decay=config["cond_flow_ema_decay"],
	)

	print(f"Trained cond flow saved to {ckpt_path}")


if __name__ == "__main__":
	# parse arguments
	parser = argparse.ArgumentParser(
		description="pretrain vae and conditional flow for mnist eldr with flow"
	)
	parser.add_argument(
		"--mode",
		type=str,
		required=True,
		choices=["global", "cond_flow"],
		help="training mode: global vae or conditional flow",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda",
		help="device to train on",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="force retraining by deleting existing checkpoints",
	)
	args = parser.parse_args()

	# load config
	config = yaml.safe_load(open("experiments/mnist_eldr_cond_flow/config.yaml"))

	# get device
	device = get_device(args.device)

	# dispatch
	if args.mode == "global":
		from experiments.mnist_eldr_estimation.step0_pretrain import train_global_vae

		train_global_vae(config, device, args.force)
	else:
		train_cond_flow_pipeline(config, device, args.force)

	print("done")
