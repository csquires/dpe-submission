import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from experiments.utils.pretrain import get_device, encode_mnist_with_vae
from src.models.vae import MNISTVAE
from src.models.flow import ClassCondVelocityMLP, train_class_cond_flow
from experiments.utils.mnist_imbalance import get_mnist_dataset


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
	config = yaml.safe_load(open("experiments/mnist/config.yaml"))

	# get device
	device = get_device(args.device)

	# dispatch
	if args.mode == "global":
		from experiments.mnist_uncond.step0_pretrain import train_global_vae

		train_global_vae(config, device, args.force)
	else:
		train_cond_flow_pipeline(config, device, args.force)

	print("done")
