import torch
from torch.utils.data import DataLoader


def get_device(device_str: str) -> str:
	"""get valid device, falling back to cpu if cuda unavailable.

	returns string device name (e.g., 'cuda', 'cpu', 'cuda:0').
	"""
	if device_str.startswith("cuda") and not torch.cuda.is_available():
		print("warning: cuda not available, falling back to cpu")
		return "cpu"
	return device_str


def encode_mnist_with_vae(
	vae,
	dataset,
	device: str,
	batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""encode full dataset through trained vae.

	returns:
		codes: [N, latent_dim] float tensor on cpu.
		labels: [N] long tensor on cpu, class indices.

	procedure:
		create dataloader with shuffle=False.
		set vae to eval mode.
		loop over batches, encode via vae.encode(), collect mu and labels.
		concatenate and return on cpu.

	despite the legacy name, this function is fully dataset-agnostic.
	it makes no dataset-specific assumptions beyond (image, label) tuples.
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


# alias for clarity: encode_with_vae is the preferred name going forward
encode_with_vae = encode_mnist_with_vae
