import os
import numpy as np
import torch


def encode_corpus(
    texts: list[str],
    model_name: str = 'sentence-transformers/all-mpnet-base-v2',
    batch_size: int = 256,
    device: str = 'cuda',
) -> torch.Tensor:
	"""Encode list of texts via pretrained SBERT.

	Procedure: Lazy-import SentenceTransformer, instantiate on device, encode with
	mean-pooling via attention mask (automatic for this model), L2-normalize,
	return embeddings as float32 tensor on CPU.

	Args:
	    texts: list of strings to encode
	    model_name: SentenceTransformer model identifier
	    batch_size: encoding batch size (256 fits ~11GB memory for all-mpnet-base-v2)
	    device: 'cuda' or 'cpu'

	Returns:
	    float32 torch.Tensor of shape [N, 768] on CPU, rows L2-normalized to unit norm
	"""
	from sentence_transformers import SentenceTransformer

	model = SentenceTransformer(model_name, device=device)
	emb_np = model.encode(
		texts,
		batch_size=batch_size,
		normalize_embeddings=True,
		convert_to_numpy=True,
		show_progress_bar=True,
	)
	return torch.from_numpy(emb_np).float()


def load_or_encode(
	texts: list[str],
	labels: list[int] | np.ndarray | torch.Tensor,
	cache_path: str,
	force: bool = False,
	**encode_kwargs,
) -> dict[str, torch.Tensor]:
	"""Load embeddings from cache if exists, else encode and save.

	Procedure: Check if cache_path exists and not force; if so, load via torch.load
	and return. Otherwise call encode_corpus(**encode_kwargs), coerce labels to
	int64 tensor, save dict via atomic write (tmp -> os.replace), return dict.

	Args:
	    texts: list of strings to encode
	    labels: class labels; list, ndarray, or tensor; coerced to int64 tensor
	    cache_path: path to save/load embeddings dict
	    force: if True, re-encode regardless of cache existence
	    **encode_kwargs: passed to encode_corpus (model_name, batch_size, device)

	Returns:
	    dict with keys:
	        'embeddings': float32 torch.Tensor [N, 768] on CPU, L2-normalized
	        'labels': int64 torch.Tensor [N] on CPU
	"""
	if os.path.exists(cache_path) and not force:
		data = torch.load(cache_path, map_location='cpu', weights_only=False)
		return data

	embeddings = encode_corpus(texts, **encode_kwargs)

	# coerce labels to int64 tensor
	if isinstance(labels, list):
		labels = torch.tensor(labels, dtype=torch.int64)
	elif isinstance(labels, np.ndarray):
		labels = torch.from_numpy(labels).to(torch.int64)
	else:
		labels = labels.to(torch.int64)

	data = {'embeddings': embeddings, 'labels': labels}

	# atomic write
	tmp_path = f"{cache_path}.tmp"
	torch.save(data, tmp_path)
	os.replace(tmp_path, cache_path)

	return data
