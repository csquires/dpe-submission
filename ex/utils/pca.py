import numpy as np
import torch


def fit_pca_basis(
    embeddings: np.ndarray,
    n_components: int = 64,
    seed: int = 42,
) -> dict:
	"""Fit sklearn PCA on embedding matrix; enforce deterministic sign convention;
	compute per-component standardization stats with safe floor.

	Args:
	    embeddings: [N, 768] float32 matrix of embeddings.
	    n_components: number of principal components to retain (default 64).
	    seed: random seed for sklearn PCA (ensures reproducible sign convention).

	Returns:
	    dict with numpy float32 values:
	        'mean':                     [768]      mean vector from PCA centering.
	        'components':               [n_components, 768]  principal axes, sign-flipped deterministically.
	        'explained_variance_ratio': [n_components]      variance ratio per component.
	        'code_mean':                [n_components]      mean of projected codes (should be ~ 0).
	        'code_std':                 [n_components]      raw std of projected codes; diagnostic only.
	        'code_std_clamped':         [n_components]      floor(code_std, 0.01 * code_std.max()).
	"""
	from sklearn.decomposition import PCA

	pca = PCA(n_components=n_components, svd_solver='auto', random_state=seed).fit(embeddings)
	components = pca.components_.copy()

	# deterministic sign convention: flip so max-magnitude entry is positive
	for i in range(components.shape[0]):
		j = int(np.argmax(np.abs(components[i])))
		if components[i, j] < 0:
			components[i] *= -1.0

	# project onto basis
	centered = embeddings - pca.mean_
	codes = centered @ components.T

	# compute standardization stats
	code_mean = codes.mean(axis=0)
	code_std = codes.std(axis=0, ddof=0)
	floor = 0.01 * float(code_std.max())
	code_std_clamped = np.maximum(code_std, floor)

	return {
		'mean': np.asarray(pca.mean_, dtype=np.float32),
		'components': np.asarray(components, dtype=np.float32),
		'explained_variance_ratio': np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
		'code_mean': np.asarray(code_mean, dtype=np.float32),
		'code_std': np.asarray(code_std, dtype=np.float32),
		'code_std_clamped': np.asarray(code_std_clamped, dtype=np.float32),
	}


def apply_basis(
	embeddings: torch.Tensor,
	basis: dict,
) -> torch.Tensor:
	"""Center, project, and standardize embeddings using basis dict.
	Outputs are in the same device and dtype as input embeddings.

	Args:
	    embeddings: [B, 768] torch tensor on any device (cpu/cuda), any dtype (float32/float64).
	    basis: dict of numpy float32 values from fit_pca_basis.

	Returns:
	    [B, n_components] torch tensor in same device/dtype as input embeddings.
	    Each element is standardized: (code - code_mean) / code_std_clamped.
	"""
	as_t = lambda v: torch.as_tensor(v, device=embeddings.device, dtype=embeddings.dtype)

	centered = embeddings - as_t(basis['mean'])
	codes = centered @ as_t(basis['components']).T
	std_codes = (codes - as_t(basis['code_mean'])) / as_t(basis['code_std_clamped'])

	return std_codes
