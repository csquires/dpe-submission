import os
import yaml
import numpy as np
import torch
import h5py
from typing import Dict, Any


"""
Shared HDF5 I/O, config loading, and seeding helpers.

Functions in this module are factored out from smodice and pendulum
experiment scripts to enable code reuse and avoid triplication.
New experiments should import these functions from here.
"""


def _require_path_env_roots() -> None:
	"""require DPE_DATA_ROOT and DPE_CKPT_ROOT to be set in the environment.

	no path is invented here. cluster setup (e.g. ~/.bashrc) must export
	DPE_DATA_ROOT (NFS data root) and DPE_CKPT_ROOT (node-local scratch ckpt
	root). a missing var is a hard error -- never a silent $HOME or
	cluster-hardcoded fallback. configs reference these as
	${DPE_DATA_ROOT}/<exp>/... and ${DPE_CKPT_ROOT}/<exp>/ckpt.
	"""
	for var in ("DPE_DATA_ROOT", "DPE_CKPT_ROOT"):
		if not os.environ.get(var):
			raise RuntimeError(
				f"{var} is not set -- cluster setup must export it "
				f"before running (see ~/.bashrc)."
			)


def _expand_env(value):
	"""recursively os.path.expandvars on every string in a yaml-loaded structure."""
	if isinstance(value, str):
		return os.path.expandvars(value)
	if isinstance(value, list):
		return [_expand_env(v) for v in value]
	if isinstance(value, dict):
		return {k: _expand_env(v) for k, v in value.items()}
	return value


def _install_yaml_envvar_patch() -> None:
	"""
	idempotently monkey-patch yaml.safe_load and yaml.load to expand env vars.

	belt-and-suspenders for the patch in ex/__init__.py: any script
	that imports from src.utils.io (most step1/step2/hpo_trial scripts do)
	gets the same protection even when run as `python script.py` rather than
	`python -m ex.<exp>.<step>`.
	"""
	if getattr(yaml, "_dpe_envvar_patched", False):
		return
	_orig_safe = yaml.safe_load
	_orig_load = yaml.load

	def _safe(stream):
		return _expand_env(_orig_safe(stream))

	def _load(stream, *args, **kwargs):
		return _expand_env(_orig_load(stream, *args, **kwargs))

	yaml.safe_load = _safe
	yaml.load = _load
	yaml._dpe_envvar_patched = True


_require_path_env_roots()
_install_yaml_envvar_patch()


def _load_config(config_path: str) -> Dict[str, Any]:
	"""
	load yaml config from config_path with env-var expansion.

	procedure:
		1. install DPE_DATA_ROOT and DPE_CKPT_ROOT defaults if unset.
		2. yaml.safe_load the file.
		3. recursively os.path.expandvars on every string value.

	returns dict with all keys described in config.yaml schema.
	raises FileNotFoundError if config_path does not exist.
	"""
	_require_path_env_roots()
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return _expand_env(config)


def _set_seed(seed: int) -> None:
	"""
	set global numpy and torch random seeds for reproducibility.

	args:
		seed: integer seed value; applied to np.random.seed and torch.manual_seed.

	side effects:
		np.random.seed(seed)
		torch.manual_seed(seed)
	"""
	np.random.seed(seed)
	torch.manual_seed(seed)


def _hdf5_exists(output_path: str) -> bool:
	"""
	check if HDF5 file exists (and is a valid file, not a temporary).
	returns True if path exists and is readable.
	"""
	return os.path.isfile(output_path) and not output_path.endswith('.tmp')


def _write_hdf5_atomic(
	output_path: str,
	datasets: Dict[str, np.ndarray],
	attrs: Dict[str, Any],
) -> None:
	"""
	write HDF5 atomically to avoid partial-file recovery issues.

	writes to {output_path}.tmp, then renames to output_path.

	args:
		output_path: final target path (str).
		datasets: dict of {name: array} to write as h5 datasets.
				  all arrays assumed float32 unless otherwise specified in attrs.
		attrs: dict of {name: value} scalar attributes (float, int, str, etc.).

	process:
		1. create parent directory if needed.
		2. tmp_path = output_path + ".tmp".
		3. with h5py.File(tmp_path, 'w') as f:
			 for name, arr in datasets.items():
				 f.create_dataset(name, data=arr, dtype=arr.dtype)
			 for name, val in attrs.items():
				 f.attrs[name] = val
		4. os.rename(tmp_path, output_path)  # atomic on POSIX.
	"""
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	tmp_path = output_path + ".tmp"

	with h5py.File(tmp_path, 'w') as f:
		for name, arr in datasets.items():
			f.create_dataset(name, data=arr, dtype=arr.dtype)
		for name, val in attrs.items():
			f.attrs[name] = val

	os.rename(tmp_path, output_path)
