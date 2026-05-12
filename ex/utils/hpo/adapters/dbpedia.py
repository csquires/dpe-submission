"""DBpedia ELDR experiment adapter.

structurally identical to MnistAdapter (same h5 keys, same
4 alphas x 40 pairs cell pool, same num_waypoints=15). only differences:
data_dir, latent_dim=64 (vs 14), and alpha values.
"""
import itertools
import h5py
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Optional

from ex.utils.hpo.adapters.base import ExperimentAdapter

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_PATH = _REPO_ROOT / "ex/semisynth/dbpedia/config.yaml"


class DbpediaAdapter(ExperimentAdapter):
    """DBpedia ELDR experiment adapter."""

    def __init__(self):
        with open(_CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        self._data_dir = config["data_dir"]
        self._device = config.get("device", "cuda")
        self._latent_dim = config["latent_dim"]
        self._num_waypoints = config.get("num_waypoints")
        self._n_alphas = len(config.get("alphas", [0, 1, 2, 3]))
        self._n_pairs = config.get("num_pairs_per_alpha", 40)

    def name(self) -> str:
        return "dbpedia"

    def data_dir(self) -> Path:
        return Path(self._data_dir)

    def cell_pool(self) -> list[tuple[int, int]]:
        return list(itertools.product(range(self._n_alphas), range(self._n_pairs)))

    def load_cell_data(self, cell: tuple[int, int], device: str) -> dict[str, torch.Tensor]:
        alpha_idx, pair_idx = cell
        path = self.data_dir() / f"alpha_{alpha_idx}_pair_{pair_idx}.h5"
        with h5py.File(path, "r") as f:
            pstar = torch.from_numpy(np.array(f["pstar_samples"])).float().to(device)
            p0 = torch.from_numpy(np.array(f["p0_samples"])).float().to(device)
            p1 = torch.from_numpy(np.array(f["p1_samples"])).float().to(device)
            true_ldrs = torch.from_numpy(np.array(f["true_ldrs"])).float().to(device)
        return {"pstar": pstar, "p0": p0, "p1": p1, "true_ldrs": true_ldrs}

    def device(self) -> str:
        return self._device

    def latent_dim(self) -> int:
        return self._latent_dim

    def num_waypoints(self) -> Optional[int]:
        return self._num_waypoints

    def metric_key(self) -> str:
        return "per_pair_mae"
