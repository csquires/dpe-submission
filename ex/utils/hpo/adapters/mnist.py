"""MNIST-ELDR experiment adapter."""
import itertools
import h5py
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Optional

from ex.utils.hpo.adapters.base import ExperimentAdapter

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_PATH = _REPO_ROOT / "ex/semisynth/mnist/config.yaml"


class MnistAdapter(ExperimentAdapter):
    """MNIST-ELDR experiment adapter.

    cell shape: 2-tuple (alpha_idx, pair_idx).
    alphas: {0, 1, 2, 3} (4 values from config.alphas).
    pairs per alpha: {0..39} (40 pairs from config.num_pairs_per_alpha).
    total cells: 4 * 40 = 160.

    h5 keys: pstar_samples, p0_samples, p1_samples, true_ldrs.
    all keys contain float32 tensors of shape (batch_size, latent_dim).
    """

    def __init__(self):
        """load config.yaml; cache device, latent_dim, data_dir, num_waypoints."""
        with open(_CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        self._data_dir = config["data_dir"]
        self._device = config.get("device", "cuda")
        self._latent_dim = config["latent_dim"]
        self._num_waypoints = config.get("num_waypoints")

    def name(self) -> str:
        """return "mnist"."""
        return "mnist"

    def data_dir(self) -> Path:
        """return Path(config["data_dir"]); may be env-var-expanded by monkeypatch."""
        return Path(self._data_dir)

    def cell_pool(self) -> list[tuple[int, int]]:
        """return [(a, p) for a in range(4), p in range(40)]."""
        return list(itertools.product(range(4), range(40)))

    def load_cell_data(self, cell: tuple[int, int], device: str) -> dict[str, torch.Tensor]:
        """load one (alpha, pair) cell from h5 file.

        args:
          cell: (alpha_idx, pair_idx) tuple.
          device: torch device string.

        opens {data_dir}/alpha_{alpha_idx}_pair_{pair_idx}.h5.
        extracts f["pstar_samples"], f["p0_samples"], f["p1_samples"], f["true_ldrs"].
        converts to float32 tensors on device.

        returns: {"pstar": tensor, "p0": tensor, "p1": tensor, "true_ldrs": tensor}.

        raises FileNotFoundError if h5 path missing.
        """
        alpha_idx, pair_idx = cell
        path = self.data_dir() / f"alpha_{alpha_idx}_pair_{pair_idx}.h5"

        with h5py.File(path, "r") as f:
            pstar = torch.from_numpy(np.array(f["pstar_samples"])).float().to(device)
            p0 = torch.from_numpy(np.array(f["p0_samples"])).float().to(device)
            p1 = torch.from_numpy(np.array(f["p1_samples"])).float().to(device)
            true_ldrs = torch.from_numpy(np.array(f["true_ldrs"])).float().to(device)

        return {
            "pstar": pstar,
            "p0": p0,
            "p1": p1,
            "true_ldrs": true_ldrs,
        }

    def device(self) -> str:
        """return config["device"] (default "cuda")."""
        return self._device

    def latent_dim(self) -> int:
        """return config["latent_dim"] (14)."""
        return self._latent_dim

    def num_waypoints(self) -> Optional[int]:
        """return config["num_waypoints"] (15)."""
        return self._num_waypoints

    def metric_key(self) -> str:
        """return "per_pair_mae"."""
        return "per_pair_mae"

    def stratify_key(self, cell: tuple[int, int]):
        """return alpha_idx (cell[0]) for per-alpha stratification.

        guarantees every alpha regime is sampled in each training/holdout
        draw, mirroring the elbo adapter.
        """
        return cell[0]
