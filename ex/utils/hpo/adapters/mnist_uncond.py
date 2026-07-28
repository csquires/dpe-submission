"""MNIST unconditioned estimation experiment adapter.

structurally identical to MnistCondFlowAdapter (same h5 keys, same
4 alphas x 40 pairs cell pool, same latent_dim=14, same num_waypoints=15).
only the data_dir + alpha values differ.
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
_CONFIG_PATH = _REPO_ROOT / "ex/semisynth/mnist_uncond/config.yaml"


class MnistUncondAdapter(ExperimentAdapter):
    """MNIST unconditioned estimation experiment adapter."""

    def __init__(self):
        with open(_CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        self._data_dir = config["data_dir"]
        # autodetect (like dbpedia/eig/occupancy): the array cpu lane has no GPU,
        # so a hardcoded 'cuda' made every cpu-method trial fail 'No CUDA GPUs
        # available'. fall back to cpu when cuda is absent -> trials run wherever dispatched.
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._latent_dim = config["latent_dim"]
        self._num_waypoints = config.get("num_waypoints")
        self._n_alphas = len(config.get("alphas", [0, 1, 2, 3]))
        self._n_pairs = config.get("num_pairs_per_alpha", 40)
        self._num_samples = config.get("num_samples")
        self._seed = int(config.get("seed", 0))

    def name(self) -> str:
        return "mnist_uncond"

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
        # cap p0/p1 to num_samples via a SEEDED RANDOM subsample (retain the class
        # balance of the reused 20k pool); same seed as the step2 adapter_base so
        # hpo and step2 train on the identical 10k.
        n = self._num_samples
        if n is not None:
            if p0.shape[0] > n:
                g = torch.Generator().manual_seed(self._seed)
                p0 = p0[torch.randperm(p0.shape[0], generator=g)[:n].to(p0.device)]
            if p1.shape[0] > n:
                g = torch.Generator().manual_seed(self._seed)
                p1 = p1[torch.randperm(p1.shape[0], generator=g)[:n].to(p1.device)]
        return {"pstar": pstar, "p0": p0, "p1": p1, "true_ldrs": true_ldrs}

    def device(self) -> str:
        return self._device

    def latent_dim(self) -> int:
        return self._latent_dim

    def num_waypoints(self) -> Optional[int]:
        return self._num_waypoints

    def metric_key(self) -> str:
        return "per_pair_mae"

    def stratify_key(self, cell: tuple[int, int]):
        """return alpha_idx (cell[0]) for per-alpha stratification."""
        return cell[0]

    # 40 pairs/alpha: base defaults (24/8/68) overflow the pool (24+8+68>40) and
    # raise in stratified_split_3way; mirror the dbpedia split (8 optuna + 2 holdout
    # per stratum, remainder -> step2).
    def n_train_per_stratum(self) -> int:
        return 8

    def n_holdout_per_stratum(self) -> int:
        return 2

    def n_step2_per_stratum(self) -> int:
        return -1
