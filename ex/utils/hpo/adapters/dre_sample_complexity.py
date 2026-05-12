"""DRE sample complexity experiment adapter.

cells are 2-tuples (kl_idx, instance_idx):
  kl_idx    in range(len(kl_divergences)) = 3
  instance_idx in range(num_instances_per_kl) = 20
  total pool: 60 cells

data file: {data_dir}/dataset.h5
  samples_p0_arr[row]   -> p0 samples, subsampled to hpo_nsamples_train
  samples_p1_arr[row]   -> p1 samples, subsampled to hpo_nsamples_train
  samples_pstar_arr[row] -> pstar samples (full)
  true_ldrs_arr[row]    -> ground-truth log-density ratios
  row = kl_idx * num_instances_per_kl + instance_idx

stratify_key returns kl_idx so draw_training_sample covers all three KL regimes.
"""

import h5py
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Optional

from ex.utils.hpo.adapters.base import ExperimentAdapter

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_PATH = _REPO_ROOT / "ex/ablations/dre_sample_complexity/config.yaml"


class DreSampleComplexityAdapter(ExperimentAdapter):

    def __init__(self):
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        self._cfg = cfg
        self._data_dir = Path(cfg["data_dir"])
        self._data_dim: int = cfg["data_dim"]
        self._n_kl: int = len(cfg["kl_divergences"])
        self._n_instances: int = cfg["num_instances_per_kl"]
        self._hpo_nsamples: int = cfg["hpo_nsamples_train"]

    def name(self) -> str:
        return "dre_sample_complexity"

    def data_dir(self) -> Path:
        return self._data_dir

    def is_ready(self) -> bool:
        return (self._data_dir / "dataset.h5").exists()

    def cell_pool(self) -> list[tuple[int, int]]:
        return [
            (kl_idx, inst_idx)
            for kl_idx in range(self._n_kl)
            for inst_idx in range(self._n_instances)
        ]

    def load_cell_data(self, cell: tuple[int, int], device: str) -> dict[str, torch.Tensor]:
        kl_idx, inst_idx = cell
        row = kl_idx * self._n_instances + inst_idx
        path = self._data_dir / "dataset.h5"

        def _t(arr) -> torch.Tensor:
            return torch.from_numpy(np.array(arr)).float().to(device)

        with h5py.File(path, "r") as f:
            p0 = _t(f["samples_p0_arr"][row][: self._hpo_nsamples])
            p1 = _t(f["samples_p1_arr"][row][: self._hpo_nsamples])
            pstar = _t(f["samples_pstar_arr"][row])
            true_ldrs = _t(f["true_ldrs_arr"][row])

        return {"p0": p0, "p1": p1, "pstar": pstar, "true_ldrs": true_ldrs}

    def device(self) -> str:
        return "cpu"

    def latent_dim(self) -> int:
        return self._data_dim

    def num_waypoints(self) -> Optional[int]:
        # cpu_runner uses METHOD_SPECS[method]["num_waypoints"] per-method;
        # this value is used by the slurm trial_runner path. return the mdre
        # default since it's the primary method for this experiment.
        return self._cfg.get("mdre_num_waypoints", 15)

    def metric_key(self) -> str:
        return "per_cell_mae"

    def stratify_key(self, cell: tuple[int, int]):
        # stratify by kl_idx so draw_training_sample covers all three kl regimes
        return cell[0]

    def default_training_M(self) -> int:
        # pool=60; pick 3 per kl regime (3 strata * 3 = 9, rounds to closest multiple)
        return 9

    def default_holdout_M(self) -> int:
        return 9
