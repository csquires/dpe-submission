"""EIG estimation experiment adapter."""
import h5py
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Optional

from experiments.utils.hpo.adapters.base import ExperimentAdapter

_CONFIG_PATH = Path(__file__).resolve().parents[4] / "experiments/eig_estimation/config1.yaml"


class EIGAdapter(ExperimentAdapter):
    """EIG estimation experiment adapter.

    cell shape: 1-tuple (design_idx,).
    cell pool: [(0,), (1,), ..., (num_designs - 1,)] where
      num_designs = num_priors * num_designs_per_setting from config.

    h5 file: {data_dir}/dataset_d={data_dim},nsamples={nsamples}.h5
    h5 keys per row (indexed by design_idx):
      theta_samples_arr, y_samples_arr, design_arr, prior_covariance_arr.

    load_cell_data returns {"theta", "y", "xi", "Sigma_pi"} — the same
    names used in hpo_trial.py's eval_cell closure. EIGPlugin wrapping
    is applied at the trial_runner level, not here.
    """

    def __init__(self):
        """load config1.yaml; cache data_dir, device, latent_dim, h5 path params."""
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        self._data_dir = cfg["data_dir"]
        self._device = cfg.get("device", "cpu")
        self._data_dim = cfg["data_dim"]
        self._nsamples = cfg["nsamples"]
        self._latent_dim = self._data_dim + 1  # per mid-level doc
        self._num_designs = cfg["num_priors"] * cfg["num_designs_per_setting"]

    def name(self) -> str:
        """return "eig_estimation"."""
        return "eig_estimation"

    def data_dir(self) -> Path:
        """return Path(config["data_dir"])."""
        return Path(self._data_dir)

    def cell_pool(self) -> list[tuple[int]]:
        """return [(i,) for i in range(num_designs)]."""
        return [(i,) for i in range(self._num_designs)]

    def load_cell_data(self, cell: tuple[int, ...], device: str) -> dict[str, torch.Tensor]:
        """load one design row from the single h5 dataset file.

        args:
          cell: (design_idx,) 1-tuple.
          device: torch device string.

        opens {data_dir}/dataset_d={data_dim},nsamples={nsamples}.h5.
        extracts row design_idx from:
          theta_samples_arr, y_samples_arr, design_arr, prior_covariance_arr.
        converts to float32 tensors on device.

        returns: {"theta": tensor, "y": tensor, "xi": tensor, "Sigma_pi": tensor}.

        raises FileNotFoundError if h5 missing, ValueError if cell invalid.
        """
        (idx,) = cell
        path = self.data_dir() / f"dataset_d={self._data_dim},nsamples={self._nsamples}.h5"

        def _t(arr) -> torch.Tensor:
            return torch.from_numpy(np.array(arr)).float().to(device)

        with h5py.File(path, "r") as f:
            theta = _t(f["theta_samples_arr"][idx])    # (n_samples, data_dim)
            y = _t(f["y_samples_arr"][idx])             # (n_samples, data_dim+1)
            xi = _t(f["design_arr"][idx])               # (data_dim+1,)
            Sigma_pi = _t(f["prior_covariance_arr"][idx])  # (data_dim, data_dim)

        return {"theta": theta, "y": y, "xi": xi, "Sigma_pi": Sigma_pi}

    def device(self) -> str:
        """return config["device"] (default "cpu")."""
        return self._device

    def latent_dim(self) -> int:
        """return data_dim + 1."""
        return self._latent_dim

    def num_waypoints(self) -> Optional[int]:
        """return None; eig does not use triangular waypoints."""
        return None

    def metric_key(self) -> str:
        """return "per_design_eig_abs_err"."""
        return "per_design_eig_abs_err"
