"""ELBO estimation experiment adapter."""
import itertools
import h5py
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Optional

from experiments.utils.hpo.adapters.base import ExperimentAdapter

# assumption: config1.yaml is the canonical config (config.yaml does not exist)
_CONFIG_PATH = Path(__file__).resolve().parents[4] / "experiments/elbo_estimation/config1.yaml"


class ELBOAdapter(ExperimentAdapter):
    """ELBO estimation experiment adapter.

    cell shape: 2-tuple (alpha_idx, flat_idx).
    alpha_idx: {0, 1, 2, 3} (4 alphas).
    flat_idx: {0 .. num_priors * len(design_eig_percentages) * num_designs_per_setting - 1}.

    h5 keys (dataset): theta0_samples_arr, y0_samples_arr, theta1_samples_arr,
      y1_samples_arr, theta_star_samples_arr, y_star_samples_arr.
      all indexed by flat_idx along axis=0.
    h5 keys (processed): true_eldrs (flat array, indexed by flat_idx).

    p0 = cat([theta0, y0], dim=-1); p1, pstar built analogously.
    latent_dim = data_dim + 1.
    """

    def __init__(self):
        """load config1.yaml; cache device, dims, dirs, pool metadata."""
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        self._data_dir = cfg["data_dir"]
        self._processed_results_dir = cfg["processed_results_dir"]
        self._device = cfg.get("device", "cuda")
        self._data_dim = cfg["data_dim"]
        self._latent_dim = self._data_dim + 1
        self._nsamples = cfg["nsamples"]
        self._num_waypoints = cfg.get("num_waypoints", None)

        # pool metadata
        self._num_alphas = len(cfg["alphas"])
        self._num_priors = cfg["num_priors"]
        self._design_eig_percentages = cfg["design_eig_percentages"]
        self._num_designs_per_setting = cfg["num_designs_per_setting"]

    def name(self) -> str:
        """return "elbo_estimation"."""
        return "elbo_estimation"

    def data_dir(self) -> Path:
        """return Path(config["data_dir"]) with env-var expansion."""
        return Path(self._data_dir)

    def cell_pool(self) -> list[tuple[int, int]]:
        """return [(a, f) for a in range(num_alphas), f in range(n_flat)].

        n_flat = num_priors * len(design_eig_percentages) * num_designs_per_setting.
        """
        n_flat = (
            self._num_priors
            * len(self._design_eig_percentages)
            * self._num_designs_per_setting
        )
        return list(itertools.product(range(self._num_alphas), range(n_flat)))

    def load_cell_data(self, cell: tuple[int, int], device: str) -> dict[str, torch.Tensor]:
        """load one (alpha_idx, flat_idx) cell from dataset + processed_results h5.

        args:
          cell: (alpha_idx, flat_idx). alpha_idx is unused for indexing; flat_idx
            indexes into all h5 arrays along axis=0.
          device: torch device string.

        opens {data_dir}/dataset_d={data_dim},nsamples={nsamples}.h5.
          extracts theta0/y0/theta1/y1/theta_star/y_star at flat_idx.
          concatenates theta+y along dim=-1 to form p0, p1, pstar.
        opens {processed_results_dir}/errors_d={data_dim},nsamples={nsamples}.h5.
          extracts true_eldrs[flat_idx] as scalar tensor.

        returns: {"pstar": (N, D+1), "p0": (N, D+1), "p1": (N, D+1),
                  "true_ldrs": scalar tensor}.

        raises FileNotFoundError if h5 path missing.
        """
        _alpha_idx, flat_idx = cell

        dname = f"dataset_d={self._data_dim},nsamples={self._nsamples}.h5"
        pname = f"errors_d={self._data_dim},nsamples={self._nsamples}.h5"
        dpath = self.data_dir() / dname
        ppath = Path(self._processed_results_dir) / pname

        with h5py.File(dpath, "r") as f:
            t0 = torch.from_numpy(np.array(f["theta0_samples_arr"][flat_idx])).float().to(device)  # (N, D)
            y0 = torch.from_numpy(np.array(f["y0_samples_arr"][flat_idx])).float().to(device)      # (N, 1)
            t1 = torch.from_numpy(np.array(f["theta1_samples_arr"][flat_idx])).float().to(device)  # (N, D)
            y1 = torch.from_numpy(np.array(f["y1_samples_arr"][flat_idx])).float().to(device)      # (N, 1)
            ts = torch.from_numpy(np.array(f["theta_star_samples_arr"][flat_idx])).float().to(device)  # (N, D)
            ys = torch.from_numpy(np.array(f["y_star_samples_arr"][flat_idx])).float().to(device)      # (N, 1)

        with h5py.File(ppath, "r") as f:
            true_ldr = torch.tensor(float(f["true_eldrs"][flat_idx])).to(device)  # scalar

        return {
            "p0": torch.cat([t0, y0], dim=-1),      # (N, D+1)
            "p1": torch.cat([t1, y1], dim=-1),      # (N, D+1)
            "pstar": torch.cat([ts, ys], dim=-1),   # (N, D+1)
            "true_ldrs": true_ldr,                  # scalar
        }

    def device(self) -> str:
        """return config["device"] (default "cuda")."""
        return self._device

    def latent_dim(self) -> int:
        """return data_dim + 1."""
        return self._latent_dim

    def num_waypoints(self) -> Optional[int]:
        """return config["num_waypoints"] or None."""
        return self._num_waypoints

    def metric_key(self) -> str:
        """return "per_cell_eldr_abs_err"."""
        return "per_cell_eldr_abs_err"

    def stratify_key(self, cell: tuple[int, int]):
        """return alpha_idx (cell[0]) for per-alpha stratification."""
        return cell[0]
