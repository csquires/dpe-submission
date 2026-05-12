"""model_selection experiment adapter.

cell shape: 1-tuple (row_idx,).
single h5 file: {data_dir}/dataset_newpstar.h5.
h5 keys (row-indexed): samples_p0_arr, samples_p1_arr,
  samples_pstar_arr[row, 0], true_ldrs_arr[row, 0],
  optionally samples_pstar_train_arr[row].
"""
import logging
import warnings
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import yaml

from experiments.utils.hpo.adapters.base import ExperimentAdapter

_logger = logging.getLogger(__name__)
_CONFIG_PATH = Path(__file__).resolve().parents[4] / "experiments/model_selection/config.yaml"


class ModelSelectionAdapter(ExperimentAdapter):
    """model_selection: 1-tuple cells (row_idx,).

    pool: all row indices 0 .. n_kl * n_instances - 1.
    default_training_M: min(7, pool_size).
    default_holdout_M: max(1, pool_size - training_M).
    device fallback: if config says cuda but unavailable, warn and use cpu.
    """

    def __init__(self):
        """load config.yaml; cache device, latent_dim, data_dir, num_waypoints, pool."""
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        self._data_dir = cfg["data_dir"]
        self._latent_dim = cfg["data_dim"]
        self._num_waypoints = cfg.get("num_waypoints", None)
        # cache raw device; fallback applied lazily in device()
        self._device_cfg = cfg.get("device", "cpu")

        n_kl = len(cfg.get("kl_distances", []))
        n_inst = cfg.get("num_instances_per_kl", 1)
        self._pool: list[tuple[int]] = [(r,) for r in range(n_kl * n_inst)]

    def name(self) -> str:
        return "model_selection"

    def data_dir(self) -> Path:
        return Path(self._data_dir)

    def cell_pool(self) -> list[tuple[int]]:
        return self._pool

    def load_cell_data(self, cell: tuple[int], device: str) -> dict[str, torch.Tensor]:
        """open single h5, slice row, return normalized dict.

        args:
          cell: (row_idx,).
          device: torch device string.

        opens {data_dir}/dataset_newpstar.h5.
        extracts samples_p0_arr[row], samples_p1_arr[row],
          samples_pstar_arr[row, 0], true_ldrs_arr[row, 0].
        optionally extracts samples_pstar_train_arr[row] if key present.
        converts all to float32 tensors on device.

        returns dict with keys: pstar, p0, p1, true_ldrs [, pstar_train].

        raises FileNotFoundError if h5 missing.
        """
        (row,) = cell
        path = self.data_dir() / "dataset_newpstar.h5"

        with h5py.File(path, "r") as f:
            p0 = torch.from_numpy(np.array(f["samples_p0_arr"][row])).float().to(device)       # (ntrain, dim)
            p1 = torch.from_numpy(np.array(f["samples_p1_arr"][row])).float().to(device)       # (ntrain, dim)
            pstar = torch.from_numpy(np.array(f["samples_pstar_arr"][row][0])).float().to(device)  # (ntest, dim)
            true_ldrs = torch.from_numpy(np.array(f["true_ldrs_arr"][row][0])).float().to(device)  # (ntest,)

            out = {"pstar": pstar, "p0": p0, "p1": p1, "true_ldrs": true_ldrs}

            if "samples_pstar_train_arr" in f:
                out["pstar_train"] = (
                    torch.from_numpy(np.array(f["samples_pstar_train_arr"][row])).float().to(device)  # (ntrain, dim)
                )

        return out

    def device(self) -> str:
        """return config device; fall back to cpu if cuda unavailable."""
        if self._device_cfg == "cuda" and not torch.cuda.is_available():
            warnings.warn("cuda not available, falling back to cpu")
            _logger.warning("cuda requested but unavailable; using cpu")
            return "cpu"
        return self._device_cfg

    def latent_dim(self) -> int:
        """return config data_dim (3)."""
        return self._latent_dim

    def num_waypoints(self) -> Optional[int]:
        """return config num_waypoints if present, else None."""
        return self._num_waypoints

    def metric_key(self) -> str:
        return "per_row_ldr_mean_ae"

    def eval_cell(
        self,
        cell,
        method,
        builder,
        hyperparams,
        requires_pstar,
        device,
        *,
        step_cb=None,
        trial_number: int | None = None,
        step_cb_interval: int = 50,
        data=None,
    ):
        """model_selection scoring: fits with pstar_train, scores on pstar.

        differs from default in that the fit-time pstar (when requires_pstar)
        is the per-cell pstar_train slice, not the test pstar. metric is
        mae(predict_ldr(pstar_test), true_ldrs).

        new kwargs (step_cb, trial_number, step_cb_interval) are accepted but
        not used in this iteration; deferred pending integration of eval-split
        semantics with the pstar_train / pstar distinction.
        """
        import torch

        if data is None:
            data = self.load_cell_data(cell, device=device)
        nwp = hyperparams.get("num_waypoints", self.num_waypoints())
        flat = {k: v for k, v in hyperparams.items() if k != "num_waypoints"}
        est = builder(
            input_dim=self.latent_dim(),
            device=device,
            num_waypoints=nwp,
            **flat,
        )
        if requires_pstar:
            ps_train = data.get("pstar_train", data["pstar"])
            est.fit(data["p0"], data["p1"], ps_train)
        else:
            est.fit(data["p0"], data["p1"])
        with torch.no_grad():
            predicted = est.predict_ldr(data["pstar"])
            return float(torch.abs(predicted.cpu() - data["true_ldrs"].cpu()).mean())

    def default_training_M(self) -> int:
        """min(7, pool_size); pool is typically tiny (7 rows)."""
        return min(7, len(self._pool))

    def default_holdout_M(self) -> int:
        """max(1, pool_size - training_M)."""
        return max(1, len(self._pool) - self.default_training_M())
