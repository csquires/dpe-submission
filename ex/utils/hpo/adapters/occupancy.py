"""Occupancy gridworld ELDR estimation experiment adapter."""
import itertools
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import yaml

from ex.utils.hpo.adapters.base import ExperimentAdapter

_CONFIG_PATH = Path(__file__).resolve().parents[4] / "ex/synth/occupancy/config.yaml"


def _encoding_subdir(enc_cfg: dict, base: str) -> str:
    """resolve <base>/<type>/sigma_{na|XXX} for data path construction.

    onehot* -> sigma_na suffix (no bandwidth param).
    blob/flow -> sigma_{value:.3f} suffix.
    """
    t = enc_cfg["type"]
    if t.startswith("onehot"):
        return os.path.join(base, t, "sigma_na")
    return os.path.join(base, t, f"sigma_{enc_cfg['sigma']:.3f}")


def _input_dim(enc_cfg: dict, n_states: int, n_actions: int) -> int:
    """map encoding type to dnn input dimension."""
    t = enc_cfg["type"]
    if t == "onehot_joint":
        return n_states * n_actions
    if t == "onehot_concat":
        return n_states + n_actions
    if t in ("gaussian_blob", "flow_pushforward"):
        return enc_cfg.get("embed_dim", 6)
    raise ValueError(f"unknown encoding type: {t}")


class OccupancyAdapter(ExperimentAdapter):
    """Occupancy gridworld ELDR estimation adapter.

    cell shape: 3-tuple (k1_idx, beta_idx, seed).
    pool: k1_values x beta_values x range(seeds_default) from config.kl_targets.
    total cells: 4 * 1 * 40 = 160 (default config).

    h5 keys: p0_samples, p1_samples, pstar_samples,
             true_ldrs_discrete (onehot) or true_ldrs_smoothed (blob/flow).
    encoding subdir from _encoding_subdir(); discrete vs smoothed dispatched
    by encoding type at __init__ time.
    """

    def __init__(self):
        """load config.yaml; resolve encoding subdir, input_dim, true_ldrs_key."""
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        self._device = cfg.get("device", "cuda")
        self._num_waypoints = cfg.get("num_waypoints")

        kt = cfg["kl_targets"]
        self._k1_values = kt["k1_values"]
        self._beta_values = kt["beta_values"]
        self._n_seeds = int(kt["seeds_default"])

        enc = dict(cfg["encoding"])
        L = cfg["gridworld"]["L"]
        n_states, n_actions = L * L, 4
        self._latent_dim = _input_dim(enc, n_states, n_actions)

        raw_base = cfg["data_dir"]
        self._data_dir = Path(_encoding_subdir(enc, raw_base))

        enc_type = enc["type"]
        self._ldr_key = "true_ldrs_discrete" if enc_type.startswith("onehot") else "true_ldrs_smoothed"

    def name(self) -> str:
        """return "occupancy"."""
        return "occupancy"

    def data_dir(self) -> Path:
        """return resolved encoding subdir path."""
        return self._data_dir

    def cell_pool(self) -> list[tuple[int, int, int]]:
        """return [(k1, b, s)] for k1 in n_k1, b in n_beta, s in n_seeds."""
        n_k1 = len(self._k1_values)
        n_beta = len(self._beta_values)
        return list(itertools.product(range(n_k1), range(n_beta), range(self._n_seeds)))

    def load_cell_data(self, cell: tuple[int, int, int], device: str) -> dict[str, torch.Tensor]:
        """load one (k1_idx, beta_idx, seed) cell from h5 file.

        opens {data_dir}/kl1_K1_beta_B_seed_S.h5.
        extracts p0_samples, p1_samples, pstar_samples, and
        true_ldrs_discrete or true_ldrs_smoothed (dispatched at init).
        converts to float32 tensors on device.

        returns {"pstar": T, "p0": T, "p1": T, "true_ldrs": T}.
        raises FileNotFoundError if h5 missing.
        """
        k1, beta, seed = cell
        path = self._data_dir / f"kl1_{k1}_beta_{beta}_seed_{seed}.h5"
        with h5py.File(path, "r") as f:
            p0 = torch.from_numpy(np.array(f["p0_samples"])).float().to(device)       # (N, d)
            p1 = torch.from_numpy(np.array(f["p1_samples"])).float().to(device)       # (N, d)
            pstar = torch.from_numpy(np.array(f["pstar_samples"])).float().to(device) # (N, d)
            true_ldrs = torch.from_numpy(np.array(f[self._ldr_key])).float().to(device)  # (N,)
        return {"pstar": pstar, "p0": p0, "p1": p1, "true_ldrs": true_ldrs}

    def device(self) -> str:
        """return config["device"]."""
        return self._device

    def latent_dim(self) -> int:
        """return input dim derived from encoding type and gridworld dims."""
        return self._latent_dim

    def num_waypoints(self) -> Optional[int]:
        """return config["num_waypoints"]."""
        return self._num_waypoints

    def metric_key(self) -> str:
        """return "per_cell_ldr_mae"."""
        return "per_cell_ldr_mae"

    def supports_tabular(self) -> bool:
        """return True — occupancy has TabularPluginDRE methods."""
        return True

    def stratify_key(self, cell: tuple[int, int, int]):
        """return (k1_idx, beta_idx) for stratified cell sampling.

        guarantees all 4 (k1, beta) regimes are covered when sampling
        from the 160-cell pool.
        """
        return (cell[0], cell[1])
