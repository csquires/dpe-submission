"""pendulum ELDR estimation experiment adapter.

cell arity: 3-tuple (k1_idx, k2_idx, seed).
pool: kl_targets.k1_values x k2_values x range(seeds_default).
h5 keys: samples_p0, samples_p1, samples_pstar, log_p_pstar.
file pattern: {data_dir}/k1_{k1}_k2_{k2}_seed_{seed}.h5.

per-sample true ldr is derived from log_p_pstar (shape [N, 3]) whose columns
are [pi^beta*, pi_O, pi_E] = [mix, p0, p1] per step1_create_data.py:
  true_ldrs = log p0(pstar) - log p1(pstar) = log_p_pstar[:, 1] - log_p_pstar[:, 2].
mean over samples equals the stored attr `integrated_eldr` exactly (verified).

v2 stratification: stratify_key returns (k1, k2) so cell_schema covers all
difficulty regimes even when pool is large (~480 cells) and naive random
sampling would miss ~47% of (k1, k2) combinations per draw.
"""
import itertools
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import yaml

from experiments.utils.hpo.adapters.base import ExperimentAdapter

_CONFIG_PATH = Path(__file__).resolve().parents[4] / "experiments/pendulum_eldr_estimation/config.yaml"


class PendulumAdapter(ExperimentAdapter):
    """pendulum_eldr_estimation: 3-tuple cells (k1_idx, k2_idx, seed).

    precondition: config.yaml kl_targets.k1_values and k2_values are populated.
    is_ready() returns False if either list is empty.
    """

    def __init__(self):
        """load config.yaml; cache device, data_dir, num_waypoints, k1/k2 values, seeds."""
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        self._data_dir = cfg["data_dir"]
        self._device = cfg.get("device", "cpu")
        self._num_waypoints = cfg.get("num_waypoints")
        kl = cfg.get("kl_targets", {})
        self._k1_values = kl.get("k1_values", [])
        self._k2_values = kl.get("k2_values", [])
        self._seeds = kl.get("seeds_default", 1)
        # pendulum samples are flat T=5-step trajectories, 18-dim each
        # (verified from h5 samples_p0.shape=(N, 18); the prior `= 4` was a
        # stale hardcode from an earlier theta/theta_dot-only schema).
        self._latent_dim = 18

    def name(self) -> str:
        """return "pendulum_eldr_estimation"."""
        return "pendulum_eldr_estimation"

    def data_dir(self) -> Path:
        """return Path(config["data_dir"])."""
        return Path(self._data_dir)

    def cell_pool(self) -> list[tuple[int, int, int]]:
        """return full (k1_idx, k2_idx, seed) grid.

        k1_idx in range(len(k1_values)), k2_idx in range(len(k2_values)),
        seed in range(seeds_default).
        """
        return list(itertools.product(
            range(len(self._k1_values)),
            range(len(self._k2_values)),
            range(self._seeds),
        ))

    def load_cell_data(self, cell: tuple[int, int, int], device: str) -> dict[str, torch.Tensor]:
        """load one (k1_idx, k2_idx, seed) cell from h5.

        args:
          cell: (k1_idx, k2_idx, seed) tuple.
          device: torch device string.

        opens {data_dir}/k1_{k1}_k2_{k2}_seed_{seed}.h5.
        extracts f["samples_p0"], f["samples_p1"], f["samples_pstar"], f["log_p_pstar"].
        derives per-sample true ldr at pstar samples as
          true_ldrs = log_p_pstar[:, 1] - log_p_pstar[:, 2]
        where columns of log_p_pstar are [mix, p0, p1] per step1_create_data.py.
        converts to float32 tensors on device.

        returns: {"pstar": tensor, "p0": tensor, "p1": tensor, "true_ldrs": tensor}.

        raises FileNotFoundError if h5 path missing.

        caveat: assumes log_p_pstar column order is [mix, p0, p1]. if step1's
        crossdens stack order ever changes, this derivation must be updated.
        """
        k1, k2, seed = cell
        path = self.data_dir() / f"k1_{k1}_k2_{k2}_seed_{seed}.h5"
        if not path.exists():
            raise FileNotFoundError(f"pendulum h5 not found: {path}")

        with h5py.File(path, "r") as f:
            p0 = torch.from_numpy(np.array(f["samples_p0"])).float().to(device)         # (n, d)
            p1 = torch.from_numpy(np.array(f["samples_p1"])).float().to(device)         # (n, d)
            pstar = torch.from_numpy(np.array(f["samples_pstar"])).float().to(device)   # (n, d)
            log_p_pstar = np.array(f["log_p_pstar"])                                    # (n, 3) cols [mix, p0, p1]
            true_ldrs = torch.from_numpy(log_p_pstar[:, 1] - log_p_pstar[:, 2]).float().to(device)  # (n,)

        return {"pstar": pstar, "p0": p0, "p1": p1, "true_ldrs": true_ldrs}

    def device(self) -> str:
        """return config["device"] (default "cpu")."""
        return self._device

    def latent_dim(self) -> int:
        """return 18 (flat T=5 pendulum trajectory: 6 features per timestep × 3)."""
        return self._latent_dim

    def num_waypoints(self) -> Optional[int]:
        """return config["num_waypoints"] or None."""
        return self._num_waypoints

    def metric_key(self) -> str:
        """return "per_cell_ldr_mae"."""
        return "per_cell_ldr_mae"

    def is_ready(self) -> bool:
        """return False if k1_values or k2_values are empty (not yet populated).

        guards against running HPO before step1 grid build fills kl_targets.
        """
        if not self._k1_values or not self._k2_values:
            return False
        return self.data_dir().exists()

    def stratify_key(self, cell: tuple[int, int, int]):
        """return (k1_idx, k2_idx) to stratify cell_schema by difficulty regime.

        v2 stratification: ensures every (k1, k2) difficulty pair is represented
        in each training/holdout draw, preventing ~47%-miss-rate on large pools.
        """
        return (cell[0], cell[1])
