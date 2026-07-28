"""dokls ablation adapter: bridges dokls HPO to the shared adapter registry.

cell shape: 1-tuple (row_idx,) where row_idx in 0..69 (70 total Gaussian instances).
h5 data loaded from {data_dir}/dataset.h5 with keys:
  samples_p0_arr [70, 8192, 3]
  samples_p1_arr [70, 8192, 3]
  pstar_arr [70, 2, 8192, 3]  (idx 0=q0, 1=q1; sliced [:N] for this trial)
  true_ldrs_arr [70, 2, 8192]
  true_eldr_arr [70, 2]
"""
import logging
import warnings
from pathlib import Path
from typing import Optional, Callable

import h5py
import numpy as np
import torch
import yaml

from ex.utils.hpo.adapters.base import ExperimentAdapter

_logger = logging.getLogger(__name__)
_CONFIG_PATH = Path(__file__).resolve().parents[4] / "ex/ablations/dokls/config.yaml"


class DoklsAdapter(ExperimentAdapter):
    """dokls ablation adapter: 1-tuple cells (row_idx,).

    cell shape: (row_idx,) where row_idx in 0..69 (70 total Gaussian instances).
    layout: 10 instances per KL regime (7 regimes), rows ordered as
      kl_idx * num_instances + instance_idx.

    pool: all row indices 0..69.
    stratified split: 10 cells per stratum (KL regime); 8 train + 2 holdout per stratum;
      step2 = -1 (all non-train/holdout cells routed to step2_pool).

    stratified by row // 10 (KL regime index, 0..6).
    """

    def __init__(
        self,
        pstar_idx: int,
        N: int,
        nstar: int | None = None,
        config_path: Path | None = None,
        route: str = "two_leg",
    ):
        """initialize dokls adapter with pstar choice, sample budget, and route.

        args:
            pstar_idx: in {0, 1} (q0=0, q1=1). raises ValueError if invalid.
            N: in {1024, 2048, 4096, 8192}. raises ValueError if invalid or > 8192.
            config_path: path to config.yaml. if None, use repo-relative fallback.
            route: in {"two_leg", "direct"}. raises ValueError if invalid.

        behavior:
            - load config.yaml; extract data_dir, device_cfg, latent_dim (expect 3),
              num_instances (expect 10).
            - cache _pstar_idx, _N, _route, _latent_dim, _data_dir, _device_cfg, _num_instances.
            - build _pool = [(r,) for r in range(70)].
            - do NOT validate data_dir exists at __init__ time; defer to is_ready().

        edge cases:
            - config missing keys: raise KeyError with helpful message.
            - config device="cuda" but torch.cuda.is_available() is False: warn and set
              _device_cfg="cpu" at init time.
        """
        # validate pstar_idx
        if pstar_idx not in {0, 1}:
            raise ValueError(f"pstar_idx must be in {{0, 1}}, got {pstar_idx}")

        # validate N (p0/p1 budget) and Nstar (p* budget; defaults to N == diagonal)
        if N not in {1024, 2048, 4096, 8192}:
            raise ValueError(f"N must be in {{1024, 2048, 4096, 8192}}, got {N}")
        if nstar is None:
            nstar = N
        if nstar not in {1024, 2048, 4096, 8192}:
            raise ValueError(f"nstar must be in {{1024, 2048, 4096, 8192}}, got {nstar}")

        # validate route
        if route not in {"two_leg", "direct"}:
            raise ValueError(f"route must be in {{'two_leg', 'direct'}}, got {route!r}")

        # load config
        path = config_path if config_path is not None else _CONFIG_PATH
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"config not found at {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"config YAML error: {e}") from e

        # extract and validate config keys
        try:
            self._data_dir = cfg["data_dir"]
        except KeyError:
            raise KeyError("config missing data_dir")

        self._device_cfg = cfg.get("device", "cpu")
        self._latent_dim = cfg.get("data_dim", 3)
        self._num_instances = cfg.get("num_instances_per_kl", 10)

        # fallback device if cuda unavailable
        if self._device_cfg == "cuda" and not torch.cuda.is_available():
            warnings.warn("cuda requested but unavailable; using cpu")
            _logger.warning("cuda requested but unavailable; using cpu")
            self._device_cfg = "cpu"

        # cache state
        self._pstar_idx = pstar_idx
        self._N = N
        self._nstar = nstar
        self._route = route
        self._pool: list[tuple[int]] = [(r,) for r in range(70)]

    def name(self) -> str:
        """experiment identifier for seed namespace and yaml keys.

        format: dokls_q{pstar_idx}_N{N}[_ns{Nstar}] (the _ns suffix only when
        decoupled, so ns2048/ns4096 don't collide in the seed namespace).
        """
        base = f"dokls_q{self._pstar_idx}_N{self._N}"
        return base if self._nstar == self._N else f"{base}_ns{self._nstar}"

    def data_dir(self) -> Path:
        """return NFS-shared data root directory."""
        return Path(self._data_dir)

    def cell_pool(self) -> list[tuple[int]]:
        """return full evaluation cell list: [(0,), (1,), ..., (69,)]."""
        return self._pool

    def load_cell_data(self, cell: tuple[int], device: str) -> dict[str, torch.Tensor]:
        """load single row from h5, slice by (pstar_idx, N), return normalized dict.

        args:
            cell: (row_idx,) where row_idx in 0..69.
            device: torch device string (cpu or cuda).

        algorithm:
            1. unpack cell = (row_idx,).
            2. open h5 file in read mode.
            3. load and slice (all to float32 tensors on device):
               - p0 = h5["samples_p0_arr"][row_idx][:N]  (N, 3)
               - p1 = h5["samples_p1_arr"][row_idx][:N]  (N, 3)
               - pstar = h5["pstar_arr"][row_idx, self._pstar_idx, :N]  (N, 3)
               - true_ldrs = h5["true_ldrs_arr"][row_idx, self._pstar_idx, :N]  (N,)
               - true_eldr = h5["true_eldr_arr"][row_idx, self._pstar_idx]  (scalar)
            4. convert all to float32 torch tensors on device.
            5. return dict with keys: pstar, p0, p1, true_ldrs, true_eldr.

        raises:
            FileNotFoundError: if h5 file missing.
            ValueError: if row_idx out of bounds.
        """
        (row_idx,) = cell
        path = self.data_dir() / "dataset.h5"

        try:
            with h5py.File(path, "r") as f:
                # all slicing via h5 indexing; convert to tensors on device
                # p0/p1 use the N budget; p* (and its true ldrs) use the Nstar budget.
                p0 = torch.from_numpy(np.array(f["samples_p0_arr"][row_idx][:self._N])).float().to(device)  # (N, 3)
                p1 = torch.from_numpy(np.array(f["samples_p1_arr"][row_idx][:self._N])).float().to(device)  # (N, 3)
                pstar = torch.from_numpy(np.array(f["pstar_arr"][row_idx, self._pstar_idx, :self._nstar])).float().to(device)  # (Nstar, 3)
                true_ldrs = torch.from_numpy(np.array(f["true_ldrs_arr"][row_idx, self._pstar_idx, :self._nstar])).float().to(device)  # (Nstar,)
                true_eldr = torch.from_numpy(np.array(f["true_eldr_arr"][row_idx, self._pstar_idx])).float().to(device)  # scalar

        except (IndexError, KeyError) as e:
            if isinstance(e, IndexError):
                raise ValueError(f"cell {cell} out of range [0, 70)")
            else:
                raise FileNotFoundError(f"h5 file missing or malformed: {path}") from e

        return {
            "pstar": pstar,
            "p0": p0,
            "p1": p1,
            "true_ldrs": true_ldrs,
            "true_eldr": true_eldr,
        }

    def device(self) -> str:
        """return torch device string; fall back to cpu if cuda unavailable."""
        if self._device_cfg == "cuda" and not torch.cuda.is_available():
            warnings.warn("cuda not available, falling back to cpu")
            _logger.warning("cuda requested but unavailable; using cpu")
            return "cpu"
        return self._device_cfg

    def latent_dim(self) -> int:
        """return input dimension for estimator builder (expect 3 from config)."""
        return self._latent_dim

    def num_waypoints(self) -> Optional[int]:
        """return None; dokls does not use triangular methods."""
        return None

    def metric_key(self) -> str:
        """return metric dict key for optimizer: per_sample_mae."""
        return "per_sample_mae"

    def stratify_key(self, cell: tuple[int]) -> int:
        """return KL regime index for stratification.

        rows are laid out as kl_idx * num_instances + instance_idx,
        so kl_idx = row // num_instances.

        example: if num_instances=10, cells 0-9 -> stratum 0, cells 10-19 -> stratum 1, etc.
        """
        return cell[0] // self._num_instances

    def n_train_per_stratum(self) -> int:
        """per-stratum cell count routed to optuna train pool."""
        return 8

    def n_holdout_per_stratum(self) -> int:
        """per-stratum cell count routed to holdout pool."""
        return 2

    def n_step2_per_stratum(self) -> int:
        """per-stratum cell count routed to step2 pool.

        -1 means send ALL non-train/non-holdout cells to step2_pool.
        """
        return -1

    def eval_cell(
        self,
        cell: tuple[int],
        method: str,
        builder,
        hyperparams: dict,
        requires_pstar: bool,
        device: str,
        *,
        step_cb: Optional[Callable[[int, float], None]] = None,
        trial_number: Optional[int] = None,
        step_cb_interval: int = 50,
        data: Optional[dict] = None,
    ) -> float:
        """build estimator via dokls-local factory, fit, return per-sample MAE in-sample.

        CRITICAL differences from base eval_cell:
        1. ignores passed `builder` parameter entirely (dokls uses local factory).
        2. forwards step_cb/step_cb_interval into est.fit (NOT dropped like model_selection).
        3. passes eval_data={"true_ldrs": ...} to est.fit so two-leg rung eval_fn fires step_cb.
        4. returns per-sample MAE in-sample on the same pstar[:N] used for fit.

        args:
            cell: (row_idx,).
            method: method name, e.g. "BDRE_NWJ". passed to factory.build.
            builder: ignored. dokls uses factory.build internally.
            hyperparams: dict of hyperparams; passed to factory.build as **flat.
            requires_pstar: typically True for dokls; if False, logs warning but proceeds.
            device: torch device string.
            step_cb: optional callback(step, loss) invoked by fit at intervals.
            trial_number: optional trial id (used by pruning scheduler).
            step_cb_interval: interval (steps) at which to invoke step_cb.
            data: optional preloaded data dict; if None, calls load_cell_data.

        returns:
            float: per-sample MAE between est.predict_ldr(pstar[:N]) and true_ldrs[:N].

        raises:
            ValueError: if method not in factory.LEG_BUILDERS or route invalid.
            FileNotFoundError: if data_dir or h5 missing.
        """
        # load or reuse data
        if data is None:
            data = self.load_cell_data(cell, device=device)

        # defensive: if requires_pstar is False, log warning (dokls always requires pstar)
        if not requires_pstar:
            _logger.warning("requires_pstar=False but dokls always requires pstar; proceeding anyway")

        # flatten hyperparams (no num_waypoints filtering needed; dokls methods don't use waypoints)
        flat = hyperparams

        # build estimator via dokls-local factory
        try:
            from ex.ablations.dokls import factory
        except ImportError as e:
            raise ImportError("dokls.factory not found; check ex/ablations/dokls/ exists") from e

        est = factory.build(
            method=method,
            route=self._route,
            input_dim=self._latent_dim,
            device=device,
            **flat,
        )

        # fit estimator with step_cb forwarded (CRITICAL for Hyperband pruning)
        est.fit(
            data["p0"],
            data["p1"],
            data["pstar"],
            step_cb=step_cb,
            eval_data={"true_ldrs": data["true_ldrs"]},
            step_cb_interval=step_cb_interval,
        )

        # compute metric in-sample on the same pstar[:N] used for fit
        with torch.no_grad():
            predicted = est.predict_ldr(data["pstar"])  # (N,)
            mae = float(torch.abs(predicted.cpu() - data["true_ldrs"].cpu()).mean())
            return mae
