"""EIG estimation experiment adapter.

cell shape: 1-tuple (design_idx,).
cell pool: [(0,), (1,), ..., (num_designs - 1,)] where
  num_designs = num_priors * num_designs_per_setting from config.

h5 file: {data_dir}/dataset_d={data_dim},nsamples={nsamples}.h5
h5 keys per row (indexed by design_idx):
  theta_samples_arr, y_samples_arr, design_arr, prior_covariance_arr,
  prior_mean_arr.

load_cell_data returns {"theta", "y", "xi", "Sigma_pi", "mu_pi"}.
eval_cell builds (p0, p1) on the fly via joint_and_shuffled, computes
true ldrs via the gaussian-linear closed form, fits the estimator, and
returns MAE between predict_ldr(joint) and the true per-sample ldrs.
the same MAE function is forwarded as eval_fn for hyperband pruning.
"""
import h5py
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Optional

from ex.utils.hpo.adapters.base import ExperimentAdapter
from ex.utils.eig_ldr import joint_and_shuffled, true_ldrs_gaussian_linear

_CONFIG_PATH = Path(__file__).resolve().parents[4] / "ex/synth/eig/config1.yaml"


class EIGAdapter(ExperimentAdapter):

    def __init__(self):
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        self._data_dir = cfg["data_dir"]
        self._device = cfg.get("device", "cpu")
        self._data_dim = cfg["data_dim"]
        self._nsamples = cfg["nsamples"]
        self._latent_dim = self._data_dim + 1
        self._num_designs = cfg["num_priors"] * cfg["num_designs_per_setting"]

    def name(self) -> str:
        return "eig"

    def data_dir(self) -> Path:
        return Path(self._data_dir)

    def cell_pool(self) -> list[tuple[int]]:
        return [(i,) for i in range(self._num_designs)]

    def load_cell_data(self, cell: tuple[int, ...], device: str) -> dict[str, torch.Tensor]:
        """load one design row; returns {theta, y, xi, Sigma_pi, mu_pi}.

        no p0/p1/pstar/true_ldrs precomputed: eval_cell builds them per call so
        the joint-vs-shuffled permutation is freshly drawn each trial.
        """
        (idx,) = cell
        path = self.data_dir() / f"dataset_d={self._data_dim},nsamples={self._nsamples}.h5"

        def _t(arr) -> torch.Tensor:
            return torch.from_numpy(np.array(arr)).float().to(device)

        with h5py.File(path, "r") as f:
            theta = _t(f["theta_samples_arr"][idx])
            y = _t(f["y_samples_arr"][idx])
            xi = _t(f["design_arr"][idx])
            Sigma_pi = _t(f["prior_covariance_arr"][idx])
            mu_pi = _t(f["prior_mean_arr"][idx])

        return {"theta": theta, "y": y, "xi": xi, "Sigma_pi": Sigma_pi, "mu_pi": mu_pi}

    def device(self) -> str:
        return self._device

    def latent_dim(self) -> int:
        return self._latent_dim

    def num_waypoints(self) -> Optional[int]:
        return None

    def metric_key(self) -> str:
        return "per_design_ldr_mae"

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
        """eig scoring: MAE between predict_ldr(joint) and closed-form true ldrs.

        steps:
          1. load cell data if not cached.
          2. build joint=(theta, y), shuffled=independent-marginal product, and
             precompute true_ldrs at the joint samples.
          3. derive a per-trial seed and split (joint, true_ldrs) for early
             stopping; the unsplit pair is used for the final metric.
          4. build the estimator. triangular methods (requires_pstar=True)
             use joint as pstar.
          5. forward step_cb / eval_data / step_cb_interval to est.fit.
          6. return MAE on the full joint sample.
        """
        if data is None:
            data = self.load_cell_data(cell, device=device)
        theta, y = data["theta"], data["y"]
        joint, shuffled = joint_and_shuffled(theta, y)
        true_ldrs = true_ldrs_gaussian_linear(
            theta, y, data["mu_pi"], data["Sigma_pi"], data["xi"]
        )

        # within-cell split for pruning eval_fn
        eval_data = None
        if trial_number is not None:
            seed = hash((trial_number, cell)) & 0xFFFFFFFF
            split_in = {"pstar": joint, "true_ldrs": true_ldrs}
            _, eval_part = self.split_for_eval_seeded(split_in, seed=seed)
            eval_data = {k: v.to(device) for k, v in eval_part.items()}

        nwp = hyperparams.get("num_waypoints", self.num_waypoints() or 0)
        flat = {k: v for k, v in hyperparams.items() if k != "num_waypoints"}
        est = builder(
            input_dim=self.latent_dim(),
            device=device,
            num_waypoints=nwp,
            **flat,
        )

        fit_args = (joint, shuffled, joint) if requires_pstar else (joint, shuffled)
        est.fit(
            *fit_args,
            step_cb=step_cb,
            eval_data=eval_data,
            step_cb_interval=step_cb_interval,
        )

        with torch.no_grad():
            preds = est.predict_ldr(joint)
            return float(torch.abs(preds.cpu() - true_ldrs.cpu()).mean())

    def split_for_eval_seeded(self, data: dict, *, seed: int) -> tuple[dict, dict]:
        """per-trial-seeded variant of split_for_eval used by eval_cell."""
        from ex.utils.hpo.adapters.eval_split import split_for_eval
        return split_for_eval(data, seed=seed)
