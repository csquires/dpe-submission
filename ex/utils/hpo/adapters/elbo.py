"""ELBO estimation experiment adapter."""
import itertools
import h5py
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Optional

from ex.utils.hpo.adapters.base import ExperimentAdapter

# assumption: config1.yaml is the canonical config (config.yaml does not exist)
_CONFIG_PATH = Path(__file__).resolve().parents[4] / "ex/synth/elbo/config1.yaml"


def _gauss_logpdf(x: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """batched MVN log-density logN(x; mu, cov). symmetrizes + jitters cov so the
    float32 round-trip from h5 cannot break the cholesky (exact symmetry / PD)."""
    d = cov.shape[-1]
    cov = 0.5 * (cov + cov.transpose(-1, -2)) + 1e-5 * torch.eye(
        d, device=cov.device, dtype=cov.dtype
    )
    return torch.distributions.MultivariateNormal(
        mu, covariance_matrix=cov, validate_args=False
    ).log_prob(x)


class ELBOAdapter(ExperimentAdapter):
    """ELBO estimation experiment adapter.

    cell shape: 2-tuple (alpha_idx, flat_idx).
    alpha_idx ∈ {0, ..., A-1} where A = len(config["alphas"]).
    flat_idx ∈ {0, ..., P*B*D-1} where P = num_priors, B = len(design_eig_percentages),
      D = num_designs_per_setting.

    dataset layout (C-order, alpha innermost):
      row_idx = ((prior * B + beta) * D + design) * A + alpha
    equivalent to: row_idx = flat_idx * A + alpha_idx.

    h5 keys (dataset): theta0_samples_arr, y0_samples_arr, theta1_samples_arr,
      y1_samples_arr, theta_star_samples_arr, y_star_samples_arr.
      all indexed by row_idx = flat_idx * num_alphas + alpha_idx along axis=0.
    h5 keys (processed summary.h5): true_eldrs (flat array, indexed by row_idx).

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
        self._num_waypoints = cfg.get("num_waypoints", None)

        # sample sizes + the single resolved dataset filename. keeping one
        # resolved attribute (rather than the whole cfg) is what tests stub.
        self._n_p0p1 = cfg.get("n_p0p1")
        self._n_pstar = cfg.get("n_pstar")
        self._dataset_filename = cfg.get("dataset_filename") or (
            f"dataset_d={self._data_dim},"
            f"n_p0p1={self._n_p0p1},n_pstar={self._n_pstar}.h5"
        )

        # pool metadata
        self._num_alphas = len(cfg["alphas"])
        self._num_priors = cfg["num_priors"]
        self._design_eig_percentages = cfg["design_eig_percentages"]
        self._num_designs_per_setting = cfg["num_designs_per_setting"]

    def name(self) -> str:
        """return "elbo"."""
        return "elbo"

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
        """load one (alpha_idx, flat_idx) cell; return {p0, p1, pstar, true_ldrs}.

        args:
          cell: (alpha_idx, flat_idx). row_idx = flat_idx * num_alphas + alpha_idx
            indexes into all h5 arrays along axis=0.
          device: torch device string.

        opens {data_dir}/{dataset_filename}: theta0/y0/theta1/y1/theta_star/y_star
          + the per-cell Gaussian params (prior mu_pi/Sigma_pi, fractional-posterior
          mu_q/Sigma_q). theta+y concatenate to p0 (prior joint), p1 (q joint),
          pstar (q target).

        true_ldrs is PER-SAMPLE -- the analytic log density ratio predict_ldr
        targets, log p0(x) - log p1(x). the shared p(y|theta) noise cancels between
        the two joints, leaving the theta-marginal Gaussian log-ratio
          true_ldrs[i] = logN(theta*_i; mu_pi, Sigma_pi) - logN(theta*_i; mu_q, Sigma_q)
        at the pstar theta samples. this is the pointwise-MAE target and is
        self-contained (no dependency on step3's summary.h5).

        returns: {"pstar": (N, D+1), "p0": (N, D+1), "p1": (N, D+1),
                  "true_ldrs": (N,) tensor}.

        raises FileNotFoundError if h5 path missing.
        """
        alpha_idx, flat_idx = cell
        row_idx = flat_idx * self._num_alphas + alpha_idx
        dpath = self.data_dir() / self._dataset_filename

        with h5py.File(dpath, "r") as f:
            t0 = torch.from_numpy(np.array(f["theta0_samples_arr"][row_idx])).float().to(device)  # (N, D)
            y0 = torch.from_numpy(np.array(f["y0_samples_arr"][row_idx])).float().to(device)      # (N, 1)
            t1 = torch.from_numpy(np.array(f["theta1_samples_arr"][row_idx])).float().to(device)  # (N, D)
            y1 = torch.from_numpy(np.array(f["y1_samples_arr"][row_idx])).float().to(device)      # (N, 1)
            ts = torch.from_numpy(np.array(f["theta_star_samples_arr"][row_idx])).float().to(device)  # (N, D)
            ys = torch.from_numpy(np.array(f["y_star_samples_arr"][row_idx])).float().to(device)      # (N, 1)
            mu_pi = torch.from_numpy(np.array(f["prior_mean_arr"][row_idx])).float().to(device)         # (D,)
            S_pi  = torch.from_numpy(np.array(f["prior_covariance_arr"][row_idx])).float().to(device)   # (D, D)
            mu_q  = torch.from_numpy(np.array(f["mu_q_arr"][row_idx])).float().to(device)               # (D,)
            S_q   = torch.from_numpy(np.array(f["Sigma_q_arr"][row_idx])).float().to(device)            # (D, D)

        # per-sample true log-ratio = theta-marginal Gaussian log-ratio at pstar
        true_ldrs = _gauss_logpdf(ts, mu_pi, S_pi) - _gauss_logpdf(ts, mu_q, S_q)  # (N,)

        return {
            "p0": torch.cat([t0, y0], dim=-1),      # (N, D+1)
            "p1": torch.cat([t1, y1], dim=-1),      # (N, D+1)
            "pstar": torch.cat([ts, ys], dim=-1),   # (N, D+1)
            "true_ldrs": true_ldrs,                 # (N,) per-sample true log-ratio
        }

    def device(self) -> str:
        """config["device"], but fall back to cpu when no gpu is visible.

        cpu-lane (array) workers have no gpu, so the classifier methods routed
        there must run on cpu rather than faulting on .to("cuda"). gpu-lane
        workers still see cuda and use it.
        """
        if self._device == "cpu" or not torch.cuda.is_available():
            return "cpu"
        return self._device

    def latent_dim(self) -> int:
        """return data_dim + 1."""
        return self._latent_dim

    def num_waypoints(self) -> Optional[int]:
        """return config["num_waypoints"] or None."""
        return self._num_waypoints

    def metric_key(self) -> str:
        """return "per_cell_ldr_mae" -- pointwise LDR MAE like the other estimation
        experiments (switched from scalar per_cell_eldr_abs_err 2026-08-09)."""
        return "per_cell_ldr_mae"

    def stratify_key(self, cell: tuple[int, int]) -> tuple[int, int, int]:
        """return (alpha_idx, prior_idx, beta_idx) for fine-grained stratification.

        cell = (alpha_idx, flat_idx).
        flat_idx encodes (prior, beta, design) in C-order:
            flat_idx = prior * B * D + beta * D + design
        where B = len(design_eig_percentages), D = num_designs_per_setting.

        decompose to recover prior and beta; design is within-stratum axis.
        yields 128 strata (4 alpha × 8 prior × 4 beta), each with 16 designs.
        """
        alpha_idx, flat_idx = cell
        n_beta = len(self._design_eig_percentages)
        n_design = self._num_designs_per_setting

        # recover prior and beta from flat_idx
        prior_idx = flat_idx // (n_beta * n_design)
        beta_idx = (flat_idx // n_design) % n_beta

        return (alpha_idx, prior_idx, beta_idx)

    # -- split configuration (peak-campaign three-way split) -------------------
    # elbo under the 2/2/12 convention: 16 cells per (alpha, prior, beta) stratum,
    # 2 → train, 2 → holdout, 12 → step2 (disjoint remainder).
    # total: 128 strata * (2 + 2 + 12) = 256 optuna + 256 holdout + 1536 step2.

    def n_train_per_stratum(self) -> int:
        """2 cells per stratum routed to optuna training pool."""
        return 2

    def n_holdout_per_stratum(self) -> int:
        """2 cells per stratum routed to holdout pool."""
        return 2

    def n_step2_per_stratum(self) -> int:
        """sentinel: route ALL non-train/non-holdout cells to step2.

        since each stratum has exactly 16 cells and we allocate 2+2=4 to hpo,
        the remainder is 12, so step2 gets every remaining design. set to -1
        to invoke the sentinel behavior in stratified_split_3way (line 207–209).
        """
        return -1
