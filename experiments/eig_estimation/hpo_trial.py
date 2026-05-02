"""run a single hpo trial for one method on eig data.

cli:
    python -m experiments.eig_estimation.hpo_trial \
        --method <name> --config-file <trial.json> \
        [--eval-designs <"i,j,k,...">] --output-dir <path>

eval cell shape: (design_idx,) — single int per cell.
default eval cells: one design per design_eig_percentage band.

per-method registry entry returns a bare DRE estimator; this harness wraps it
with EIGPlugin (and a triangular-fit adapter where needed) before computing
|estimate_eig - true_eig|.
"""

import argparse
import json
import os

import h5py
import torch

from src.utils.io import _load_config
from src.eig_estimation.plugin import EIGPlugin
from experiments.eig_estimation.hpo_search_spaces import SEARCH_SPACES
from experiments.utils.hpo.trial import parse_cells, run_trial


CONFIG_PATH = "experiments/eig_estimation/config1.yaml"


class TriangularEIGAdapter:
    """re-routes fit(p0, p1) to inner.fit(p0, p1, p0).

    triangular methods require fit(p0, p1, pstar) but EIGPlugin only supplies
    (p0, p1). using p0 as a stand-in pstar matches the inline adapter pattern
    in step2_run_algorithms.py.
    """
    def __init__(self, inner):
        self.inner = inner

    def fit(self, p0, p1):
        self.inner.fit(p0, p1, p0)

    def predict_ldr(self, xs):
        return self.inner.predict_ldr(xs)


def compute_true_eig(Sigma_pi: torch.Tensor, xi: torch.Tensor, sigma2: float = 1.0) -> torch.Tensor:
    """analytic EIG for gaussian linear regression: 0.5 * log1p(xi.T @ Sigma_pi @ xi / sigma2)."""
    quad = (xi.T @ Sigma_pi @ xi).squeeze()
    return 0.5 * torch.log1p(quad / sigma2)


def parse_args():
    """cli: --method, --config-file (trial json), [--eval-designs], --output-dir."""
    p = argparse.ArgumentParser(description="run single hpo trial on eig data")
    p.add_argument("--method", required=True, choices=list(SEARCH_SPACES.keys()))
    p.add_argument("--config-file", required=True, help="path to trial json")
    p.add_argument("--eval-designs", default=None,
                   help="comma-separated design indices; default = one per design_eig_percentage")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def default_eval_cells(dataset_h5, percentages: list[float]) -> list[tuple[int]]:
    """one design index per percentage band, mapped onto [0, nrows)."""
    nrows = dataset_h5["design_arr"].shape[0]
    return [(min(int(p * nrows), nrows - 1),) for p in percentages]


def main():
    args = parse_args()
    config = _load_config(CONFIG_PATH)
    with open(args.config_file) as f:
        trial = json.load(f)

    data_dim = config["data_dim"]
    nsamples = config["nsamples"]
    device = config["device"]
    percentages = config.get("design_eig_percentages", [0.5, 0.6, 0.7, 0.8, 0.9, 0.999])

    dataset_path = f"{config['data_dir']}/dataset_d={data_dim},nsamples={nsamples}.h5"

    entry = SEARCH_SPACES[args.method]
    builder = entry["builder"]
    is_triangular = args.method.startswith("Triangular")

    h5 = h5py.File(dataset_path, "r")
    try:
        if args.eval_designs:
            eval_cells = parse_cells(args.eval_designs)
        else:
            eval_cells = default_eval_cells(h5, percentages)

        def eval_cell(cell):
            (idx,) = cell
            theta = torch.from_numpy(h5["theta_samples_arr"][idx]).to(device)
            y = torch.from_numpy(h5["y_samples_arr"][idx]).to(device)
            xi = torch.from_numpy(h5["design_arr"][idx]).to(device)
            Sigma_pi = torch.from_numpy(h5["prior_covariance_arr"][idx]).to(device)
            true_eig = compute_true_eig(Sigma_pi, xi).item()

            est = builder(input_dim=data_dim + 1, device=device, num_waypoints=0,
                          **{k: v for k, v in trial["hyperparams"].items() if k != 'num_waypoints'})
            if is_triangular:
                est = TriangularEIGAdapter(est)
            plugin = EIGPlugin(est)

            est_eig = plugin.estimate_eig(theta, y)
            est_eig = float(est_eig.item() if hasattr(est_eig, "item") else est_eig)
            return abs(est_eig - true_eig)

        run_trial(
            experiment="eig",
            method=args.method,
            trial_id=trial["trial_id"],
            hyperparams=trial["hyperparams"],
            eval_cells=eval_cells,
            eval_cell=eval_cell,
            output_dir=args.output_dir,
            metric_key="per_design_eig_abs_err",
        )
    finally:
        h5.close()


if __name__ == "__main__":
    main()
