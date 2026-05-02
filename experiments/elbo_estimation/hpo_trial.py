"""run a single hpo trial on elbo data.

cli:
    python -m experiments.elbo_estimation.hpo_trial \
        --method <name> --config-file <trial.json> \
        [--eval-cells <"alpha:flat,...">] --output-dir <path>

eval cell shape: (alpha_idx, flat_idx). default cells: 4 (one per alpha) at a
representative flat_idx in the middle of the prior/eig%/design grid.

ground truth ELDR comes from processed_results/errors_d=...,nsamples=....h5 key
'true_eldrs' (flat array of length 24000 per the dataset's reshape order
(NUM_PRIORS, len(DESIGN_EIG_PERCENTAGES), NUM_DESIGNS_PER_SETTING, len(ALPHAS))
with alphas innermost). metric: |mean(predict_ldr(pstar)) - true_eldr|.
"""

import argparse
import json

import h5py
import torch

from src.utils.io import _load_config
from experiments.elbo_estimation.hpo_search_spaces import SEARCH_SPACES
from experiments.utils.hpo.trial import parse_cells, run_trial


CONFIG_PATH = "experiments/elbo_estimation/config1.yaml"


def parse_args():
    """cli: --method, --config-file (trial json), [--eval-cells], --output-dir."""
    p = argparse.ArgumentParser(description="run single hpo trial on elbo data")
    p.add_argument("--method", required=True, choices=list(SEARCH_SPACES.keys()))
    p.add_argument("--config-file", required=True, help="path to trial json")
    p.add_argument("--eval-cells", default="0:1696,1:1697,2:1698,3:1699",
                   help="comma-separated alpha_idx:flat_idx pairs (one per alpha)")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    config = _load_config(CONFIG_PATH)
    with open(args.config_file) as f:
        trial = json.load(f)

    data_dim = config["data_dim"]
    nsamples = config["nsamples"]
    device = config["device"]

    dataset_path = f"{config['data_dir']}/dataset_d={data_dim},nsamples={nsamples}.h5"
    processed_path = (
        f"{config['processed_results_dir']}/errors_d={data_dim},nsamples={nsamples}.h5"
    )

    with h5py.File(processed_path, "r") as f:
        true_eldrs = f["true_eldrs"][()]

    entry = SEARCH_SPACES[args.method]
    builder = entry["builder"]
    requires_pstar = entry["requires_pstar"]
    num_waypoints = entry.get("num_waypoints", 5)

    eval_cells = parse_cells(args.eval_cells)

    h5 = h5py.File(dataset_path, "r")
    try:
        def eval_cell(cell):
            _alpha_idx, flat_idx = cell
            t0 = torch.from_numpy(h5["theta0_samples_arr"][flat_idx]).float().to(device)
            y0 = torch.from_numpy(h5["y0_samples_arr"][flat_idx]).float().to(device)
            t1 = torch.from_numpy(h5["theta1_samples_arr"][flat_idx]).float().to(device)
            y1 = torch.from_numpy(h5["y1_samples_arr"][flat_idx]).float().to(device)
            ts = torch.from_numpy(h5["theta_star_samples_arr"][flat_idx]).float().to(device)
            ys = torch.from_numpy(h5["y_star_samples_arr"][flat_idx]).float().to(device)
            p0 = torch.cat([t0, y0], dim=1)
            p1 = torch.cat([t1, y1], dim=1)
            pstar = torch.cat([ts, ys], dim=1)

            est = builder(input_dim=data_dim + 1, device=device,
                          num_waypoints=trial["hyperparams"].get('num_waypoints', num_waypoints), **{k: v for k, v in trial["hyperparams"].items() if k != 'num_waypoints'})
            if requires_pstar:
                est.fit(p0, p1, pstar)
            else:
                est.fit(p0, p1)
            est_eldr = float(torch.mean(est.predict_ldr(pstar)).item())
            return abs(est_eldr - float(true_eldrs[flat_idx]))

        run_trial(
            experiment="elbo",
            method=args.method,
            trial_id=trial["trial_id"],
            hyperparams=trial["hyperparams"],
            eval_cells=eval_cells,
            eval_cell=eval_cell,
            output_dir=args.output_dir,
            metric_key="per_cell_eldr_abs_err",
        )
    finally:
        h5.close()


if __name__ == "__main__":
    main()
