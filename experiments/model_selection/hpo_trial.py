"""run a single hpo trial on model_selection gaussian ldr data.

cli:
    python -m experiments.model_selection.hpo_trial \
        --method <name> --config-file <trial.json> \
        [--eval-cells "0,10,20,30,40,50,60"] --output-dir <path>

eval cell shape: (row_idx,). default cells: seven rows distributed across the
full row range. ground truth LDR comes from HDF5 key 'true_ldrs_arr' at
test_set_idx=0. metric: mean absolute error (MAE) between predicted and true
LDR per row.
"""

import argparse
import json
import warnings

import h5py
import numpy as np
import torch

from src.utils.io import _load_config
from experiments.model_selection.hpo_search_spaces import SEARCH_SPACES
from experiments.utils.hpo.trial import parse_cells, run_trial


CONFIG_PATH = "experiments/model_selection/config.yaml"


def parse_args():
    """cli: --method, --config-file (trial json), [--eval-cells], --output-dir."""
    p = argparse.ArgumentParser(description="run single hpo trial on model_selection data")
    p.add_argument("--method", required=True, choices=list(SEARCH_SPACES.keys()))
    p.add_argument("--config-file", required=True, help="path to trial json")
    p.add_argument("--eval-cells", default="0,10,20,30,40,50,60",
                   help="comma-separated row indices")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    config = _load_config(CONFIG_PATH)

    with open(args.config_file) as f:
        trial = json.load(f)

    # resolve device with fallback
    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("cuda not available, falling back to cpu")
        device = "cpu"

    entry = SEARCH_SPACES[args.method]
    builder = entry["builder"]
    requires_pstar = entry["requires_pstar"]
    num_waypoints = entry.get("num_waypoints", 5)

    dataset_path = f"{config['data_dir']}/dataset_newpstar.h5"
    h5 = h5py.File(dataset_path, "r")
    try:
        def eval_cell(cell):
            # unpack single-element tuple
            (row_idx,) = cell

            # read from hdf5 (always use test_set_idx=0 for metric)
            # batch dimensions: p0 (ntrain, 3), p1 (ntrain, 3), pstar_test (ntest, 3), true_ldr (ntest,)
            p0_np = h5["samples_p0_arr"][row_idx]
            p1_np = h5["samples_p1_arr"][row_idx]
            pstar_test_np = h5["samples_pstar_arr"][row_idx, 0]
            true_ldr = h5["true_ldrs_arr"][row_idx, 0]

            # convert to torch on device
            p0 = torch.from_numpy(p0_np).float().to(device)  # (ntrain, 3)
            p1 = torch.from_numpy(p1_np).float().to(device)  # (ntrain, 3)
            pstar_test = torch.from_numpy(pstar_test_np).float().to(device)  # (ntest, 3)

            # build estimator
            est = builder(
                input_dim=3,
                device=device,
                num_waypoints=num_waypoints,
                **trial["hyperparams"]
            )

            # fit with or without pstar samples
            if requires_pstar:
                pstar_train_np = h5["samples_pstar_train_arr"][row_idx]
                pstar_train = torch.from_numpy(pstar_train_np).float().to(device)  # (ntrain, 3)
                est.fit(p0, p1, pstar_train)
            else:
                est.fit(p0, p1)

            # predict and compute metric
            est_ldr = est.predict_ldr(pstar_test)  # (ntest,) or float
            est_ldr_np = est_ldr.detach().cpu().numpy()
            mae = float(np.mean(np.abs(est_ldr_np - true_ldr)))
            return mae

        eval_cells = parse_cells(args.eval_cells)
        run_trial(
            experiment="model_selection",
            method=args.method,
            trial_id=trial["trial_id"],
            hyperparams=trial["hyperparams"],
            eval_cells=eval_cells,
            eval_cell=eval_cell,
            output_dir=args.output_dir,
            metric_key="per_row_ldr_mean_ae"
        )
    finally:
        h5.close()


if __name__ == "__main__":
    main()
