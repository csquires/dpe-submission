"""run a single hpo trial on pendulum eldr data.

cli:
    python -m experiments.pendulum_eldr_estimation.hpo_trial \
        --method <name> --config-file <trial.json> \
        [--eval-cells <"k1:k2:seed,...">] --output-dir <path>

eval cell shape: (k1_idx, k2_idx, seed). default cells: full kl-grid at seed=0.

raises ValueError at startup if kl_targets.k1_values or k2_values are empty.
no encoding axis (pendulum is continuous-state).
"""

import argparse
import json

import h5py
import torch

from src.utils.io import _load_config
from experiments.pendulum_eldr_estimation.hpo_search_spaces import SEARCH_SPACES
from experiments.utils.hpo.trial import parse_cells, run_trial


CONFIG_PATH = "experiments/pendulum_eldr_estimation/config.yaml"


def parse_args():
    """cli: --method, --config-file (trial json), [--eval-cells], --output-dir."""
    p = argparse.ArgumentParser(description="run single hpo trial on pendulum data")
    p.add_argument("--method", required=True, choices=list(SEARCH_SPACES.keys()))
    p.add_argument("--config-file", required=True, help="path to trial json")
    p.add_argument("--eval-cells", default=None,
                   help="comma-separated k1:k2:seed triplets; default = full kl-grid at seed=0")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    config = _load_config(CONFIG_PATH)
    with open(args.config_file) as f:
        trial = json.load(f)

    device = config["device"]
    num_waypoints = config["num_waypoints"]
    data_dir = config["data_dir"]

    k1_values = config["kl_targets"].get("k1_values", [])
    k2_values = config["kl_targets"].get("k2_values", [])
    if not k1_values or not k2_values:
        raise ValueError(
            "kl_targets.k1_values and k2_values are empty in config.yaml. "
            "pendulum HPO cannot run until you populate these "
            "(see kl_targets section in config; populate after running step1+grid build)."
        )

    if args.eval_cells is None:
        eval_cells = [
            (k1, k2, 0)
            for k1 in range(len(k1_values))
            for k2 in range(len(k2_values))
        ]
    else:
        eval_cells = parse_cells(args.eval_cells)

    entry = SEARCH_SPACES[args.method]
    builder = entry["builder"]
    requires_pstar = entry["requires_pstar"]

    def eval_cell(cell):
        k1, k2, seed = cell
        path = f"{data_dir}/kl1_{k1}_kl2_{k2}_seed_{seed}.h5"
        with h5py.File(path, "r") as f:
            p0 = torch.from_numpy(f["samples_p0"][()]).to(device)
            p1 = torch.from_numpy(f["samples_p1"][()]).to(device)
            pstar = torch.from_numpy(f["samples_pstar"][()]).to(device)
            true_ldrs = torch.from_numpy(f["true_ldrs"][()]).to(device)

        input_dim = pstar.shape[-1]
        est = builder(input_dim=input_dim, device=device,
                      num_waypoints=num_waypoints, **trial["hyperparams"])
        if requires_pstar:
            est.fit(p0, p1, pstar)
        else:
            est.fit(p0, p1)
        return torch.mean(torch.abs(est.predict_ldr(pstar).cpu() - true_ldrs.cpu())).item()

    run_trial(
        experiment="pendulum",
        method=args.method,
        trial_id=trial["trial_id"],
        hyperparams=trial["hyperparams"],
        eval_cells=eval_cells,
        eval_cell=eval_cell,
        output_dir=args.output_dir,
        metric_key="per_cell_ldr_mae",
    )


if __name__ == "__main__":
    main()
