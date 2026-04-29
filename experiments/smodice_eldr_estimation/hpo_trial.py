"""run a single hpo trial on smodice gridworld eldr data.

cli:
    python -m experiments.smodice_eldr_estimation.hpo_trial \
        --method <name> --config-file <trial.json> \
        [--eval-cells <"k1:k2:seed,...">] --output-dir <path>

eval cell shape: (k1_idx, k2_idx, seed). default cells: full kl-grid at seed=0.

per-cell metric: mean(|predict_ldr(pstar) - true_ldrs|). encoding subdir
resolution and tabular-method kwargs injection are handled here so the registry
stays generic.
"""

import argparse
import json
import os

import h5py
import torch

from src.utils.io import _load_config
from src.sampling.frozen_flow import FrozenFlow
from experiments.smodice_eldr_estimation.hpo_search_spaces import SEARCH_SPACES
from experiments.utils.hpo.trial import parse_cells, run_trial


CONFIG_PATH = "experiments/smodice_eldr_estimation/config.yaml"


def parse_args():
    """cli: --method, --config-file (trial json), [--eval-cells], --output-dir."""
    p = argparse.ArgumentParser(description="run single hpo trial on smodice gridworld data")
    p.add_argument("--method", required=True, choices=list(SEARCH_SPACES.keys()))
    p.add_argument("--config-file", required=True, help="path to trial json")
    p.add_argument("--eval-cells", default=None,
                   help="comma-separated k1:k2:seed triplets; default = full kl-grid at seed=0")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def derive_input_dim(encoding_cfg: dict, n_states: int, n_actions: int) -> int:
    """map encoding type -> dnn input dim."""
    t = encoding_cfg["type"]
    if t == "onehot_joint":
        return n_states * n_actions
    if t == "onehot_concat":
        return n_states + n_actions
    if t in ("gaussian_blob", "flow_pushforward"):
        return encoding_cfg.get("embed_dim", 6)
    raise ValueError(f"unknown encoding type: {t}")


def encoding_subdir(encoding_cfg: dict, base: str) -> str:
    """resolve <base>/<type>/sigma_{na|XXX} subdir for data path construction."""
    t = encoding_cfg["type"]
    if t.startswith("onehot"):
        return os.path.join(base, t, "sigma_na")
    return os.path.join(base, t, f"sigma_{encoding_cfg['sigma']:.3f}")


def main():
    args = parse_args()
    config = _load_config(CONFIG_PATH)
    with open(args.config_file) as f:
        trial = json.load(f)

    device = torch.device(config["device"])
    num_waypoints = config["num_waypoints"]
    L = config["gridworld"]["L"]
    n_states, n_actions = L * L, 4

    encoding_cfg = dict(config["encoding"])
    encoding_cfg.update(n_states=n_states, n_actions=n_actions, L=L)
    if encoding_cfg["type"] == "flow_pushforward":
        flow_sub = encoding_cfg.get("flow", {})
        encoding_cfg["flow_module"] = FrozenFlow(
            dim=encoding_cfg["embed_dim"],
            n_layers=flow_sub.get("layers", 4),
            seed=flow_sub.get("seed", config["seed"]),
        )

    encoding_type = encoding_cfg["type"]
    subdir = encoding_subdir(encoding_cfg, config["data_dir"])
    input_dim = derive_input_dim(encoding_cfg, n_states, n_actions)

    entry = SEARCH_SPACES[args.method]
    if encoding_type not in entry["encoding_compat"]:
        raise ValueError(
            f"method {args.method} does not support encoding {encoding_type}; "
            f"supported: {entry['encoding_compat']}"
        )
    builder = entry["builder"]
    requires_pstar = entry["requires_pstar"]
    needs_latent = entry.get("needs_latent", False)
    is_tabular = args.method in ("TabularPluginDRE", "SmoothedTabularPluginDRE")

    if args.eval_cells is None:
        n_k1 = len(config["kl_targets"]["k1_values"])
        n_k2 = len(config["kl_targets"]["k2_values"])
        eval_cells = [(k1, k2, 0) for k1 in range(n_k1) for k2 in range(n_k2)]
    else:
        eval_cells = parse_cells(args.eval_cells)

    # step3 convention: discrete ground truth for onehot encodings, smoothed otherwise.
    true_ldrs_key = "true_ldrs_discrete" if encoding_type.startswith("onehot") else "true_ldrs_smoothed"

    def eval_cell(cell):
        k1, k2, seed = cell
        path = os.path.join(subdir, f"kl1_{k1}_kl2_{k2}_seed_{seed}.h5")
        with h5py.File(path, "r") as f:
            p0 = torch.from_numpy(f["p0_samples"][()]).float().to(device)
            p1 = torch.from_numpy(f["p1_samples"][()]).float().to(device)
            pstar = torch.from_numpy(f["pstar_samples"][()]).float().to(device)
            true_ldrs = torch.from_numpy(f[true_ldrs_key][()]).float().to(device)
            p0_lat = torch.from_numpy(f["p0_latent"][()]).long().to(device) if needs_latent else None
            p1_lat = torch.from_numpy(f["p1_latent"][()]).long().to(device) if needs_latent else None

        flat_hp = dict(trial["hyperparams"])
        if is_tabular:
            flat_hp["n_states"] = n_states
            flat_hp["n_actions"] = n_actions
            flat_hp["encoding_cfg"] = encoding_cfg
            if args.method == "TabularPluginDRE":
                flat_hp["decode"] = "argmax" if encoding_type.startswith("onehot") else "nn"

        est = builder(input_dim=input_dim, device=device,
                      num_waypoints=num_waypoints, **flat_hp)
        if needs_latent:
            est.fit(p0, p1, latent_p0=p0_lat, latent_p1=p1_lat)
        elif requires_pstar:
            est.fit(p0, p1, pstar)
        else:
            est.fit(p0, p1)
        est_ldrs = est.predict_ldr(pstar)
        return torch.mean(torch.abs(est_ldrs.cpu() - true_ldrs.cpu())).item()

    run_trial(
        experiment="smodice",
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
