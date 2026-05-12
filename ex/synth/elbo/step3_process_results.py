"""
Step 3: Process Results for ELBO Estimation

Reads the gathered results file and computes per-(alpha, design_eig_percentage)
summaries of ELBO estimation MAE for each method.

Run gather first:
    python -m ex.utils.step2_runner.gather \
        --experiment elbo \
        --config ex/synth/elbo/config1.yaml
Then run this script:
    DPE_DATA_ROOT=/data/... python -u ex/synth/elbo/step3_process_results.py
"""
import argparse
import logging
import os

import h5py
import numpy as np
import torch
import yaml

from src.utils.io import _load_config
from ex.synth.elbo.step2_adapter import gather_output_path, list_cells


def compute_true_eldr(mu_pi, Sigma_pi, mu_q, Sigma_q, xi, obs_y) -> float:
    """Analytic ELDR = E_q[log p0(θ,y)/p1(θ,y)] for Gaussian case."""
    d = mu_pi.shape[0]
    Sigma_pi_inv = torch.linalg.inv(Sigma_pi)

    log_det_pi = torch.linalg.slogdet(Sigma_pi)[1]
    diff_pi = mu_q - mu_pi
    E_log_p_theta = (
        -0.5 * d * np.log(2 * np.pi)
        - 0.5 * log_det_pi
        - 0.5 * torch.trace(Sigma_pi_inv @ Sigma_q)
        - 0.5 * diff_pi @ Sigma_pi_inv @ diff_pi
    )

    log_det_q = torch.linalg.slogdet(Sigma_q)[1]
    E_log_q_theta = (
        -0.5 * d * np.log(2 * np.pi)
        - 0.5 * log_det_q
        - 0.5 * d
    )

    xi_flat = xi.squeeze()
    obs_y_flat = obs_y.squeeze()
    pred_mean = xi_flat @ mu_q
    pred_var  = xi_flat @ Sigma_q @ xi_flat
    E_log_p_y = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * ((obs_y_flat - pred_mean) ** 2 + pred_var)
    )

    prior_pred_mean = xi_flat @ mu_pi
    prior_pred_var  = xi_flat @ Sigma_pi @ xi_flat + 1.0
    log_p_y_xi = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * torch.log(prior_pred_var)
        - 0.5 * (obs_y_flat - prior_pred_mean) ** 2 / prior_pred_var
    )

    return (E_log_p_theta + E_log_p_y - E_log_q_theta - log_p_y_xi).item()


def agg_metric(vals):
    """Return (mean, se, n) for a 1-D array of floats, ignoring NaN."""
    valid = vals[~np.isnan(vals)]
    n = len(valid)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(valid))
    se = float(np.std(valid, ddof=1) / np.sqrt(n)) if n >= 2 else np.nan
    return mean, se, n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",  default="ex/synth/elbo/config1.yaml")
    parser.add_argument("--winners", default="scratch/gold_winners/winners.elbo.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    config = _load_config(args.config)

    # --- grid dimensions ---
    n_priors  = config["num_priors"]                    # 5
    alphas    = config["alphas"]                        # [0, 0.5, 0.75, 1.0]
    deps      = config["design_eig_percentages"]        # [0.5, 0.6, ..., 0.999]
    n_designs = config["num_designs_per_setting"]       # 10
    n_alpha   = len(alphas)
    n_dep     = len(deps)
    n_cells   = len(list_cells(config))                 # 1200

    # verify ordering matches expected C-order (alpha fastest)
    assert n_cells == n_priors * n_dep * n_designs * n_alpha, \
        f"cell count mismatch: {n_cells} != {n_priors}*{n_dep}*{n_designs}*{n_alpha}"

    # --- compute true ELDRs ---
    data_dir  = config["data_dir"]
    ds_file   = config.get(
        "dataset_filename",
        f"dataset_d={config['data_dim']},nsamples={config['nsamples']}.h5"
    )
    ds_path   = os.path.join(data_dir, ds_file)
    print(f"Loading true ELDRs from: {ds_path}")
    true_eldrs = np.zeros(n_cells, dtype=np.float32)
    with h5py.File(ds_path, "r") as f:
        for idx in range(n_cells):
            mu_pi    = torch.from_numpy(f["prior_mean_arr"][idx])
            Sigma_pi = torch.from_numpy(f["prior_covariance_arr"][idx])
            mu_q     = torch.from_numpy(f["mu_q_arr"][idx])
            Sigma_q  = torch.from_numpy(f["Sigma_q_arr"][idx])
            xi       = torch.from_numpy(f["design_arr"][idx])
            obs_y    = torch.from_numpy(f["obs_y_arr"][idx])
            true_eldrs[idx] = compute_true_eldr(mu_pi, Sigma_pi, mu_q, Sigma_q, xi, obs_y)
    print(f"  true ELDRs range: [{true_eldrs.min():.4f}, {true_eldrs.max():.4f}]")

    # --- load methods from winners ---
    with open(args.winners) as f:
        winners = yaml.safe_load(f)
    methods = sorted(winners["methods"].keys())
    print(f"Methods ({len(methods)}): {methods}")

    # --- load gathered estimates ---
    gathered = gather_output_path(config)
    if not os.path.exists(gathered):
        raise FileNotFoundError(
            f"Gathered file not found: {gathered}\n"
            "Run gather first:\n"
            "  python -m ex.utils.step2_runner.gather \\\n"
            "      --experiment elbo \\\n"
            "      --config ex/synth/elbo/config1.yaml"
        )
    print(f"Reading gathered results: {gathered}")
    est_by_method = {}
    missing_in_gather = []
    with h5py.File(gathered, "r") as f:
        for m in methods:
            key = f"est_eldrs_arr_{m}"
            if key not in f:
                missing_in_gather.append(m)
                continue
            arr = f[key][:]                         # (n_cells, 1) or (n_cells,)
            est_by_method[m] = arr.reshape(n_cells) # always (n_cells,)
    if missing_in_gather:
        logging.warning(f"methods not in gathered file: {missing_in_gather}")

    # --- compute errors ---
    def _empty_grid():
        return np.full((n_dep, n_alpha), np.nan, dtype=np.float32)

    mean_mae = {m: _empty_grid() for m in methods}
    se_mae   = {m: _empty_grid() for m in methods}
    n_mae    = {m: np.zeros((n_dep, n_alpha), dtype=np.int32) for m in methods}
    seed_vals= {}

    for m in methods:
        if m not in est_by_method:
            continue
        est    = est_by_method[m]                   # (n_cells,), may have NaN
        errors = np.abs(est - true_eldrs)           # NaN where est is NaN
        # C-order reshape: prior is slowest, alpha is fastest
        errors_4d = errors.reshape(n_priors, n_dep, n_designs, n_alpha)

        for dep_idx in range(n_dep):
            for alpha_idx in range(n_alpha):
                vals = errors_4d[:, dep_idx, :, alpha_idx].flatten()
                mu, se, n = agg_metric(vals)
                mean_mae[m][dep_idx, alpha_idx] = mu
                se_mae[m][dep_idx, alpha_idx]   = se
                n_mae[m][dep_idx, alpha_idx]    = n

        all_vals = errors_4d.flatten()
        seed_vals[m] = all_vals[~np.isnan(all_vals)].astype(np.float32)

    # --- save summary ---
    processed_dir = config["processed_results_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "summary.h5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("alphas", data=np.array(alphas, dtype=np.float32))
        f.create_dataset("design_eig_percentages", data=np.array(deps, dtype=np.float32))
        f.create_dataset("true_eldrs", data=true_eldrs)
        f.attrs["methods"] = methods
        for m in methods:
            if m not in est_by_method:
                continue
            f.create_dataset(f"mae_{m}_mean",        data=mean_mae[m])
            f.create_dataset(f"mae_{m}_se",          data=se_mae[m])
            f.create_dataset(f"mae_{m}_n",           data=n_mae[m])
            f.create_dataset(f"mae_{m}_seed_values", data=seed_vals[m])

    # --- print summary table ---
    col_w = 24
    print(f"\n{'='*130}")
    print("ELBO Estimation — MAE  (mean ± se, n cells)")
    print(f"{'='*130}")
    hdr = "Method".ljust(32)
    for dep_idx in range(n_dep):
        for alpha_idx in range(n_alpha):
            hdr += f"dep={deps[dep_idx]:.3f},a={alphas[alpha_idx]}".rjust(col_w)
    print(hdr)
    print("-" * 130)
    for m in methods:
        row = m.ljust(32)
        for dep_idx in range(n_dep):
            for alpha_idx in range(n_alpha):
                n = int(n_mae[m][dep_idx, alpha_idx])
                if n == 0:
                    cell = "NaN(0)"
                else:
                    cell = f"{mean_mae[m][dep_idx,alpha_idx]:.4f}±{se_mae[m][dep_idx,alpha_idx]:.4f}({n})"
                row += cell.rjust(col_w)
        print(row)
    print(f"{'='*130}")

    # completion stats
    valid_methods = [m for m in methods if m in est_by_method]
    total_cells = n_dep * n_alpha
    covered = sum(
        1 for m in valid_methods
        for dep_idx in range(n_dep)
        for alpha_idx in range(n_alpha)
        if n_mae[m][dep_idx, alpha_idx] > 0
    )
    print(f"\nSaved: {out_path}")
    print(f"Grid: {n_dep} deps × {n_alpha} alphas, {n_priors*n_designs} samples/cell")
    print(f"Methods in output: {len(valid_methods)}/{len(methods)}")
    print(f"(method, cell) pairs with ≥1 valid result: {covered}/{len(valid_methods)*total_cells}")


if __name__ == "__main__":
    main()
