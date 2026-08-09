"""
Step 3: Process Results for ELBO Estimation

Computes ground-truth ELDR values and aggregates per-method MAE estimates.

**Design:** Graceful skip on missing gathered file.
- Always: compute true_eldrs from dataset, write to summary.h5.
- Conditionally: if gathered file exists, aggregate method estimates.
- If gathered missing: log info, exit cleanly (HPO can proceed with true_eldrs only).

Run datagen first:
    DPE_DATA_ROOT=/data/... python -u ex/synth/elbo/step1_create_data.py

Then run this script to compute true_eldrs:
    DPE_DATA_ROOT=/data/... python -u ex/synth/elbo/step3_process_results.py

After step2 + gather, re-run this script to add method grids:
    python -m ex.utils.step2_runner.gather \\
        --experiment elbo \\
        --config ex/synth/elbo/config1.yaml
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
from ex.synth.elbo.step2_adapter import gather_output_path


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


def mae_cols(errors_4d, cols):
    """MAE stats per (design-eig-percentage, alpha column group).

    Args:
      errors_4d: (P, B, D, A) absolute ELDR errors, NaN where a cell failed.
      cols: list of alpha-index lists, one per output column. [[a] for a in
        range(A)] reproduces the per-alpha grid; [list(range(A))] pools every
        alpha into a single column.
    Returns:
      dict of mean/se/med/q1/q3/n grids, each (B, len(cols)).
    """
    n_deps, n_cols = errors_4d.shape[1], len(cols)
    out = {k: np.full((n_deps, n_cols), np.nan, dtype=np.float32)
           for k in ("mean", "se", "med", "q1", "q3")}
    out["n"] = np.zeros((n_deps, n_cols), dtype=np.int32)
    for bi in range(n_deps):
        for ci, ais in enumerate(cols):
            vals = errors_4d[:, bi][:, :, ais].flatten()   # (P, D, |ais|) -> flat
            mu, se, n = agg_metric(vals)
            out["mean"][bi, ci], out["se"][bi, ci], out["n"][bi, ci] = mu, se, n
            valid = vals[~np.isnan(vals)]
            if valid.size:
                out["q1"][bi, ci], out["med"][bi, ci], out["q3"][bi, ci] = \
                    np.percentile(valid, [25, 50, 75])
    return out


def regret_cols(reg_m, cols, bp, bd):
    """Regret median-of-medians + bootstrap band per (dep, alpha column group).

    Args:
      reg_m: (P, B, D, A) per-cell normalized regret for one method.
      cols: alpha-index lists, as in mae_cols.
      bp, bd: (n_boot, P) / (n_boot, D) resample index matrices.
    Returns:
      dict of mom/lo/hi/bstd grids, each (B, len(cols)).

    point estimate: median over priors, then median over the column's pooled
    (design, alpha) axis. the bootstrap resamples priors and designs only;
    alpha is a fixed 4-level design factor, so each resample keeps every alpha
    level the column covers.
    """
    n_deps, n_cols, n_boot = reg_m.shape[1], len(cols), bp.shape[0]
    out = {k: np.full((n_deps, n_cols), np.nan, dtype=np.float32)
           for k in ("mom", "lo", "hi", "bstd")}
    for bi in range(n_deps):
        for ci, ais in enumerate(cols):
            mat = reg_m[:, bi][:, :, ais]                  # (P, D, |ais|)
            if not np.isfinite(mat).any():
                continue
            out["mom"][bi, ci] = np.nanmedian(np.nanmedian(mat, axis=0))
            res = mat[bp[:, :, None], bd[:, None, :]]      # (n_boot, P, D, |ais|)
            bmom = np.nanmedian(np.nanmedian(res, axis=1).reshape(n_boot, -1), axis=1)
            bmom = bmom[np.isfinite(bmom)]
            if bmom.size:
                out["lo"][bi, ci] = np.percentile(bmom, 25)
                out["hi"][bi, ci] = np.percentile(bmom, 75)
            if bmom.size >= 2:
                out["bstd"][bi, ci] = bmom.std(ddof=1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",  default="ex/synth/elbo/config1.yaml")
    parser.add_argument("--winners", default="scratch/gold_winners/winners.elbo.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = _load_config(args.config)

    # --- grid dimensions (derived from config contract) ---
    alphas    = config["alphas"]
    deps      = config["design_eig_percentages"]
    n_alphas  = len(alphas)
    n_deps    = len(deps)
    n_priors  = config["num_priors"]
    n_designs = config["num_designs_per_setting"]

    # expected grid size (alpha innermost): prior × dep × design × alpha
    n_cells_expected = n_priors * n_deps * n_designs * n_alphas

    print(f"Grid structure: {n_priors} priors × {n_deps} design-eig-percentages × {n_designs} designs × {n_alphas} alphas = {n_cells_expected} cells")

    # --- compute true ELDRs ---
    data_dir  = config["data_dir"]
    # dataset filename contract: config["dataset_filename"] with split sample sizes.
    # fallback for backwards compat: old format with single nsamples.
    ds_file = config.get("dataset_filename")
    if ds_file is None:
        # backwards compat: old format with single nsamples
        ds_file = f"dataset_d={config['data_dim']},nsamples={config.get('nsamples', 50000)}.h5"
    ds_path = os.path.join(data_dir, ds_file)
    print(f"Loading true ELDRs from: {ds_path}")
    true_eldrs = np.zeros(n_cells_expected, dtype=np.float32)
    with h5py.File(ds_path, "r") as f:
        n_cells_actual = f["design_arr"].shape[0]
    assert n_cells_expected == n_cells_actual, \
        f"Grid mismatch: config expects {n_cells_expected} cells, dataset has {n_cells_actual}"

    with h5py.File(ds_path, "r") as f:
        for idx in range(n_cells_expected):
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

    # --- load gathered estimates (graceful skip if missing) ---
    gathered = gather_output_path(config)
    est_by_method = {}
    missing_in_gather = []

    if os.path.exists(gathered):
        print(f"Reading gathered results: {gathered}")
        with h5py.File(gathered, "r") as f:
            for m in methods:
                key = f"est_eldrs_arr_{m}"
                if key not in f:
                    missing_in_gather.append(m)
                    continue
                arr = f[key][:]                               # (n_cells, 1) or (n_cells,)
                est_by_method[m] = arr.reshape(n_cells_expected) # always (n_cells,)
        if missing_in_gather:
            logging.warning(f"methods not in gathered file: {missing_in_gather}")
    else:
        logging.info(
            f"Gathered file not found: {gathered}\n"
            "Will compute true_eldrs only (before step2/gather). "
            "Re-run this script after gather to add method estimates."
        )
        missing_in_gather = methods  # mark all as missing

    # --- compute errors (conditional on gathered) ---
    # two column layouts over the alpha axis: one column per alpha (the
    # per-alpha grids) and one column pooling every alpha (the pooled variant).
    per_alpha = [[a] for a in range(n_alphas)]
    all_alpha = [list(range(n_alphas))]

    mae, mae_pool, seed_vals = {}, {}, {}
    for m in methods:
        if m not in est_by_method:
            continue
        est    = est_by_method[m]                   # (n_cells,), may have NaN
        errors = np.abs(est - true_eldrs)           # NaN where est is NaN
        # C-order reshape: prior is slowest, alpha is fastest
        errors_4d = errors.reshape(n_priors, n_deps, n_designs, n_alphas)
        mae[m]      = mae_cols(errors_4d, per_alpha)
        mae_pool[m] = mae_cols(errors_4d, all_alpha)

        all_vals = errors_4d.flatten()
        seed_vals[m] = all_vals[~np.isnan(all_vals)].astype(np.float32)

    # per-cell normalized regret across methods (eig-style: 0 = best method on a
    # cell, 1 = worst), aggregated per (dep, column) as median over priors then
    # designs (median-of-medians) with a bootstrap IQR band.
    reg_stats, reg_pool = {}, {}
    if est_by_method:
        algs = [m for m in methods if m in est_by_method]
        err5 = np.stack([
            np.abs(est_by_method[m] - true_eldrs).astype(np.float64).reshape(
                n_priors, n_deps, n_designs, n_alphas)
            for m in algs])                                    # (M, P, B, D, A)
        finite = np.isfinite(err5)
        best = np.where(finite, err5, np.inf).min(axis=0)
        worst = np.where(finite, err5, -np.inf).max(axis=0)
        span = worst - best
        reg = (err5 - best[None]) / np.where(span > 0, span, np.nan)[None]
        tied = (span == 0) & np.isfinite(best)
        reg = np.where(np.broadcast_to(tied[None], reg.shape) & finite, 0.0, reg)

        rng = np.random.default_rng(config["seed"])
        n_boot = 1000
        bp = rng.integers(0, n_priors, size=(n_boot, n_priors))
        bd = rng.integers(0, n_designs, size=(n_boot, n_designs))
        for mi, m in enumerate(algs):
            reg_stats[m] = regret_cols(reg[mi], per_alpha, bp, bd)
            reg_pool[m]  = regret_cols(reg[mi], all_alpha, bp, bd)

    # --- save summary.h5 ---
    # always write true_eldrs and axes (HPO dependency).
    # optionally write method grids (only if gathered exists).
    processed_dir = config["processed_results_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "summary.h5")
    with h5py.File(out_path, "w") as f:
        # always write these (contract for HPO)
        f.create_dataset("alphas", data=np.array(alphas, dtype=np.float32))
        f.create_dataset("design_eig_percentages", data=np.array(deps, dtype=np.float32))
        f.create_dataset("true_eldrs", data=true_eldrs)

        # optionally write method grids (only if gathered exists).
        # each stat comes in two shapes: (n_deps, n_alphas) per-alpha columns
        # and a (n_deps, 1) '_pooled_' variant aggregating across all alphas.
        if est_by_method:
            f.attrs["methods"] = methods
            for m in methods:
                if m not in est_by_method:
                    continue
                for k, v in mae[m].items():
                    f.create_dataset(f"mae_{m}_{k}", data=v)
                for k, v in mae_pool[m].items():
                    f.create_dataset(f"mae_{m}_pooled_{k}", data=v)
                f.create_dataset(f"mae_{m}_seed_values", data=seed_vals[m])
                for k, v in reg_stats.get(m, {}).items():
                    f.create_dataset(f"regret_{m}_{k}", data=v)
                for k, v in reg_pool.get(m, {}).items():
                    f.create_dataset(f"regret_{m}_pooled_{k}", data=v)
        else:
            f.attrs["methods"] = []  # mark that no methods are present yet

    # --- print summary (conditional on gathered) ---
    if est_by_method:
        # print method performance table
        col_w = 24
        print(f"\n{'='*130}")
        print("ELBO Estimation — MAE  (mean ± se, n cells)")
        print(f"{'='*130}")
        hdr = "Method".ljust(32)
        for dep_idx in range(n_deps):
            for alpha_idx in range(n_alphas):
                hdr += f"dep={deps[dep_idx]:.3f},a={alphas[alpha_idx]}".rjust(col_w)
        print(hdr)
        print("-" * 130)
        for m in methods:
            if m not in mae:
                continue
            row = m.ljust(32)
            for dep_idx in range(n_deps):
                for alpha_idx in range(n_alphas):
                    n = int(mae[m]["n"][dep_idx, alpha_idx])
                    if n == 0:
                        cell = "NaN(0)"
                    else:
                        cell = f"{mae[m]['mean'][dep_idx,alpha_idx]:.4f}±{mae[m]['se'][dep_idx,alpha_idx]:.4f}({n})"
                    row += cell.rjust(col_w)
            print(row)
        print(f"{'='*130}")

        # alpha-pooled companion: same rows, alphas aggregated into one column
        print(f"\n{'='*130}")
        print("ELBO Estimation — MAE, alphas pooled  (mean ± se, n cells)")
        print(f"{'='*130}")
        hdr = "Method".ljust(32) + "".join(f"dep={d:.3f}".rjust(col_w) for d in deps)
        print(hdr)
        print("-" * 130)
        for m in methods:
            if m not in mae_pool:
                continue
            row = m.ljust(32)
            for dep_idx in range(n_deps):
                n = int(mae_pool[m]["n"][dep_idx, 0])
                cell = "NaN(0)" if n == 0 else \
                    f"{mae_pool[m]['mean'][dep_idx,0]:.4f}±{mae_pool[m]['se'][dep_idx,0]:.4f}({n})"
                row += cell.rjust(col_w)
            print(row)
        print(f"{'='*130}")

        # completion stats
        valid_methods = [m for m in methods if m in est_by_method]
        total_cells = n_deps * n_alphas
        covered = sum(
            1 for m in valid_methods
            for dep_idx in range(n_deps)
            for alpha_idx in range(n_alphas)
            if mae[m]["n"][dep_idx, alpha_idx] > 0
        )
        print(f"\nSaved: {out_path}")
        print(f"Grid: {n_deps} design-eig-percentages × {n_alphas} alphas, {n_priors*n_designs} prior-design pairs")
        print(f"Methods in output: {len(valid_methods)}/{len(methods)}")
        print(f"(method, cell) pairs with ≥1 valid result: {covered}/{len(valid_methods)*total_cells}")
    else:
        # true_eldrs-only case
        print(f"\nSaved: {out_path}")
        print(f"True ELDRs range: [{true_eldrs.min():.4f}, {true_eldrs.max():.4f}]")
        print(f"Grid: {n_deps} design-eig-percentages × {n_alphas} alphas × {n_priors * n_designs} prior-design pairs")
        print("Method estimates: none yet (gathered file not found). Re-run after gather for full report.")


if __name__ == "__main__":
    main()
