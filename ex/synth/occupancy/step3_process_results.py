"""
step3_process_results.py

Read the gathered results_all_cells.h5 file and compute per-(k1,k2) summaries
of pointwise LDR MAE and integrated ELDR error for each method.

Methods are taken from the winners YAML (not config['algorithms'], which is stale).
Gather must be run first:
    python -m ex.utils.step2_runner.gather \
        --experiment smodice_eldr_estimation \
        --config ex/synth/occupancy/config.yaml
"""

import os
import argparse
import logging
import h5py
import numpy as np
import yaml


def agg_metric(vals):
    """Return (mean, se, n) for a list of floats, ignoring NaN."""
    valid = np.array([v for v in vals if not np.isnan(v)])
    if len(valid) == 0:
        return np.nan, np.nan, 0
    mean = np.mean(valid)
    n = len(valid)
    se = np.std(valid, ddof=1) / np.sqrt(n) if n >= 2 else np.nan
    return mean, se, n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ex/synth/occupancy/config.yaml")
    parser.add_argument(
        "--winners",
        default="scratch/gold_winners/winners.smodice_eldr_estimation.yaml",
        help="path to winners yaml (method list source)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # load config with ${DPE_DATA_ROOT} substitution
    from src.utils.io import _load_config
    config = _load_config(args.config)

    # install gridworld-derived fields needed by encoding helpers
    L = config["gridworld"]["L"]
    enc = dict(config["encoding"])
    enc["n_states"] = L * L
    enc["n_actions"] = 4
    enc["L"] = L
    config["encoding"] = enc

    # derive encoding subdir (mirrors step2_adapter._encoding_subdir)
    encoding_type = enc["type"]
    if encoding_type.startswith("onehot"):
        sigma_dir = "sigma_na"
    else:
        sigma_dir = f"sigma_{enc['sigma']:.3f}"
    encoding_subdir = os.path.join(encoding_type, sigma_dir)

    # paths
    data_dir = config["data_dir"]
    raw_results_dir = config["raw_results_dir"]
    processed_results_dir = config["processed_results_dir"]

    gather_path = os.path.join(raw_results_dir, encoding_subdir, "results_all_cells.h5")
    if not os.path.exists(gather_path):
        raise FileNotFoundError(
            f"Gather output not found: {gather_path}\n"
            "Run gather first:\n"
            "  python -m ex.utils.step2_runner.gather \\\n"
            "      --experiment smodice_eldr_estimation \\\n"
            "      --config ex/synth/occupancy/config.yaml"
        )

    # method list from winners yaml
    with open(args.winners, "r") as f:
        winners = yaml.safe_load(f)
    methods = sorted(winners["methods"].keys())
    print(f"Methods ({len(methods)}): {methods}")

    # grid dimensions
    kl_targets = config["kl_targets"]
    k1_values = np.array(kl_targets["k1_values"], dtype=np.float32)
    k2_values = np.array(kl_targets["k2_values"], dtype=np.float32)
    n_k1 = len(k1_values)
    n_k2 = len(k2_values)
    n_seeds = kl_targets["seeds_default"]
    n_cells = n_k1 * n_k2 * n_seeds  # 480

    def _decode_cell(flat_idx):
        seed = flat_idx % n_seeds
        rest = flat_idx // n_seeds
        k2_idx = rest % n_k2
        k1_idx = rest // n_k2
        return k1_idx, k2_idx, seed

    # results[method][k1_idx][k2_idx] = list of (pointwise_mae, eldr_err)
    results = {m: [[[] for _ in range(n_k2)] for _ in range(n_k1)] for m in methods}

    missing_data = []
    skipped_nan = 0
    missing_methods_in_gather = set()

    with h5py.File(gather_path, "r") as gf:
        available_keys = set(gf.keys())
        for flat_idx in range(n_cells):
            k1_idx, k2_idx, seed = _decode_cell(flat_idx)

            # load ground truth
            data_path = os.path.join(
                data_dir, encoding_subdir,
                f"kl1_{k1_idx}_kl2_{k2_idx}_seed_{seed}.h5",
            )
            if not os.path.exists(data_path):
                missing_data.append(data_path)
                continue
            with h5py.File(data_path, "r") as df:
                true_ldrs = df["true_ldrs_smoothed"][:]
                integrated_eldr = float(df.attrs["integrated_eldr"])

            for method in methods:
                key = f"est_ldrs_{method}"
                if key not in available_keys:
                    missing_methods_in_gather.add(method)
                    continue
                est_ldrs = gf[key][flat_idx]  # shape (num_samples,)
                if np.any(np.isnan(est_ldrs)):
                    skipped_nan += 1
                    continue
                pointwise_mae = float(np.mean(np.abs(est_ldrs - true_ldrs)))
                eldr_err = float(abs(np.mean(est_ldrs) - integrated_eldr))
                results[method][k1_idx][k2_idx].append((pointwise_mae, eldr_err))

    if missing_data:
        logging.warning(f"missing {len(missing_data)} data files (cells skipped)")
    if skipped_nan:
        logging.warning(f"skipped {skipped_nan} (method, cell) pairs with NaN est_ldrs")
    if missing_methods_in_gather:
        logging.warning(f"methods not found in gather file: {sorted(missing_methods_in_gather)}")

    # aggregate per (method, k1_idx, k2_idx) over seeds
    def empty_grid():
        return np.full((n_k1, n_k2), np.nan, dtype=np.float32)

    mean_mae  = {m: empty_grid() for m in methods}
    se_mae    = {m: empty_grid() for m in methods}
    n_mae     = {m: np.zeros((n_k1, n_k2), dtype=np.int32) for m in methods}
    mean_eldr = {m: empty_grid() for m in methods}
    se_eldr   = {m: empty_grid() for m in methods}
    n_eldr    = {m: np.zeros((n_k1, n_k2), dtype=np.int32) for m in methods}

    for m in methods:
        for k1_idx in range(n_k1):
            for k2_idx in range(n_k2):
                pairs = results[m][k1_idx][k2_idx]
                if not pairs:
                    continue
                mae_vals  = [p[0] for p in pairs]
                eldr_vals = [p[1] for p in pairs]
                mu, se, n = agg_metric(mae_vals)
                mean_mae[m][k1_idx, k2_idx] = mu
                se_mae[m][k1_idx, k2_idx]   = se
                n_mae[m][k1_idx, k2_idx]     = n
                mu, se, n = agg_metric(eldr_vals)
                mean_eldr[m][k1_idx, k2_idx] = mu
                se_eldr[m][k1_idx, k2_idx]   = se
                n_eldr[m][k1_idx, k2_idx]     = n

    # collect flattened per-seed arrays (all valid seeds across all cells)
    seed_mae  = {m: np.array([p[0] for k1 in range(n_k1) for k2 in range(n_k2)
                               for p in results[m][k1][k2]], dtype=np.float32)
                 for m in methods}
    seed_eldr = {m: np.array([p[1] for k1 in range(n_k1) for k2 in range(n_k2)
                               for p in results[m][k1][k2]], dtype=np.float32)
                 for m in methods}

    # write summary h5
    out_dir = os.path.join(processed_results_dir, encoding_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.h5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("k1_values", data=k1_values)
        f.create_dataset("k2_values", data=k2_values)
        f.attrs["methods"] = methods
        for m in methods:
            f.create_dataset(f"pointwise_mae_{m}_mean", data=mean_mae[m])
            f.create_dataset(f"pointwise_mae_{m}_se",   data=se_mae[m])
            f.create_dataset(f"pointwise_mae_{m}_n",    data=n_mae[m])
            f.create_dataset(f"eldr_err_{m}_mean",      data=mean_eldr[m])
            f.create_dataset(f"eldr_err_{m}_se",        data=se_eldr[m])
            f.create_dataset(f"eldr_err_{m}_n",         data=n_eldr[m])
            f.create_dataset(f"pointwise_mae_{m}_seed_values", data=seed_mae[m])
            f.create_dataset(f"eldr_err_{m}_seed_values",      data=seed_eldr[m])

    # print summary table (pointwise_mae)
    col_w = 26
    print("\n" + "=" * 130)
    print("ELDR Estimation — pointwise_mae  (mean ± se, n seeds)")
    print("=" * 130)
    hdr = "Method".ljust(32)
    for k1_idx in range(n_k1):
        for k2_idx in range(n_k2):
            hdr += f"K1={k1_values[k1_idx]:.1f},K2={k2_values[k2_idx]:.1f}".rjust(col_w)
    print(hdr)
    print("-" * 130)
    for m in methods:
        row = m.ljust(32)
        for k1_idx in range(n_k1):
            for k2_idx in range(n_k2):
                n = n_mae[m][k1_idx, k2_idx]
                if n == 0:
                    cell = "NaN(0)"
                else:
                    cell = f"{mean_mae[m][k1_idx,k2_idx]:.4f}±{se_mae[m][k1_idx,k2_idx]:.4f}({n})"
                row += cell.rjust(col_w)
        print(row)
    print("=" * 130)

    # print eldr_err summary
    print("\n" + "=" * 130)
    print("ELDR Estimation — eldr_err  (mean ± se, n seeds)")
    print("=" * 130)
    print(hdr)
    print("-" * 130)
    for m in methods:
        row = m.ljust(32)
        for k1_idx in range(n_k1):
            for k2_idx in range(n_k2):
                n = n_eldr[m][k1_idx, k2_idx]
                if n == 0:
                    cell = "NaN(0)"
                else:
                    cell = f"{mean_eldr[m][k1_idx,k2_idx]:.4f}±{se_eldr[m][k1_idx,k2_idx]:.4f}({n})"
                row += cell.rjust(col_w)
        print(row)
    print("=" * 130)

    print(f"\nSaved: {out_path}")
    print(f"Grid: {n_k1}×{n_k2}, {n_seeds} seeds/cell, {n_cells} total cells")
    feasible = sum(
        1 for k1 in range(n_k1) for k2 in range(n_k2)
        if any(n_mae[m][k1, k2] > 0 for m in methods)
    )
    print(f"Cells with ≥1 valid result: {feasible}/{n_k1 * n_k2}")


if __name__ == "__main__":
    main()
