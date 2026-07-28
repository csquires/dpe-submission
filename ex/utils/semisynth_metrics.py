"""compute three metrics from gathered step2 estimates + per-cell ground truth.

metrics: eldr_abs_err (test vs train estimate means), mae_train (pointwise fit),
regret (cross-method normalized per cell). output h5 with per-method per-cell
1-D arrays; return methods ordered by ex.utils.faceted_lines.order_methods.
"""
import os

import h5py
import numpy as np
from ex.utils.faceted_lines import order_methods


def compute_metrics(gathered_h5: str, cell_data_paths: list[str], out_h5: str,
                    *, seed: int = 0, mae_gt_key: str = 'true_ldrs',
                    eldr_gt_key: str = 'samples_test_true_ldrs') -> list[str]:
    """read gathered est_ldrs_{m} (n_cells, n_train) + per-cell true_ldrs and
    samples_test_true_ldrs; write per-method per-cell arrays for the 3 metrics.

    cell_data_paths[i] is the data-file path for gathered cell i (SAME order as the
    gathered arrays). methods discovered by the `est_ldrs_` prefix. handles NaN cells
    (failed est) by excluding them from that method and from the regret best/worst at
    that cell. output h5: per method eldr_abs_err_{m}, mae_train_{m}, regret_{m}
    (each 1-D, len n_cells), attr `methods`. returns methods ordered by
    ex.utils.faceted_lines.order_methods.

    args:
        gathered_h5: path to gathered results h5 (contains est_ldrs_* datasets).
        cell_data_paths: list of per-cell data h5 file paths, len n_cells, same order
            as gathered arrays. each has true_ldrs (n_train,) and samples_test_true_ldrs (n_test,).
        out_h5: output path (written in write mode).
        seed: random seed (unused here, for consistency with append_test_set signature).
    returns:
        ordered_methods: list[str], methods sorted by ex.utils.faceted_lines.order_methods.
    """
    # step 1: load gathered estimates and discover methods
    est_dict = {}
    with h5py.File(gathered_h5, 'r') as f:
        for key in f.keys():
            if key.startswith('est_ldrs_'):
                method = key.replace('est_ldrs_', '')
                est_dict[method] = f[key][:].astype(np.float32)

    n_cells = est_dict[list(est_dict.keys())[0]].shape[0]
    for method in est_dict:
        assert est_dict[method].shape[0] == n_cells, \
            f'method {method} shape {est_dict[method].shape} does not match n_cells {n_cells}'

    # step 2: load per-cell ground truth. mae_gt aligns pointwise with the est row
    # (for the mae metric); eldr_gt is the ELDR-scalar target. guard missing
    # files/keys -> invalid cell (metrics NaN); some experiments have gaps.
    mae_gt_list = [None] * n_cells
    eldr_gt_list = [None] * n_cells
    valid = np.zeros(n_cells, dtype=bool)
    for i in range(n_cells):
        p = cell_data_paths[i]
        if not os.path.exists(p):
            continue
        try:
            with h5py.File(p, 'r') as f:
                if mae_gt_key not in f or eldr_gt_key not in f:
                    continue
                mae_gt_list[i] = f[mae_gt_key][:].astype(np.float32)
                eldr_gt_list[i] = f[eldr_gt_key][:].astype(np.float32)
                valid[i] = True
        except (OSError, KeyError):
            continue
    n_missing = int((~valid).sum())
    if n_missing:
        print(f'[semisynth_metrics] {n_missing}/{n_cells} cells missing/unreadable -> NaN')

    # step 3: eldr_abs_err = |mean(est) - mean(eldr_gt)| (invalid -> NaN).
    # est mean is over the full estimate row (all predictions for the cell).
    eldr_abs_err = {}
    for method, est in est_dict.items():
        m_arr = np.full(n_cells, np.nan, dtype=np.float32)
        for i in range(n_cells):
            if not valid[i]:
                continue
            m_arr[i] = np.abs(np.mean(est[i]) - np.mean(eldr_gt_list[i]))
        eldr_abs_err[method] = m_arr

    # step 4: mae = mean(|est - mae_gt|), est sliced to mae_gt length (must
    # correspond pointwise). invalid -> NaN.
    mae_train = {}
    for method, est in est_dict.items():
        m_arr = np.full(n_cells, np.nan, dtype=np.float32)
        for i in range(n_cells):
            if not valid[i]:
                continue
            L = len(mae_gt_list[i])
            m_arr[i] = np.mean(np.abs(est[i, :L] - mae_gt_list[i]))
        mae_train[method] = m_arr

    # step 5: compute regret (per-method, per-cell, cross-method normalized)
    methods_list = list(est_dict.keys())
    # (n_methods, n_cells), populated from eldr_abs_err dict
    err_array = np.stack([eldr_abs_err[m] for m in methods_list], axis=0).astype(np.float64)

    regret_array = np.zeros_like(err_array)
    for i in range(n_cells):
        err_at_cell = err_array[:, i]  # (n_methods,)
        finite = np.isfinite(err_at_cell)
        n_finite = finite.sum()

        if n_finite < 2:
            # < 2 finite methods: cannot normalize
            regret_array[:, i] = np.nan
        else:
            best = np.nanmin(err_at_cell)  # scalar, finite
            worst = np.nanmax(err_at_cell)  # scalar, finite
            span = worst - best

            if span == 0:
                # exact tie: 0 for finite methods, NaN for NaN methods
                regret_array[:, i] = np.where(finite, 0.0, np.nan)
            else:
                # standard regret: (err - best) / span, NaN for NaN methods
                regret_array[:, i] = np.where(finite, (err_at_cell - best) / span, np.nan)

    regret = {methods_list[m_idx]: regret_array[m_idx, :].astype(np.float32)
              for m_idx in range(len(methods_list))}

    # step 6: order methods
    ordered_methods = order_methods(methods_list)

    # step 7: write output h5
    with h5py.File(out_h5, 'w') as f:
        for method in ordered_methods:
            f.create_dataset(f'eldr_abs_err_{method}', data=eldr_abs_err[method], dtype=np.float32)
            f.create_dataset(f'mae_train_{method}', data=mae_train[method], dtype=np.float32)
            f.create_dataset(f'regret_{method}', data=regret[method], dtype=np.float32)
        f.attrs['methods'] = ordered_methods

    return ordered_methods
