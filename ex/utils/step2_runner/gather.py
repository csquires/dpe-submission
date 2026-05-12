"""step2_runner gather: assemble per-cell h5 fragments into a unified results.h5.

per-cell fragments produced by worker.py live at:
    <DPE_DATA_ROOT>/<exp>/step2_results/<method>/cell_<i>.h5

each containing 'est_ldrs' dataset of shape (ntest_sets, nsamples_test) plus
attributes (hyperparams, bucket_id, ok, ...).

this gather step writes one unified file per experiment by stacking per-cell
arrays along axis 0:
    <experiments>/<exp>/raw_results/results.h5
        est_ldrs_arr_<method>: shape (n_cells, ntest_sets, nsamples_test)

matches the existing step2 output convention so step3/4 work unchanged.

usage:
    DPE_DATA_ROOT=... python -m ex.utils.step2_runner.gather \
        --experiment model_selection \
        [--method CTSM]    # default: all methods present
"""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import h5py
import numpy as np


def _data_root() -> Path:
    root = os.environ.get("DPE_DATA_ROOT")
    if not root:
        raise SystemExit("DPE_DATA_ROOT must be set")
    return Path(root)


def gather_method(exp: str, method: str, n_cells_total: int,
                  fragments_dir: Path) -> tuple[np.ndarray, dict]:
    """gather one method's fragments into an array of shape (n_cells, ...).

    cells without a fragment are filled with NaN; cells with ok=False are also NaN.
    returns (est_ldrs_arr, summary_dict).
    """
    method_dir = fragments_dir / method
    if not method_dir.exists():
        raise FileNotFoundError(f"no fragments at {method_dir}")
    summary = {"n_total": n_cells_total, "n_ok": 0, "n_failed": 0, "n_missing": 0}
    arr = None
    sample_shape = None
    for cell_idx in range(n_cells_total):
        p = method_dir / f"cell_{cell_idx}.h5"
        if not p.exists():
            summary["n_missing"] += 1
            continue
        with h5py.File(p, "r") as f:
            ok = f.attrs.get("ok", True)
            if not bool(ok):
                summary["n_failed"] += 1
                continue
            est = f["est_ldrs"][...]
        if sample_shape is None:
            sample_shape = est.shape
            arr = np.full((n_cells_total, *sample_shape), np.nan, dtype=np.float32)
        if arr is None or est.shape != sample_shape:
            print(f"  WARN: cell {cell_idx} shape {est.shape} != expected {sample_shape}; skipping")
            summary["n_failed"] += 1
            continue
        arr[cell_idx] = est.astype(np.float32)
        summary["n_ok"] += 1
    return arr, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--method", default=None,
                   help="single method to gather; default: gather all methods present")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None,
                   help="output h5 path (default: ex/<exp>/raw_results/results.h5)")
    args = p.parse_args()

    adapter = importlib.import_module(f"ex.{args.experiment}.step2_adapter")
    config_path = args.config or f"ex/{args.experiment}/config.yaml"
    config = adapter.load_config(config_path)
    n_cells_total = len(list(adapter.list_cells(config)))

    fragments_dir = _data_root() / args.experiment / "step2_results"
    if not fragments_dir.exists():
        raise SystemExit(f"no fragments dir at {fragments_dir}")

    if args.method:
        methods = [args.method]
    else:
        methods = sorted(d.name for d in fragments_dir.iterdir() if d.is_dir())

    # adapter may provide gather_output_path/dataset_name to control naming.
    out_path = Path(args.out) if args.out else None
    if out_path is None:
        if hasattr(adapter, "gather_output_path"):
            out_path = Path(adapter.gather_output_path(config))
        else:
            out_path = Path(f"ex/{args.experiment}/raw_results/results.h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # adapter may also override the dataset-name template (e.g. elbo writes
    # 'est_eldrs_arr_<method>' instead of the default 'est_ldrs_arr_<method>').
    if hasattr(adapter, "gather_dataset_name"):
        ds_name_fn = lambda m: adapter.gather_dataset_name(m, config)  # noqa: E731
    else:
        ds_name_fn = lambda m: f"est_ldrs_arr_{m}"  # noqa: E731

    for method in methods:
        arr, summary = gather_method(args.experiment, method, n_cells_total, fragments_dir)
        if arr is None:
            print(f"[{method}] no usable cells; skipping write")
            continue
        ds_name = ds_name_fn(method)
        with h5py.File(out_path, "a") as f:
            if ds_name in f:
                del f[ds_name]
            f.create_dataset(ds_name, data=arr)
        print(f"[{method}] {summary}; wrote {ds_name} shape={arr.shape} -> {out_path}")

    # optional adapter post-step (e.g. eig appends 'true_eigs_arr').
    if hasattr(adapter, "gather_postprocess"):
        try:
            adapter.gather_postprocess(config, str(out_path))
        except Exception as e:
            print(f"[gather_postprocess] failed: {e}")


if __name__ == "__main__":
    main()
