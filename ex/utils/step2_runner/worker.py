"""step2_runner worker: process one (method, cell_chunk) of an experiment.

invocation (typically from a watchdog-spawned sbatch job):

    python -m ex.utils.step2_runner.worker \
        --experiment <exp_short_name> \
        --method <method_name> \
        --cell-indices "0,1,2,3,...,19" \
        --winners /path/to/winners.yaml \
        --output-dir /path/to/<exp>/step2_results \
        --config /path/to/config.yaml

dispatches to the experiment's step2_adapter module. each cell processed
independently; per-cell h5 result lands at:

    <output_dir>/<method>/cell_<idx>.h5

idempotent: cells with existing result file are skipped. partial chunks are OK
(every cell that completes writes its result atomically).
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import h5py
import numpy as np

from ex.utils.step2_runner.load_winners import load_winners, resolve_hp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True,
                   help="experiment short name (matches ex/<exp>/step2_adapter.py)")
    p.add_argument("--method", required=True)
    p.add_argument("--cell-indices", required=True,
                   help="comma-separated list of cell indices to process")
    p.add_argument("--winners", required=True, help="path to winners.yaml")
    p.add_argument("--output-dir", required=True,
                   help="root output dir; per-cell results written under <method>/cell_<idx>.h5")
    p.add_argument("--config", default=None,
                   help="path to experiment config.yaml (default: ex/<exp>/config.yaml)")
    p.add_argument("--device", default=None)
    return p.parse_args()


def _atomic_h5_write(path: Path, payload: dict, attrs: dict | None = None) -> None:
    """write a small h5 file atomically. payload values must be array-likes or scalars."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        for k, v in payload.items():
            f.create_dataset(k, data=v)
        if attrs:
            for k, v in attrs.items():
                # h5 attrs must be scalars or strings; serialize dicts as json strings
                if isinstance(v, dict):
                    f.attrs[k] = json.dumps(v, default=str)
                else:
                    f.attrs[k] = v
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()

    # parse cell indices
    cell_indices = [int(x) for x in args.cell_indices.split(",") if x.strip()]
    if not cell_indices:
        raise SystemExit("--cell-indices empty")

    # load adapter
    adapter_mod = f"ex.{args.experiment}.step2_adapter"
    try:
        adapter = importlib.import_module(adapter_mod)
    except ImportError as e:
        raise SystemExit(f"could not import {adapter_mod}: {e}")

    # load config
    config_path = args.config or f"ex/{args.experiment}/config.yaml"
    config = adapter.load_config(config_path)

    # load winners
    winners = load_winners(args.winners)

    # device
    device = args.device or config.get("device", "cuda")

    # output directory
    output_root = Path(args.output_dir) / args.method
    output_root.mkdir(parents=True, exist_ok=True)

    # process each cell
    n_done, n_skipped, n_failed = 0, 0, 0
    chunk_t0 = time.time()
    for cell_idx in cell_indices:
        out_path = output_root / f"cell_{cell_idx}.h5"
        if out_path.exists():
            print(f"[skip exists] {out_path}")
            n_skipped += 1
            continue

        try:
            bucket_id = adapter.bucket_for_cell(cell_idx, config)
            hp = resolve_hp(winners, args.method, bucket_id)
        except (KeyError, ValueError) as e:
            print(f"[hp error] cell={cell_idx} bucket={bucket_id!r}: {e}")
            n_failed += 1
            continue

        t0 = time.time()
        err = None
        try:
            result = adapter.fit_and_eval(
                method=args.method,
                hp=hp,
                cell_idx=cell_idx,
                config=config,
                device=device,
            )
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
            result = None
        elapsed = time.time() - t0

        if result is None or err is not None:
            # write a sentinel so we don't retry forever
            _atomic_h5_write(
                out_path,
                payload={"est_ldrs": np.array([np.nan])},
                attrs={
                    "experiment": args.experiment,
                    "method": args.method,
                    "cell_idx": cell_idx,
                    "bucket_id": str(bucket_id) if bucket_id is not None else "",
                    "hyperparams": hp,
                    "elapsed_seconds": float(elapsed),
                    "error": err or "no result",
                    "ok": False,
                },
            )
            print(f"[fail] cell={cell_idx} elapsed={elapsed:.1f}s err={(err or 'no result')[:80]}")
            n_failed += 1
            continue

        # required result keys (adapter contract): 'est_ldrs' (np.ndarray), 'mae_per_test_set' (np.ndarray)
        # optional: 'true_ldrs' (np.ndarray, for diagnostic).
        payload = {"est_ldrs": np.asarray(result["est_ldrs"])}
        if "mae_per_test_set" in result:
            payload["mae_per_test_set"] = np.asarray(result["mae_per_test_set"])
        if "true_ldrs" in result:
            payload["true_ldrs"] = np.asarray(result["true_ldrs"])

        _atomic_h5_write(
            out_path,
            payload=payload,
            attrs={
                "experiment": args.experiment,
                "method": args.method,
                "cell_idx": cell_idx,
                "bucket_id": str(bucket_id) if bucket_id is not None else "",
                "hyperparams": hp,
                "elapsed_seconds": float(elapsed),
                "ok": True,
            },
        )
        n_done += 1
        print(f"[ok]   cell={cell_idx} elapsed={elapsed:.1f}s")

    chunk_elapsed = time.time() - chunk_t0
    print(f"\n[chunk done] method={args.method} cells={cell_indices} "
          f"done={n_done} skipped={n_skipped} failed={n_failed} elapsed={chunk_elapsed:.1f}s")


if __name__ == "__main__":
    main()
