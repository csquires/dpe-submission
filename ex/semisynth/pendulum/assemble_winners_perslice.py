"""assemble per-slice winners.yaml for pendulum peak campaign.

reads best_hp.json produced by aggregate_holdout for every
(method, slice) pair in ex.utils.hpo.optuna.configs.pendulum_peak_avi,
emits schema A yaml with `per_bucket` keyed by the step2 adapter's
bucket_for_cell string ("k1_idx_{k1}").

output:
    scratch/gold_winners/winners.pendulum.perslice.yaml
    (writes ONLY this file; the june winners.pendulum.yaml is left untouched)

usage:
    python -m ex.semisynth.pendulum.assemble_winners_perslice [--allow-missing]

flags:
    --allow-missing   emit winners.yaml even if some (method, slice) pairs
                      lack best_hp.json. missing entries are omitted from
                      per_bucket; step2 falls through to the top-level
                      `hyperparams` default.  default: hard-fail on any
                      missing pair so gaps are surfaced.

fixed_hp from the study config is merged INTO each per-bucket hyperparams
block, matching the model_selection assemble_winners.py convention.

slice-to-bucket mapping: pendulum slices are tuples (k1, beta) with beta=0
for all campaign slices; adapter's bucket_for_cell collapses to k1 alone
(f"k1_idx_{k1}"), so we drop beta here to match.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from ex.utils.hpo.optuna.study_config import load_config
from ex.utils.hpo.optuna.storage import _serialize_slice


CFG_MODULE = "ex.utils.hpo.optuna.configs.pendulum_peak_avi"
OUT_PATH = Path("scratch/gold_winners/winners.pendulum.perslice.yaml")


def slice_to_bucket(slice_tuple: tuple[int, int]) -> str:
    """map pendulum slice (k1, beta) -> bucket key used by adapter.

    adapter collapses to k1 alone (beta constant across campaign), so we
    ignore beta here.
    """
    k1, _beta = slice_tuple
    return f"k1_idx_{k1}"


def load_best_hp(root: Path, exp: str, method: str, slice_ser: str) -> dict | None:
    """read best_hp.json for one (method, slice). returns None if absent."""
    bhp = root / "holdout" / exp / method / f"slice_{slice_ser}" / "best_hp.json"
    if not bhp.exists():
        return None
    return json.loads(bhp.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true",
                    help="emit partial winners.yaml; default is hard-fail")
    ap.add_argument("--dpe-data-root", default=None,
                    help="override DPE_DATA_ROOT (default: env var)")
    args = ap.parse_args()

    import os
    root = Path(args.dpe_data_root or os.environ["DPE_DATA_ROOT"])
    cfg = load_config(CFG_MODULE)
    fixed_hp = dict(cfg.fixed_hp or {})

    methods_block: dict = {}
    missing: list[tuple[str, str]] = []  # (method, bucket_key)

    for m in cfg.methods:
        per_bucket: dict = {}
        for s in cfg.slices:
            ser = _serialize_slice(s)
            bkt = slice_to_bucket(s)
            best = load_best_hp(root, cfg.experiment, m, ser)
            if best is None:
                missing.append((m, bkt))
                continue
            hp = dict(best.get("best_hp") or {})
            hp.update(fixed_hp)
            per_bucket[bkt] = {
                "hyperparams": hp,
                "score": {
                    "best_value_median": best.get("best_value_median"),
                    "winner_trial_number": best.get("winner_trial_number"),
                    "best_step": best.get("best_step"),
                    "source": "hpo_holdout_perslice",
                },
            }
        if not per_bucket:
            # method has zero coverage; keep out of methods_block entirely so
            # load_winners raises (unknown method) rather than a silent no-op.
            continue
        # emit method block with per_bucket only.  top-level `hyperparams` is
        # optional; omit it here so step2 fails-clear if a cell falls into a
        # bucket without a per_bucket entry.  callers who want fallthrough can
        # add a default section post-hoc.
        methods_block[m] = {"per_bucket": per_bucket}

    n_pairs = len(cfg.methods) * len(cfg.slices)
    n_present = n_pairs - len(missing)
    print(f"pairs present:  {n_present}/{n_pairs}")
    print(f"pairs missing:  {len(missing)}")
    if missing:
        print("missing (method, bucket):")
        for m, b in missing:
            print(f"  {m}  {b}")

    if missing and not args.allow_missing:
        print(f"\nHARD FAIL: {len(missing)} pairs missing best_hp.json.",
              file=sys.stderr)
        print("Re-run when holdouts complete, or pass --allow-missing.",
              file=sys.stderr)
        return 2

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {"methods": methods_block,
           "provenance": {
               "config": CFG_MODULE,
               "n_methods": len(methods_block),
               "n_pairs_present": n_present,
               "n_pairs_missing": len(missing),
           }}
    with open(OUT_PATH, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)
    print(f"\nwrote {OUT_PATH} ({len(methods_block)} methods)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
