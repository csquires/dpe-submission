"""Quick progress + validity check for step2 results."""
import glob
import os

import h5py
import numpy as np

DATA_ROOT = os.environ.get("DPE_DATA_ROOT") or os.path.expanduser("~/dpe-data")

targets = {"elbo_estimation": 1200, "smodice_eldr_estimation": 480}

for exp, target in targets.items():
    base = f"{DATA_ROOT}/{exp}/step2_results"
    print(f"=== {exp} (target: {target} per method) ===")
    print(f"  {'method':<35} {'done':>5}/{'tgt':<5}  {'valid':>5}  {'failed':>6}  {'pct':>5}")
    print(f"  {'-'*35} {'-'*5}  {'-'*5}  {'-'*6}  {'-'*5}")
    for method_dir in sorted(glob.glob(base + "/*")):
        method = os.path.basename(method_dir)
        files = glob.glob(method_dir + "/cell_*.h5")
        if not files:
            continue
        done = len(files)
        valid = failed = 0
        for f in files:
            with h5py.File(f, "r") as h:
                ok = h.attrs.get("ok", True)
                if ok is False or ok == False:  # noqa: E712
                    failed += 1
                else:
                    val = h["est_ldrs"][()].flat[0] if "est_ldrs" in h else float("nan")
                    if np.isnan(val):
                        failed += 1
                    else:
                        valid += 1
        pct = done * 100 // target
        print(f"  {method:<35} {done:>5}/{target:<5}  {valid:>5}  {failed:>6}  {pct:>4}%", flush=True)
    print(flush=True)
