"""per-array-element runner for step2 cpu dispatch.

claims K queue lines from the BACK of the shared queue file (atomic via
flock), parses each line to recover step2_runner.worker CLI args, and runs
each worker invocation in a subprocess. complementary to a watchdog (or
watchdog_lite) that pops from the front and dispatches to preempt — the
two together give a "double-end drain" across cpu (back) and gpu (front).

each queue line is the full preempt-targeted sbatch invocation that the gpu
watchdog would have eval'd; we discard the sbatch wrapper and extract the
worker invocation:
    --experiment, --method, --cell-indices, --winners, --output-dir, [--config]

then exec it directly on cpu via `python -m experiments.utils.step2_runner.worker`
with --device cpu. each worker call is idempotent (skips cells with existing
results), so a duplicate claim is safe.

usage (slurm array element):
  python -m experiments.utils.step2_runner.cpu_array_element \\
      --queue-file <queue> --lock-file <lock> \\
      --n-per-element K [--method-filter CTSM,BDRE,...] \\
      [--device cpu]

env:
  SLURM_ARRAY_TASK_ID is read for logging only.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from experiments.utils.watchdog import pop_lines_back_atomic


_FLAG_PATTERNS = {
    "experiment":   re.compile(r"--experiment\s+(\S+)"),
    "method":       re.compile(r"--method\s+(\S+)"),
    "cell_indices": re.compile(r"--cell-indices\s+'([^']+)'|--cell-indices\s+(\S+)"),
    "winners":      re.compile(r"--winners\s+(\S+)"),
    "output_dir":   re.compile(r"--output-dir\s+(\S+)"),
    "config":       re.compile(r"--config\s+(\S+)"),
}


def parse_queue_line(line: str) -> dict | None:
    """extract worker CLI args from a queue line.

    queue line format: <method>\\t<bucket_tag>\\t<sbatch_cmd>
    sbatch_cmd contains the worker invocation inside --wrap="...".

    returns dict with keys: method, bucket_tag, experiment, cell_indices,
    winners, output_dir, [config]. returns None on malformed line.
    """
    parts = line.split("\t", 2)
    if len(parts) != 3:
        return None
    method, bucket_tag, sbatch_cmd = parts
    out = {"method": method, "bucket_tag": bucket_tag}
    for key, pat in _FLAG_PATTERNS.items():
        m = pat.search(sbatch_cmd)
        if not m:
            if key == "config":
                out["config"] = None
                continue
            return None
        # cell_indices pattern has two groups (quoted or unquoted)
        if key == "cell_indices":
            out[key] = m.group(1) or m.group(2)
        else:
            out[key] = m.group(1)
    return out


def run_one(parsed: dict, device: str, workdir: str) -> tuple[bool, float]:
    """run one worker invocation as a subprocess. returns (ok, elapsed_seconds)."""
    cmd = [
        sys.executable, "-m", "experiments.utils.step2_runner.worker",
        "--experiment", parsed["experiment"],
        "--method", parsed["method"],
        "--cell-indices", parsed["cell_indices"],
        "--winners", parsed["winners"],
        "--output-dir", parsed["output_dir"],
        "--device", device,
    ]
    if parsed.get("config"):
        cmd.extend(["--config", parsed["config"]])
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    elapsed = time.time() - t0
    ok = proc.returncode == 0
    print(f"[element] method={parsed['method']} cells={parsed['cell_indices'][:60]} "
          f"elapsed={elapsed:.1f}s rc={proc.returncode}")
    if proc.stdout:
        for line in proc.stdout.splitlines()[:20]:
            print(f"  | {line}")
    if not ok and proc.stderr:
        for line in proc.stderr.splitlines()[-10:]:
            print(f"  ! {line}")
    return ok, elapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue-file", type=Path, required=True)
    p.add_argument("--lock-file", type=Path, required=True)
    p.add_argument("--n-per-element", type=int, required=True,
                   help="max lines to claim per array element")
    p.add_argument("--method-filter", default="",
                   help="comma-separated method whitelist (matches METHOD column)")
    p.add_argument("--device", default="cpu", help="device passed to worker")
    p.add_argument("--workdir", default=os.environ.get("DPE_WORKDIR") or os.getcwd())
    p.add_argument("--empty-retries", type=int, default=3)
    p.add_argument("--empty-sleep-seconds", type=int, default=60)
    args = p.parse_args()

    method_filter = None
    if args.method_filter:
        method_filter = {m.strip() for m in args.method_filter.split(",") if m.strip()}

    array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "?")
    print(f"[cpu_array_element id={array_id}] queue={args.queue_file} "
          f"n_per_element={args.n_per_element} filter={method_filter} device={args.device}")

    claimed: list[str] = []
    for attempt in range(args.empty_retries + 1):
        claimed = pop_lines_back_atomic(
            args.queue_file, args.lock_file, args.n_per_element, method_filter,
        )
        if claimed: break
        if attempt < args.empty_retries:
            time.sleep(args.empty_sleep_seconds)
    if not claimed:
        print("[element] queue empty; exiting")
        return 0

    print(f"[element] claimed {len(claimed)} lines")
    n_ok, n_failed = 0, 0
    total_t = 0.0
    for line in claimed:
        parsed = parse_queue_line(line)
        if parsed is None:
            print(f"[element] BAD_LINE: {line[:120]}")
            n_failed += 1
            continue
        ok, elapsed = run_one(parsed, args.device, args.workdir)
        total_t += elapsed
        if ok: n_ok += 1
        else: n_failed += 1

    print(f"[element done] ok={n_ok} failed={n_failed} total_elapsed={total_t:.1f}s")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
