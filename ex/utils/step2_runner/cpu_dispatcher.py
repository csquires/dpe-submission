"""sbatch the cpu array job that drains the step2 queue from the back.

submits one slurm array job to the array partition. each element invokes
ex.utils.step2_runner.cpu_array_element to claim K queue lines via
flock-back-pop and run them inline on cpu. concurrency capped via %N to
respect the array_qos MaxJobsPU limit.

usage:
  python -m ex.utils.step2_runner.cpu_dispatcher \\
      --queue-file <path> --lock-file <path> \\
      --array-size N --concurrency 100 \\
      --n-per-element K --walltime HH:MM:SS \\
      --output-root <DPE_DATA_ROOT>/<exp>/step2_cpu_array_logs \\
      [--method-filter CTSM,BDRE,...] \\
      [--cpus-per-task 4] [--mem 16G] [--device cpu] \\
      [--dry-run]

intended to be paired with submit_watchdog_lite.sh (front-pop GPU drain) so
the two consumers feed from the same queue file from opposite ends.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

WORKDIR = os.environ.get("DPE_WORKDIR", "/home/aviamala/dpe-submission")
CONDA_ENV = os.environ.get("DPE_CONDA_ENV", "fac")


def _build_wrap(queue_file: Path, lock_file: Path, n_per_element: int,
                method_filter: str, device: str) -> str:
    """build --wrap string for the array element."""
    args = [
        "python", "-m", "ex.utils.step2_runner.cpu_array_element",
        "--queue-file", str(queue_file),
        "--lock-file", str(lock_file),
        "--n-per-element", str(n_per_element),
        "--device", device,
        "--workdir", WORKDIR,
    ]
    if method_filter:
        args.extend(["--method-filter", method_filter])
    quoted = " ".join(shlex.quote(a) for a in args)
    return (
        f"set +u && source ~/.bashrc && conda activate {CONDA_ENV} && set -u && "
        f"cd {WORKDIR} && {quoted}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue-file", type=Path, required=True)
    p.add_argument("--lock-file", type=Path, default=None,
                   help="flock file (default: <queue_file>.lock)")
    p.add_argument("--array-size", type=int, required=True,
                   help="total number of array elements; size to "
                        "ceil(num_eligible_lines / n_per_element)")
    p.add_argument("--concurrency", type=int, default=100,
                   help="max simultaneous elements (slurm %%N modifier)")
    p.add_argument("--n-per-element", type=int, default=2,
                   help="lines claimed per array element")
    p.add_argument("--walltime", default="6:00:00")
    p.add_argument("--partition", default="array")
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--mem", default="16G")
    p.add_argument("--device", default="cpu")
    p.add_argument("--method-filter", default="",
                   help="comma-separated cpu-eligible method whitelist")
    p.add_argument("--output-root", type=Path, required=True,
                   help="dir for cpu_array element logs (one out/err per element)")
    p.add_argument("--job-name", default="step2_cpuarr")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.queue_file.exists():
        sys.exit(f"queue file not found: {args.queue_file}")
    lock_file = args.lock_file or args.queue_file.with_suffix(args.queue_file.suffix + ".lock")
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_pattern = args.output_root / "elem_%a.out"

    array_spec = f"1-{args.array_size}%{args.concurrency}"
    wrap = _build_wrap(args.queue_file, lock_file, args.n_per_element,
                       args.method_filter, args.device)

    cmd = [
        "sbatch",
        f"--partition={args.partition}",
        f"--time={args.walltime}",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem={args.mem}",
        f"--array={array_spec}",
        f"--job-name={args.job_name}",
        f"--output={log_pattern}",
        "--requeue",
        f"--wrap={wrap}",
    ]

    if args.dry_run:
        print("dry-run: would submit:")
        print(" ".join(shlex.quote(c) for c in cmd))
        return 0

    print(f"submitting cpu array: size={args.array_size} concurrency={args.concurrency} "
          f"n_per_element={args.n_per_element} method_filter={args.method_filter or '(all)'}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"sbatch failed: {proc.stderr}", file=sys.stderr)
        return 1
    print(proc.stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
