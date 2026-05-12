"""sbatch the cpu array job that drains the watchdog queue from the back.

submits one slurm array job to the array partition. each element invokes
cpu_array_element to claim k trials from the queue back (atomic via flock)
and run them inline. concurrency is capped via the %N syntax (matches the
array_qos MaxJobsPU=100 limit on this cluster).

usage:
  python -m experiments.utils.hpo.cpu_dispatcher \\
      --queue-file <path> \\
      --array-size N --concurrency 100 \\
      --n-per-element K --walltime HH:MM:SS \\
      [--method-filter TSM,BDRE] \\
      [--cpus-per-task 4] [--mem 16G] [--inner-threads 2] \\
      [--dry-run]

the array job is parented to the watchdog logdir for log aggregation. each
array element writes trial_<id>.json to the SAME paths as the gpu trial
runner; downstream stages consume both interchangeably.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

WORKDIR = os.environ.get("DPE_WORKDIR") or os.getcwd()


def _build_wrap_cmd(queue_file: Path, lock_file: Path,
                    n_per_element: int, method_filter: str,
                    inner_threads: int, n_jobs: int = 1,
                    assignment_file: Optional[Path] = None,
                    assignment_b64: Optional[str] = None) -> str:
    """build the --wrap string for the array element.

    modes (in priority order):
      assignment_b64: payload baked inline into wrap (preferred; no NFS file)
      assignment_file: payload at NFS path (fallback for huge payloads)
      legacy back-pop: element flock-pops from queue_file

    sets BLAS env vars BEFORE python launches (read at numpy/torch import time).
    PYTHONHASHSEED pinned for cross-process determinism.
    """
    if assignment_b64 is not None:
        # b64 is shell-safe in single quotes (no special chars besides =, /, +)
        worker_args = (
            f"--assignment-b64 '{assignment_b64}' "
            f"--n-per-element {n_per_element} --n-jobs {n_jobs}"
        )
    elif assignment_file is not None:
        worker_args = (
            f"--assignment-file {shlex.quote(str(assignment_file))} "
            f"--n-per-element {n_per_element} --n-jobs {n_jobs}"
        )
    else:
        method_arg = f"--method-filter '{method_filter}'" if method_filter else ""
        worker_args = (
            f"--queue-file {shlex.quote(str(queue_file))} "
            f"--lock-file {shlex.quote(str(lock_file))} "
            f"--n-per-element {n_per_element} --n-jobs {n_jobs} {method_arg}"
        )
    return (
        f"set +u && source ~/.bashrc && conda activate fac && set -u && "
        f"export OMP_NUM_THREADS={inner_threads} && "
        f"export MKL_NUM_THREADS={inner_threads} && "
        f"export OPENBLAS_NUM_THREADS={inner_threads} && "
        f"export CPU_INNER_THREADS={inner_threads} && "
        f"export PYTHONHASHSEED=42 && "
        f"export HDF5_USE_FILE_LOCKING=FALSE && "
        f"cd {WORKDIR} && "
        f"python -m experiments.utils.hpo.cpu_array_element {worker_args}"
    )


def _build_sbatch(queue_file: Path, lock_file: Path,
                  array_size: int, concurrency: int,
                  walltime: str, cpus_per_task: int, mem: str,
                  n_per_element: int, method_filter: str,
                  inner_threads: int, log_dir: Path,
                  jobname: str, dependency: str = "",
                  n_jobs: int = 1,
                  assignment_file: Optional[Path] = None,
                  assignment_b64: Optional[str] = None) -> list[str]:
    wrap = _build_wrap_cmd(queue_file, lock_file, n_per_element,
                           method_filter, inner_threads, n_jobs,
                           assignment_file=assignment_file,
                           assignment_b64=assignment_b64)
    cmd = [
        "sbatch",
        "--parsable",
        "--partition=array",
        f"--time={walltime}",
        f"--mem={mem}",
        f"--cpus-per-task={cpus_per_task}",
        f"--array=0-{array_size - 1}%{concurrency}",
        f"--job-name={jobname}",
        f"--output={log_dir}/%A_%a.out",
    ]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(f"--wrap={wrap}")
    return cmd


def submit_cpu_array(
    queue_file: Path,
    lock_file: Path,
    array_size: int,
    log_dir: Path,
    *,
    concurrency: int = 100,
    n_per_element: int = 8,
    walltime: str = "4:00:00",
    cpus_per_task: int = 2,
    mem: str = "8G",
    inner_threads: int = 2,
    method_filter: str = "",
    dependency: str = "",
    job_name: str = "arr",
    n_jobs: int = 1,
    assignment_file: Optional[Path] = None,
    assignment_b64: Optional[str] = None,
) -> str:
    """sbatch a slurm array job to partition=array; returns array_jid string.

    validates resource constraints and submits via sbatch --parsable.
    raises ValueError on sbatch failure (rc != 0) with detailed error message
    including return code, stderr, and command text.

    constraints:
    - array_size > 0 (raises ValueError if not)
    - concurrency * cpus_per_task <= 256 (array_qos cap; raises ValueError if not)

    dependency: optional slurm dependency spec (e.g. "after:12345"). forwarded
    as --dependency=<spec> if non-empty; if empty, no dependency flag added.
    """
    # validate array_size > 0
    if array_size <= 0:
        raise ValueError(f"array_size must be > 0, got {array_size}")

    # validate concurrency * cpus_per_task <= 256
    max_concurrent_cpus = concurrency * cpus_per_task
    if max_concurrent_cpus > 256:
        raise ValueError(
            f"concurrency * cpus_per_task = {concurrency} * {cpus_per_task} "
            f"= {max_concurrent_cpus} exceeds array_qos limit of 256"
        )

    # ensure log_dir exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # build sbatch cmd (use refactored _build_sbatch with dependency)
    cmd = _build_sbatch(
        queue_file=queue_file,
        lock_file=lock_file,
        array_size=array_size,
        concurrency=concurrency,
        walltime=walltime,
        cpus_per_task=cpus_per_task,
        mem=mem,
        n_per_element=n_per_element,
        method_filter=method_filter,
        inner_threads=inner_threads,
        log_dir=log_dir,
        jobname=job_name,
        dependency=dependency,
        n_jobs=n_jobs,
        assignment_file=assignment_file,
        assignment_b64=assignment_b64,
    )

    # run sbatch
    result = subprocess.run(cmd, capture_output=True, text=True)

    # parse output or raise ValueError on failure
    if result.returncode != 0:
        raise ValueError(
            f"sbatch failed (rc={result.returncode})\n"
            f"stderr: {result.stderr}\n"
            f"cmd: {' '.join(shlex.quote(str(c)) for c in cmd)}"
        )

    # extract and return array_jid from --parsable output
    array_jid = result.stdout.strip().splitlines()[0]
    return array_jid


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cpu array dispatcher")
    p.add_argument("--queue-file", type=Path, required=True)
    p.add_argument("--lock-file", type=Path, default=None,
                   help="defaults to <queue-file>.lock in same dir")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="log dir for array elements; default <queue parent>/cpu_array_logs")
    p.add_argument("--array-size", type=int, default=10,
                   help="number of array elements (default 10)")
    p.add_argument("--concurrency", type=int, default=100,
                   help="max concurrent elements (default 100, matches array_qos MaxJobsPU)")
    p.add_argument("--n-per-element", type=int, default=4,
                   help="trials per array element (default 4)")
    p.add_argument("--walltime", type=str, default="04:00:00",
                   help="per-element walltime HH:MM:SS or 'auto' to compute; default 04:00:00")
    p.add_argument("--cpus-per-task", type=int, default=4,
                   help="cpus per element (default 4 -> 2 workers x 2 inner threads)")
    p.add_argument("--mem", type=str, default="16G",
                   help="memory per element (default 16G)")
    p.add_argument("--inner-threads", type=int, default=2,
                   help="BLAS threads per worker (default 2); total cores ~= "
                        "(cpus_per_task / inner_threads) * inner_threads")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="parallel workers per element via mp.Pool fork "
                        "(default 1 = sequential, single-process baseline)")
    p.add_argument("--method-filter", type=str, default=None,
                   help="csv of allowed methods; default: all cpu-eligible")
    p.add_argument("--jobname", type=str, default=None,
                   help="slurm job name; default auto-derives "
                        "'arr_<exp>[_<methods>]' from --queue-file "
                        "(parent.parent.name) and --method-filter")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    queue_file = args.queue_file.resolve()
    lock_file = args.lock_file or queue_file.with_suffix(queue_file.suffix + ".lock")
    log_dir = args.log_dir or queue_file.parent / "cpu_array_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # validate constraints before submission
    max_concurrent_cpus = args.concurrency * args.cpus_per_task
    if max_concurrent_cpus > 256:
        print(
            f"[cpu_dispatcher] ERROR: concurrency * cpus_per_task "
            f"= {args.concurrency} * {args.cpus_per_task} = {max_concurrent_cpus} "
            f"exceeds array_qos limit of 256",
            file=sys.stderr
        )
        return 1

    if args.array_size <= 0:
        print(
            f"[cpu_dispatcher] ERROR: array_size must be > 0, got {args.array_size}",
            file=sys.stderr
        )
        return 1

    # resolve --walltime auto
    if args.walltime == "auto":
        from experiments.utils.walltime_caps import compute_element_walltime
        method_list = args.method_filter.split(",") if args.method_filter else []
        try:
            args.walltime = compute_element_walltime(method_list, args.n_per_element)
            print(f"[cpu_dispatcher] resolved --walltime auto to {args.walltime}")
        except ValueError as e:
            print(f"[cpu_dispatcher] --walltime auto failed: {e}", file=sys.stderr)
            return 1

    # resolve --jobname auto: derive arr_<exp>[_<methods>] for squeue legibility.
    # try filename first (top-level "<exp>_watchdog_queue.txt" pattern), then
    # parent-dir layout (.../<exp>/watchdog/queue.jsonl).
    if args.jobname is None:
        fname = queue_file.stem
        if "_watchdog" in fname:
            exp = fname.split("_watchdog")[0]
        elif queue_file.parent.name in ("watchdog", "queues"):
            exp = queue_file.parent.parent.name
        else:
            exp = queue_file.parent.name or "exp"
        parts = ["arr", exp]
        if args.method_filter:
            mfs = args.method_filter.replace(",", "_")
            parts.append(mfs[:24])
        args.jobname = "_".join(parts)

    cmd = _build_sbatch(
        queue_file=queue_file, lock_file=lock_file,
        array_size=args.array_size, concurrency=args.concurrency,
        walltime=args.walltime, cpus_per_task=args.cpus_per_task,
        mem=args.mem, n_per_element=args.n_per_element,
        method_filter=args.method_filter or "",
        inner_threads=args.inner_threads, log_dir=log_dir,
        jobname=args.jobname,
        dependency="",
        n_jobs=args.n_jobs,
    )

    print("[cpu_dispatcher] sbatch command:")
    for c in cmd[:-1]:
        print(f"  {c}")
    print(f"  --wrap=<{len(cmd[-1].removeprefix('--wrap='))} chars>")

    if args.dry_run:
        print("[cpu_dispatcher] dry-run; not submitted")
        return 0

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[cpu_dispatcher] sbatch failed (rc={result.returncode}):",
              file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 1
    array_jid = result.stdout.strip().splitlines()[0]
    print(f"[cpu_dispatcher] submitted: array_jid={array_jid} "
          f"size={args.array_size} concurrency={args.concurrency}")
    print(f"[cpu_dispatcher] logs: {log_dir}/{array_jid}_*.out")
    print(f"[cpu_dispatcher] monitor: squeue -j {array_jid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
