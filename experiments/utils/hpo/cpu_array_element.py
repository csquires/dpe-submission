"""per-array-element runner for cpu hpo dispatch.

claims K trials from the BACK of the shared watchdog queue file (atomic via
flock), parses each sbatch line to recover trial_runner CLI args, runs the
trials in-process using cpu_runner._eval_trial, writes trial_<id>.json output
matching the gpu trial_runner schema.

each sbatch line is the full preempt-targeted sbatch invocation that the gpu
watchdog would have eval'd; we discard the sbatch wrapper and extract:
  --experiment, --method, --config-file, --eval-cells-file,
  --output-dir, --stage
then run the trial directly on cpu, sharing a single preloaded cell cache
across all trials assigned to this element.

usage (slurm array element):
  python -m experiments.utils.hpo.cpu_array_element \\
      --queue-file <queue> --lock-file <lock> \\
      --n-per-element K --output-root <root> [--method-filter TSM,BDRE,...]

env:
  SLURM_ARRAY_TASK_ID is read for logging only; claim ordering is by queue
  back-pop, not array index.
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from experiments.utils.hpo.adapters import get_adapter
from experiments.utils.hpo.cell_schema import coerce_cells_from_json
from experiments.utils.hpo.method_specs import METHOD_SPECS
from experiments.utils.hpo.registry import LEGACY_ALIASES
from experiments.utils.watchdog import pop_lines_back_atomic
from experiments.utils.hpo import cpu_runner


# ---------------------------------------------------------------------------
# queue line parsing
# ---------------------------------------------------------------------------

_FLAG_PATTERNS = {
    "experiment":      re.compile(r"--experiment\s+(\S+)"),
    "method":          re.compile(r"--method\s+(\S+)"),
    "config_file":     re.compile(r"--config-file\s+(\S+)"),
    "eval_cells_file": re.compile(r"--eval-cells-file\s+(\S+)"),
    "output_dir":      re.compile(r"--output-dir\s+(\S+)"),
    "stage":           re.compile(r'--stage\s+([^\s"]+)'),
}


def parse_queue_line(line: str) -> Optional[dict]:
    """extract trial_runner CLI args from a queue line.

    queue line format: <method>\\t<pilot_tag>\\t<sbatch_cmd>
    sbatch_cmd contains the trial_runner invocation inside --wrap="...".

    returns dict with keys: method, pilot_tag, experiment, config_file,
    eval_cells_file, output_dir, stage. returns None on malformed line.
    """
    parts = line.split("\t", 2)
    if len(parts) != 3:
        return None
    method, pilot_tag, sbatch_cmd = parts
    parsed = {"method": method, "pilot_tag": pilot_tag}
    for key, pat in _FLAG_PATTERNS.items():
        m = pat.search(sbatch_cmd)
        if not m:
            return None
        parsed[key] = m.group(1)
    return parsed


# ---------------------------------------------------------------------------
# trial execution
# ---------------------------------------------------------------------------

def run_one_trial(parsed: dict, cell_cache_keys: set[tuple]) -> dict:
    """run a single trial inline (no fork) using cpu_runner._eval_trial.

    assumes cpu_runner._CELL_DATA_CACHE is preloaded with the cells used by
    this trial. returns the result dict written to trial_<id>.json.

    args:
      parsed: dict from parse_queue_line.
      cell_cache_keys: set of cell tuples currently in the cache. used to
        determine which cells need loading (vs already preloaded).
    """
    method = LEGACY_ALIASES.get(parsed["method"], parsed["method"])
    spec = METHOD_SPECS[method]
    adapter = get_adapter(parsed["experiment"])

    # load trial config
    cfg = json.loads(Path(parsed["config_file"]).read_text())

    # load cells for this trial
    cells_raw = json.loads(Path(parsed["eval_cells_file"]).read_text())
    cells = coerce_cells_from_json(cells_raw["cells"])
    eval_sample_seed = cells_raw.get("eval_sample_seed")

    # ensure cache holds every cell this trial needs
    new_cells = [c for c in cells if (parsed["experiment"], c) not in cell_cache_keys]
    for cell in new_cells:
        key = (parsed["experiment"], cell)
        cpu_runner._CELL_DATA_CACHE[key] = adapter.load_cell_data(cell, device="cpu")
        cell_cache_keys.add(key)

    # run the eval inline (no fork; one trial at a time per element)
    result = cpu_runner._eval_trial(
        cfg,
        cells=cells,
        method=method,
        metric_key=adapter.metric_key(),
        latent_dim=adapter.latent_dim(),
        num_waypoints=spec.get("num_waypoints"),
        requires_pstar=spec.get("requires_pstar", False),
        cell_seed_ns=adapter.cell_seed_namespace(),
        output_dir=parsed["output_dir"],
        stage=parsed["stage"],
        inner_threads=int(os.environ.get("CPU_INNER_THREADS", "2")),
        eval_sample_seed=eval_sample_seed,
        experiment=parsed["experiment"],
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cpu array element runner")
    p.add_argument("--queue-file", type=Path, default=None,
                   help="legacy back-pop mode: shared queue file path")
    p.add_argument("--lock-file", type=Path, default=None,
                   help="legacy back-pop mode: flock file path")
    p.add_argument("--n-per-element", type=int, default=4,
                   help="max trials to claim per element (default 4)")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="parallel workers per element (default 1 = sequential, "
                        "matches single-process pilot baseline). cpus_per_task "
                        "should equal n_jobs * inner_threads.")
    p.add_argument("--method-filter", type=str, default=None,
                   help="csv method names to claim; no default (all methods claimable)")
    p.add_argument("--max-elapsed-seconds", type=int, default=None,
                   help="exit early if cumulative trial time exceeds this")
    p.add_argument("--empty-retries", type=int, default=3,
                   help="legacy back-pop only: race-buffer retries when queue "
                        "is empty (default 3). ignored in --assignment-file mode")
    p.add_argument("--empty-sleep-seconds", type=int, default=10,
                   help="legacy back-pop only: seconds between empty-queue retries")
    p.add_argument("--assignment-file", type=Path, default=None,
                   help="push-file mode: text file of queue lines pre-assigned "
                        "by watchdog; takes lines [task_id*n : (task_id+1)*n]")
    p.add_argument("--assignment-b64", type=str, default=None,
                   help="push-inline mode: base64-encoded utf-8 string of "
                        "newline-joined queue lines, embedded in sbatch wrap. "
                        "preferred over --assignment-file: no NFS file, no "
                        "cleanup, payload is part of the slurm job state")
    return p.parse_args()


def _resolve_method_filter(arg: Optional[str]) -> Optional[set[str]]:
    """resolve method filter from CLI arg; return None if no filter specified.

    v3 design: drop default cpu-eligible filter. all queue lines are claimable
    from the back. slow methods may end up here, will hit per-element walltime
    cap, and orphan-recovery requeues them for preempt to retry.

    if arg is None or empty -> return None (no method filter, claim all).
    else -> return set of method names from CSV.
    """
    if arg:
        return set(arg.split(","))
    # v3: no default filter, all methods claimable from queue back
    return None


def _run_sequential(parsed_tasks: list[dict], queue_file: Path, lock_file: Path,
                    claimed: list[str], max_elapsed: Optional[int]) -> int:
    """run trials sequentially (n_jobs=1 baseline).

    preserves current single-process behavior: one trial at a time,
    shared cell cache across all trials assigned to this element.
    """
    cache_keys: set[tuple] = set()
    t_start = time.perf_counter()
    n_ok = 0
    n_err = 0

    for i, (raw_line, parsed) in enumerate(zip(claimed, parsed_tasks)):
        if (max_elapsed is not None and
                time.perf_counter() - t_start > max_elapsed):
            print(f"[cpu_array_element] elapsed budget exceeded; "
                  f"requeueing {len(claimed) - i} unclaimed lines",
                  flush=True)
            _requeue(queue_file, lock_file, claimed[i:])
            break

        try:
            run_one_trial(parsed, cache_keys)
            n_ok += 1
        except Exception as e:
            print(f"[cpu_array_element] trial error ({parsed.get('method')} "
                  f"trial_runner config={parsed.get('config_file')}): "
                  f"{type(e).__name__}: {e}", flush=True)
            n_err += 1

    elapsed = time.perf_counter() - t_start
    print(f"[cpu_array_element] sequential done: ok={n_ok} err={n_err} "
          f"elapsed={elapsed:.1f}s", flush=True)
    return 0 if n_err == 0 else 1


def _run_pool(parsed_tasks: list[dict], n_jobs: int,
              queue_file: Path, lock_file: Path,
              claimed: list[str], max_elapsed: Optional[int]) -> int:
    """fork-based mp.Pool of n_jobs workers; each worker calls _eval_trial.

    cells preloaded in parent BEFORE fork so workers inherit via COW.
    """
    # 1. preload all cells used by claimed tasks (in parent, single-threaded)
    t_preload_start = time.perf_counter()
    cells_per_task = []
    for parsed in parsed_tasks:
        cells_raw = json.loads(Path(parsed["eval_cells_file"]).read_text())
        cells = coerce_cells_from_json(cells_raw["cells"])
        eval_sample_seed = cells_raw.get("eval_sample_seed")
        cells_per_task.append((cells, eval_sample_seed))

    all_cells_with_exp: set[tuple] = set()
    for (cells, _), parsed in zip(cells_per_task, parsed_tasks):
        for c in cells:
            all_cells_with_exp.add((parsed["experiment"], c))

    for (exp, cell) in all_cells_with_exp:
        key = (exp, cell)  # tuple key per spec 11
        if key not in cpu_runner._CELL_DATA_CACHE:
            adapter = get_adapter(exp)
            cpu_runner._CELL_DATA_CACHE[key] = adapter.load_cell_data(
                cell, device="cpu")
    print(f"[cpu_array_element] preloaded {len(all_cells_with_exp)} (exp, cell) "
          f"entries in {time.perf_counter() - t_preload_start:.1f}s", flush=True)

    # 2. build worker arg tuples (parsed_task, cells, eval_sample_seed)
    worker_args = [
        (p, c, s) for p, (c, s) in zip(parsed_tasks, cells_per_task)
    ]

    # 3. forkserver pool, dispatch.
    # forkserver (not fork): fork-after-pytorch-import deadlocks on flow
    # methods (FMDRE etc.) because POSIX fork only copies the calling
    # thread but child inherits parent's torch threadpool globals; the
    # workers then deadlock on the first heavy autograd dispatch.
    # forkserver pre-forks a clean child from the current parent state
    # before any worker dispatch — cells preloaded above are still
    # inherited via COW (initial fork from parent), but the broken
    # threadpool state is severed at the forkserver layer.
    t_pool_start = time.perf_counter()
    n_ok = 0
    n_err = 0
    ctx = mp.get_context("forkserver")
    with ctx.Pool(processes=n_jobs, maxtasksperchild=16) as pool:
        try:
            for result in pool.imap_unordered(_pool_worker, worker_args):
                if isinstance(result, Exception):
                    n_err += 1
                else:
                    n_ok += 1
                if max_elapsed is not None and \
                   time.perf_counter() - t_pool_start > max_elapsed:
                    print(f"[cpu_array_element] elapsed budget exceeded; "
                          f"terminating pool", flush=True)
                    pool.terminate()
                    break
        except KeyboardInterrupt:
            pool.terminate()
            raise

    elapsed = time.perf_counter() - t_pool_start
    print(f"[cpu_array_element] pool done: ok={n_ok} err={n_err} "
          f"elapsed={elapsed:.1f}s", flush=True)
    return 0 if n_err == 0 else 1


def _pool_worker(args: tuple) -> object:
    """fork-pool worker: dispatches one trial via cpu_runner._eval_trial.

    accepts (parsed, cells, eval_sample_seed) tuple.
    returns trial result dict or Exception on error.
    cells already preloaded in parent; workers inherit via COW.
    """
    from experiments.utils.hpo import cpu_runner
    parsed, cells, eval_sample_seed = args
    try:
        method = LEGACY_ALIASES.get(parsed["method"], parsed["method"])
        spec = METHOD_SPECS[method]
        adapter = get_adapter(parsed["experiment"])
        cfg = json.loads(Path(parsed["config_file"]).read_text())

        # cells already preloaded in parent; workers inherit via COW
        # note: _eval_trial reads from _CELL_DATA_CACHE directly using the
        # cell tuple as key. with v2 cache keying (spec 11), key is (exp, cell).

        return cpu_runner._eval_trial(
            cfg,
            cells=cells,
            method=method,
            metric_key=adapter.metric_key(),
            latent_dim=adapter.latent_dim(),
            num_waypoints=spec.get("num_waypoints"),
            requires_pstar=spec.get("requires_pstar", False),
            cell_seed_ns=adapter.cell_seed_namespace(),
            output_dir=parsed["output_dir"],
            stage=parsed["stage"],
            inner_threads=int(os.environ.get("CPU_INNER_THREADS", "1")),
            eval_sample_seed=eval_sample_seed,
            experiment=parsed["experiment"],   # NEW kwarg for cache lookup, see spec 11
        )
    except Exception as e:
        return e


def main() -> int:
    args = _parse_args()
    method_filter = _resolve_method_filter(args.method_filter)
    array_idx_raw = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    array_idx = int(array_idx_raw) if array_idx_raw.isdigit() else 0

    if args.assignment_b64:
        mode = "push-inline (decode b64 from sbatch wrap)"
    elif args.assignment_file:
        mode = "push-file (read assigned slice from file)"
    else:
        mode = "back-pop (claim from queue)"
    print(f"[cpu_array_element] task_id={array_idx} n_jobs={args.n_jobs} "
          f"n_per_element={args.n_per_element} mode={mode}", flush=True)

    claimed: list[str] = []
    if args.assignment_b64 or args.assignment_file:
        # PUSH model: watchdog pre-assigned work. element takes slice
        # [task_id*n : (task_id+1)*n]. no flock contention, no back-pop.
        if args.assignment_b64:
            import base64
            payload = base64.b64decode(args.assignment_b64).decode("utf-8")
        else:
            if not args.assignment_file.exists():
                print(f"[cpu_array_element] assignment file missing: "
                      f"{args.assignment_file}", flush=True)
                return 0
            payload = args.assignment_file.read_text()
        all_lines = [ln for ln in payload.splitlines() if ln.strip()]
        start = array_idx * args.n_per_element
        claimed = all_lines[start : start + args.n_per_element]
        if not claimed:
            print(f"[cpu_array_element] task_id={array_idx} has no assigned "
                  f"work (total lines={len(all_lines)})", flush=True)
            return 0
    else:
        # LEGACY back-pop mode: pull from shared queue under flock.
        for attempt in range(args.empty_retries + 1):
            claimed = pop_lines_back_atomic(args.queue_file, args.lock_file,
                                            args.n_per_element, method_filter)
            if claimed:
                break
            if attempt < args.empty_retries:
                time.sleep(args.empty_sleep_seconds)
        if not claimed:
            return 0

    # parse all claims first; filter malformed
    parsed_tasks = []
    for raw_line in claimed:
        parsed = parse_queue_line(raw_line)
        if parsed is None:
            print(f"[cpu_array_element] skipping malformed: {raw_line[:80]}",
                  flush=True)
            continue
        parsed_tasks.append(parsed)
    if not parsed_tasks:
        return 1

    # branch: sequential (n_jobs=1, baseline) or pool (n_jobs>1)
    if args.n_jobs == 1:
        return _run_sequential(parsed_tasks, args.queue_file, args.lock_file,
                               claimed, args.max_elapsed_seconds)
    else:
        return _run_pool(parsed_tasks, args.n_jobs, args.queue_file,
                         args.lock_file, claimed, args.max_elapsed_seconds)


def _requeue(queue_file: Path, lock_file: Path, lines: list[str]) -> None:
    """append unclaimed lines back to queue under flock."""
    import fcntl
    import os
    with open(lock_file, "a+") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        with open(queue_file, "a") as f:
            for line in lines:
                f.write(line if line.endswith("\n") else line + "\n")
            f.flush()
            os.fsync(f.fileno())
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    sys.exit(main())
