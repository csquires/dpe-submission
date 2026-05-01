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
from experiments.utils.walltime_caps import cpu_eligible_methods


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
    new_cells = [c for c in cells if c not in cell_cache_keys]
    for cell in new_cells:
        cpu_runner._CELL_DATA_CACHE[cell] = adapter.load_cell_data(cell, device="cpu")
        cell_cache_keys.add(cell)

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
        inner_threads=int(__import__("os").environ.get("CPU_INNER_THREADS", "2")),
        eval_sample_seed=eval_sample_seed,
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cpu array element runner")
    p.add_argument("--queue-file", type=Path, required=True)
    p.add_argument("--lock-file", type=Path, required=True)
    p.add_argument("--n-per-element", type=int, default=4,
                   help="max trials to claim per element (default 4)")
    p.add_argument("--method-filter", type=str, default=None,
                   help="csv method names to claim; defaults to all cpu-eligible")
    p.add_argument("--max-elapsed-seconds", type=int, default=None,
                   help="exit early if cumulative trial time exceeds this")
    return p.parse_args()


def _resolve_method_filter(arg: Optional[str]) -> Optional[set[str]]:
    if arg:
        return set(arg.split(","))
    # default: all cpu-eligible methods (from walltime_caps)
    return cpu_eligible_methods()


def main() -> int:
    args = _parse_args()
    method_filter = _resolve_method_filter(args.method_filter)
    array_idx = __import__("os").environ.get("SLURM_ARRAY_TASK_ID", "?")

    print(f"[cpu_array_element] task_id={array_idx} claiming up to "
          f"{args.n_per_element} trials from {args.queue_file}",
          flush=True)

    claimed = pop_lines_back_atomic(args.queue_file, args.lock_file,
                                    args.n_per_element, method_filter)
    if not claimed:
        print(f"[cpu_array_element] queue empty or no eligible lines; exit 0",
              flush=True)
        return 0

    print(f"[cpu_array_element] claimed {len(claimed)} lines", flush=True)

    cache_keys: set[tuple] = set()
    t_start = time.perf_counter()
    n_ok = 0
    n_err = 0

    for i, raw_line in enumerate(claimed):
        if (args.max_elapsed_seconds is not None and
                time.perf_counter() - t_start > args.max_elapsed_seconds):
            print(f"[cpu_array_element] elapsed budget exceeded; "
                  f"requeueing {len(claimed) - i} unclaimed lines",
                  flush=True)
            _requeue(args.queue_file, args.lock_file, claimed[i:])
            break

        parsed = parse_queue_line(raw_line)
        if parsed is None:
            print(f"[cpu_array_element] skipping malformed line: {raw_line[:80]}",
                  flush=True)
            n_err += 1
            continue

        try:
            run_one_trial(parsed, cache_keys)
            n_ok += 1
        except Exception as e:
            print(f"[cpu_array_element] trial error ({parsed.get('method')} "
                  f"trial_runner config={parsed.get('config_file')}): "
                  f"{type(e).__name__}: {e}", flush=True)
            n_err += 1

    elapsed = time.perf_counter() - t_start
    print(f"[cpu_array_element] done: ok={n_ok} err={n_err} elapsed={elapsed:.1f}s",
          flush=True)
    return 0 if n_err == 0 else 1


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
