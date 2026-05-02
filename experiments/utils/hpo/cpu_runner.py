"""multiprocessed hpo runner for interactive cpu-only jobs.

IMPORTANT: BLAS thread counts are read at numpy/torch import time. setting
OMP_NUM_THREADS et al here (before any heavy import) is the only way to
constrain BLAS oversubscription in worker processes. callers wanting a
non-default inner thread count must set these env vars BEFORE invoking
this module (e.g., in the slurm submit script).

preloads all eval cell data once in the parent process; workers inherit via
fork (linux copy-on-write, no re-serialization per trial). dispatches n_trials
hpo trials with n_jobs concurrent workers using multiprocessing.Pool.

cpu budget:  n_jobs * inner_threads cores  (+1 for the parent).

quick smoke test (dre_sample_complexity, 5 epochs, 5 cells, 2 workers):
  python -m experiments.utils.hpo.cpu_runner \\
      --experiment dre_sample_complexity \\
      --method MDRE \\
      --n-trials 6 --n-cells 5 --n-jobs 2 --inner-threads 2 \\
      --output-dir /tmp/hpo_smoke \\
      --override-hyperparams '{"num_epochs": 5, "latent_dim": 16}'

monitoring while running (separate terminal):
  count finished:   watch -n5 "ls /tmp/hpo_smoke/broad/*.json 2>/dev/null | wc -l"
  live leaderboard: python -m experiments.utils.hpo.cpu_runner \\
                        --show-results --output-dir /tmp/hpo_smoke/broad
"""

import argparse
import functools
import gc
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

# BLAS thread caps must be set BEFORE numpy/torch import. respect any caller
# override; default to 2 (matches conventional inner_threads default).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import torch

from experiments.utils.hpo.adapters import get_adapter
from experiments.utils.hpo.cell_schema import draw_training_sample
from experiments.utils.hpo.method_specs import METHOD_SPECS
from experiments.utils.hpo.registry import LEGACY_ALIASES
from experiments.utils.hpo.sample import gen_config

# module-level: preloaded in main, inherited by forked workers via copy-on-write.
# never written after fork; no lock needed.
# key: (experiment, cell) tuple for namespacing; bare cell for backward compat.
_CELL_DATA_CACHE: dict[tuple[str, tuple] | tuple, dict[str, torch.Tensor]] = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _set_blas_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)
    torch.set_num_threads(n)


def _trimmed_mean(vals: list[float], trim_frac: float = 0.2) -> float:
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return float("inf")
    n = len(finite)
    k = round(trim_frac * n)
    core = sorted(finite) if n <= 2 * k else sorted(finite)[k : n - k]
    return float(np.mean(core))


def _load_results(results_dir: Path) -> list[dict]:
    trials = []
    for p in sorted(results_dir.glob("trial_*.json")):
        try:
            d = json.loads(p.read_text())
            if "score" in d and d["score"] is not None:
                trials.append(d)
        except Exception:
            continue
    return sorted(trials, key=lambda t: float(t["score"]))


# ---------------------------------------------------------------------------
# worker (runs in forked subprocess; reads _CELL_DATA_CACHE as a global)
# ---------------------------------------------------------------------------

def _eval_trial(
    trial_config: dict,
    *,
    cells: list[tuple],
    method: str,
    metric_key: str,
    latent_dim: int,
    num_waypoints: Optional[int],
    requires_pstar: bool,
    cell_seed_ns: str,
    output_dir: str,
    stage: str,
    inner_threads: int,
    eval_sample_seed: Optional[int] = None,
    experiment: Optional[str] = None,
) -> dict:
    """evaluate one hpo trial on all preloaded cells; write result json; return result."""
    _set_blas_threads(inner_threads)

    trial_id = trial_config["trial_id"]
    hyperparams = trial_config["hyperparams"]
    builder = METHOD_SPECS[method]["builder"]

    per_cell: dict[str, float] = {}
    t0 = time.perf_counter()

    for cell in cells:
        cs = ":".join(str(x) for x in cell)
        seed_int = hash((cell_seed_ns, trial_id, cs)) & 0xFFFFFFFF
        torch.manual_seed(seed_int)
        np.random.seed(seed_int)
        random.seed(seed_int)

        try:
            # cache lookup with experiment namespacing; backward compat fallback to bare cell key
            key = (experiment, cell) if experiment else cell
            data = _CELL_DATA_CACHE[key]
            # delegate to adapter.eval_cell so per-experiment data schema +
            # metric semantics live in the adapter, not inlined here. pass
            # pre-loaded `data` so the cell cache is preserved.
            from experiments.utils.hpo.adapters import get_adapter
            adapter = get_adapter(experiment)
            mae = adapter.eval_cell(
                cell, method, builder, hyperparams, requires_pstar, "cpu", data=data
            )
        except Exception as e:
            print(f"  [trial {trial_id}] cell {cs}: {type(e).__name__}: {e}", flush=True)
            continue
        finally:
            gc.collect()

        if not math.isfinite(mae):
            print(f"  [trial {trial_id}] cell {cs}: non-finite mae, skipping", flush=True)
            continue

        per_cell[cs] = mae

    elapsed = time.perf_counter() - t0
    finite_vals = list(per_cell.values())
    mean_metric = float(np.mean(finite_vals)) if finite_vals else float("nan")
    score = _trimmed_mean(finite_vals)

    result = {
        "method": method,
        "trial_id": trial_id,
        "hyperparams": hyperparams,
        metric_key: per_cell,
        "mean_metric": mean_metric,
        "score": score,
        "elapsed_seconds": elapsed,
        "pilot_variant": stage,
        "eval_sample_seed": eval_sample_seed,
        "training_cells": [list(c) for c in cells],
        "workflow_version": "v1",
    }

    out_dir = Path(output_dir) / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trial_{trial_id}.json"
    # pid-suffixed tmp avoids collision if a defensive dup-claim races us
    tmp_path = out_dir / f"trial_{trial_id}.json.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)
    # fsync parent dir so the rename is durable on crash (NFS/shared FS)
    try:
        dir_fd = os.open(out_dir, os.O_RDONLY)
        os.fsync(dir_fd)
        os.close(dir_fd)
    except OSError:
        pass

    print(
        f"[trial {trial_id:4d}] {method}: {len(per_cell)}/{len(cells)} cells, "
        f"score={score:.4f}, {elapsed:.1f}s -> {out_path.name}",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# leaderboard display
# ---------------------------------------------------------------------------

def _show_results(results_dir: Path, top: int) -> None:
    trials = _load_results(results_dir)
    if not trials:
        print(f"[cpu_runner] no results in {results_dir}")
        return
    n = len(trials)
    print(f"\n=== top {min(top, n)}/{n} results  ({results_dir}) ===")
    print(f"  {'rank':>4}  {'trial':>5}  {'score':>8}  {'mean_mae':>8}  hyperparams")
    for rank, t in enumerate(trials[:top], 1):
        print(
            f"  {rank:4d}  {t['trial_id']:5d}  {float(t['score']):8.4f}  "
            f"{float(t.get('mean_metric', float('nan'))):8.4f}  {t['hyperparams']}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="multiprocessed hpo runner for interactive cpu jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "monitoring (from a second terminal):\n"
            "  count done:      watch -n5 \"ls <output-dir>/broad/*.json 2>/dev/null | wc -l\"\n"
            "  live leaderboard: python -m experiments.utils.hpo.cpu_runner \\\n"
            "                        --show-results --output-dir <output-dir>/broad\n"
        ),
    )
    p.add_argument("--experiment", default=None,
                   help="adapter name, e.g. dre_sample_complexity")
    p.add_argument("--method", default=None,
                   help="method name or alias, e.g. MDRE, BDRE, TSM")
    p.add_argument("--n-trials", type=int, default=50,
                   help="total hpo trials to run (default 50)")
    p.add_argument("--n-cells", type=int, default=8,
                   help="eval cells per trial (default 8)")
    p.add_argument("--n-jobs", type=int, default=4,
                   help="parallel workers = concurrent trials (default 4)")
    p.add_argument("--inner-threads", type=int, default=2,
                   help="blas threads per worker (default 2); total cores ~ n_jobs * inner_threads")
    p.add_argument("--seed", type=int, default=1729,
                   help="cell sampling seed (default 1729)")
    p.add_argument("--stage", default="broad", choices=["broad", "refined"],
                   help="stage label for output files (default broad)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="output root; results go to <output-dir>/<stage>/trial_N.json")
    p.add_argument("--override-hyperparams", type=str, default=None,
                   help=(
                       "json dict to pin specific hyperparams, overriding sampled values. "
                       "useful for smoke tests: '{\"num_epochs\": 5, \"latent_dim\": 16}'"
                   ))
    p.add_argument("--resume", action="store_true",
                   help="skip trials whose output json already exists")
    p.add_argument("--show-results", action="store_true",
                   help="print leaderboard from existing results and exit; "
                        "--output-dir should point to the stage dir (e.g. .../broad)")
    p.add_argument("--top", type=int, default=10,
                   help="rows to show with --show-results (default 10)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # --show-results mode: just print leaderboard and exit
    if args.show_results:
        _show_results(args.output_dir, args.top)
        return

    if args.experiment is None or args.method is None:
        print("error: --experiment and --method are required", file=sys.stderr)
        sys.exit(1)

    # resolve method alias -> canonical
    method = LEGACY_ALIASES.get(args.method, args.method)
    if method not in METHOD_SPECS:
        print(
            f"error: unknown method {method!r}; known: {sorted(METHOD_SPECS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        adapter = get_adapter(args.experiment)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    if not adapter.is_ready():
        print(
            f"error: adapter {args.experiment!r} not ready "
            "(data_dir missing or incomplete?)",
            file=sys.stderr,
        )
        sys.exit(1)

    # draw eval cells (stratified if adapter provides stratify_key)
    cell_pool = adapter.cell_pool()
    n_cells = min(args.n_cells, len(cell_pool))
    stratify_fn = adapter.stratify_key if any(
        adapter.stratify_key(c) is not None for c in cell_pool[:5]
    ) else None
    cells = draw_training_sample(cell_pool, M=n_cells, seed=args.seed,
                                 stratify_fn=stratify_fn)

    spec = METHOD_SPECS[method]
    num_waypoints: Optional[int] = spec.get("num_waypoints")
    requires_pstar: bool = spec.get("requires_pstar", False)

    print(
        f"[cpu_runner] experiment={args.experiment!r}  method={method!r}\n"
        f"[cpu_runner] n_trials={args.n_trials}  n_cells={n_cells}  "
        f"n_jobs={args.n_jobs}  inner_threads={args.inner_threads}\n"
        f"[cpu_runner] estimated cpu usage: ~{args.n_jobs * args.inner_threads} cores\n"
        f"[cpu_runner] output: {args.output_dir / args.stage}",
        flush=True,
    )

    # preload all cell data into module-level global BEFORE forking.
    # forked workers inherit via copy-on-write; no re-serialization per trial.
    global _CELL_DATA_CACHE
    t_load = time.perf_counter()
    print(f"[cpu_runner] preloading {n_cells} cells...", flush=True)
    for cell in cells:
        _CELL_DATA_CACHE[(args.experiment, cell)] = adapter.load_cell_data(cell, device="cpu")
    print(f"[cpu_runner] preloaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    # generate trial configs (random hyperparameter samples)
    override = json.loads(args.override_hyperparams) if args.override_hyperparams else {}
    if override:
        print(f"[cpu_runner] hyperparameter overrides: {override}", flush=True)
    registry = {method: {"search_space": spec["base_search_space"]}}
    trial_configs = []
    # seed global random per trial_id for deterministic gen_config across runs.
    # without this, sample_param uses global random's process-local state,
    # giving different hyperparams between cpu_runner invocations.
    for trial_id in range(args.n_trials):
        random.seed(args.seed + trial_id)
        cfg = gen_config(registry, method, trial_id)
        cfg["hyperparams"].update(override)
        trial_configs.append(cfg)

    # skip already-finished trials when resuming
    out_stage_dir = args.output_dir / args.stage
    if args.resume:
        todo = [
            c for c in trial_configs
            if not (out_stage_dir / f"trial_{c['trial_id']}.json").exists()
        ]
        n_skip = len(trial_configs) - len(todo)
        if n_skip:
            print(f"[cpu_runner] --resume: skipping {n_skip} finished trials", flush=True)
        trial_configs = todo

    if not trial_configs:
        print("[cpu_runner] all trials already done.", flush=True)
        _show_results(out_stage_dir, args.top)
        return

    # bind fixed args; trial_config is the only varying argument per call
    worker = functools.partial(
        _eval_trial,
        cells=cells,
        method=method,
        metric_key=adapter.metric_key(),
        latent_dim=adapter.latent_dim(),
        num_waypoints=num_waypoints,
        requires_pstar=requires_pstar,
        cell_seed_ns=adapter.cell_seed_namespace(),
        output_dir=str(args.output_dir),
        stage=args.stage,
        inner_threads=args.inner_threads,
        eval_sample_seed=args.seed,
        experiment=args.experiment,
    )

    print(
        f"[cpu_runner] dispatching {len(trial_configs)} trials, "
        f"{args.n_jobs} at a time...",
        flush=True,
    )
    t0 = time.perf_counter()
    results = []

    # fork context: workers inherit _CELL_DATA_CACHE without pickling it.
    # imap_unordered yields results as they complete -> live progress.
    # maxtasksperchild bounds per-worker memory growth (allocator fragmentation,
    # caching) on long pools.
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.n_jobs, maxtasksperchild=16) as pool:
        for result in pool.imap_unordered(worker, trial_configs):
            results.append(result)
            n_done = len(results)
            n_total = len(trial_configs)
            elapsed = time.perf_counter() - t0
            eta = (elapsed / n_done) * (n_total - n_done)
            print(
                f"[cpu_runner] {n_done}/{n_total} done  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
                f"best_so_far={min(r['score'] for r in results):.4f}",
                flush=True,
            )

    total_elapsed = time.perf_counter() - t0
    print(
        f"\n[cpu_runner] finished: {len(results)} trials in {total_elapsed:.1f}s "
        f"({total_elapsed / max(len(results), 1):.1f}s/trial)",
        flush=True,
    )
    _show_results(out_stage_dir, args.top)

    # write summary json for downstream analysis
    summary = {
        "experiment": args.experiment,
        "method": method,
        "n_trials_run": len(results),
        "n_cells": n_cells,
        "elapsed_seconds": total_elapsed,
        "cells": [list(c) for c in cells],
        "trials_by_score": sorted(
            [
                {
                    "trial_id": r["trial_id"],
                    "score": r["score"],
                    "mean_metric": r.get("mean_metric"),
                    "hyperparams": r["hyperparams"],
                }
                for r in results
            ],
            key=lambda x: float(x["score"]),
        ),
    }
    summary_path = args.output_dir / "summary.json"
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(summary_path)
    print(f"[cpu_runner] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
