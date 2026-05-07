"""5-stage HPO workflow orchestrator.

stages: recalibrate -> broad -> refined -> holdout -> persist.
each stage is a pure function; no global state. atomic I/O via fcntl.flock
and tmp+os.replace. persist must be invoked separately (see main() docstring).
"""

import os
import sys
import json
import math
import yaml
import time
import fcntl
import argparse
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Tuple

from experiments.utils.hpo.sample import gen_config
from experiments.utils.hpo.cell_schema import (
    draw_training_sample,
    draw_holdout_sample_clamped,
    coerce_cells_from_json,
)
from experiments.utils.hpo.method_specs import METHOD_SPECS
from experiments.utils.hpo.budget import stage_budget

try:
    from experiments.utils.hpo.narrow import narrow_spec
except ImportError:
    # narrow not yet implemented; placeholder for forward-compat
    def narrow_spec(spec: tuple, values: list) -> tuple:  # type: ignore
        return spec

try:
    from experiments.utils.hpo.adapters import get_adapter
except ImportError:
    get_adapter = None  # type: ignore


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomically_write_yaml(yaml_file: Path, data: Dict[str, Any]) -> None:
    """write yaml via tmp+flush+fsync+os.replace for durability."""
    yaml_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = yaml_file.parent / f"{yaml_file.name}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, yaml_file)


def _atomically_write_json(json_file: Path, data: Any) -> None:
    """write json via tmp+flush+fsync+os.replace for durability."""
    json_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_file.parent / f"{json_file.name}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, json_file)


def _has_completed_result(result_file: Path) -> bool:
    """true iff result_file exists with a finite numeric `score` field.

    used by broad/refined to skip ID re-issue on relaunch (idempotency).
    silently swallows malformed JSON / missing score; treats those as "not done".
    """
    if not result_file.exists():
        return False
    try:
        d = json.loads(result_file.read_text())
    except Exception:
        return False
    score = d.get("score")
    return isinstance(score, (int, float)) and math.isfinite(score)


def _append_queue_line(queue_file: Path, method: str, pilot_tag: str,
                       sbatch_cmd: str) -> None:
    """atomically append tab-delimited (method, pilot_tag, sbatch_cmd) to queue.

    fcntl.flock(LOCK_EX) on companion .lock file; write under lock; unlock.
    raises OSError on I/O failure.
    """
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = Path(str(queue_file) + ".lock")
    line = f"{method}\t{pilot_tag}\t{sbatch_cmd}\n"
    with open(lock_file, "a") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            with open(queue_file, "a") as qf:
                qf.write(line)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def _load_recalibrated_spec(exp: str, method: str,
                             output_dir: Path) -> Optional[Dict[str, Tuple]]:
    """load recalibrated_specs/<exp>.yaml[methods][method].

    B13: (exp, method, output_dir) signature.
    B14: outer spec list -> tuple; choice inner list preserved as list.

    wave-3 short-circuit: if DPE_WAVE3=1 is set, skip recalibrated yamls so
    the wave-3 lock applied to METHOD_SPECS at module load is what gets used.

    returns None if file missing or method absent.
    """
    import os as _os
    if _os.environ.get("DPE_WAVE3") == "1":
        return None
    spec_file = output_dir.parent / "recalibrated_specs" / f"{exp}.yaml"
    if not spec_file.exists():
        return None
    data = yaml.safe_load(spec_file.read_text()) or {}
    method_specs = data.get("methods", {}).get(method)
    if not method_specs:
        return None
    result: Dict[str, Tuple] = {}
    for param, spec_raw in method_specs.items():
        if not isinstance(spec_raw, (list, tuple)):
            result[param] = spec_raw
            continue
        kind = spec_raw[0]
        if kind == "choice":
            # B14: preserve inner list; only outer container is tupled
            result[param] = tuple([kind, list(spec_raw[1])])
        else:
            result[param] = tuple(spec_raw)
    return result


def _load_trial_results(results_dir: Path) -> List[Dict[str, Any]]:
    """scan results_dir for trial_*.json; filter valid scores; sort ascending."""
    if not results_dir.exists():
        return []
    trials = []
    for p in results_dir.glob("trial_*.json"):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "score" not in data:
            continue
        score = data["score"]
        if score is None:
            continue
        try:
            float(score)
        except (TypeError, ValueError):
            continue
        trials.append(data)
    return sorted(trials, key=lambda t: float(t["score"]))


def _resolve_stratify_fn(adapter) -> Optional[Callable]:
    """return adapter.stratify_key as a callable if any cell yields non-None;
    else return None (un-stratified path).
    """
    pool = adapter.cell_pool()
    if not pool:
        return None
    if all(adapter.stratify_key(c) is None for c in pool[:5]):
        return None
    return adapter.stratify_key


def _training_M(adapter) -> int:
    """env-var override DPE_TRAINING_M takes precedence over adapter default.

    falls back to adapter.default_training_M(); intended for refined-only mini
    campaigns that want a fixed eval-cell count without editing each adapter.
    """
    v = os.environ.get("DPE_TRAINING_M")
    if v is not None:
        return int(v)
    return adapter.default_training_M()


def _holdout_M(adapter) -> int:
    """same pattern as _training_M for the holdout draw."""
    v = os.environ.get("DPE_HOLDOUT_M")
    if v is not None:
        return int(v)
    return adapter.default_holdout_M()


def _wait_for_trials(results_dir: Path, n: int, timeout_sec: int = 3600,
                     poll_sec: int = 30) -> int:
    """poll results_dir for trial_*.json count; return when >= n or timeout.
    returns observed count. used between stages in --stage all to ensure
    upstream trials finish before downstream reads them. accepts partial
    completion (returns whatever count was reached on timeout).
    """
    deadline = time.time() + timeout_sec
    last_n = -1
    while time.time() < deadline:
        files = [p for p in (results_dir.glob("trial_*.json") if results_dir.exists() else [])
                 if p.stem.removeprefix("trial_").isdigit()]
        cur_n = len(files)
        if cur_n >= n:
            print(f"_wait_for_trials: {results_dir.name} reached {cur_n}/{n}")
            return cur_n
        if cur_n != last_n:
            print(f"_wait_for_trials: {results_dir.name} at {cur_n}/{n}")
            last_n = cur_n
        time.sleep(poll_sec)
    print(f"_wait_for_trials: TIMEOUT {results_dir.name} at {last_n}/{n}")
    return last_n


def _build_sbatch_cmd(experiment: str, method: str, config_file: Path,
                      cells_file: Path, output_dir: Path, stage: str,
                      logdir: Path) -> str:
    """sbatch invocation matching trial_runner CLI contract.

    trial_runner adds <stage>/ subdir to output_dir, so we pass the ROOT
    output_dir (without stage suffix). watchdog expands {time}/{exclude}.
    """
    workdir = "/home/aviamala/dpe-submission"
    trial_id = Path(config_file).stem.removeprefix("trial_")
    job_name = f"{trial_id}_{stage}_{method}_{experiment}"
    return (
        f"sbatch --partition=preempt --time={{time}} --exclude={{exclude}} "
        f"--gpus=1 --cpus-per-task=4 --mem=32G --requeue "
        f"--job-name={job_name} "
        f"--output={logdir}/%j.out "
        f"--wrap=\"set +u && source ~/.bashrc && conda activate fac && set -u && "
        f"export HDF5_USE_FILE_LOCKING=FALSE && cd {workdir} && "
        f"python -m experiments.utils.hpo.trial_runner "
        f"--experiment {experiment} --method {method} "
        f"--config-file {config_file} --eval-cells-file {cells_file} "
        f"--output-dir {output_dir} --stage {stage}\""
    )


# ---------------------------------------------------------------------------
# stage 1: recalibrate
# ---------------------------------------------------------------------------

def recalibrate(method: str, exp: str, output_dir: Path) -> Dict[str, Any]:
    """narrow hyperparameter search space using prior top-10 trials.

    1. validate method in METHOD_SPECS (B8: dict access).
    2. if no winners file or method not in winners, or no broad/refined results:
       return fallback (base_search_space).
    3. collect top-10 trial results from broad+refined dirs.
    4. narrow each param via narrow_spec(base_spec, values).
    5. persist narrowed spec to recalibrated_specs/<exp>.yaml under flock (B7).

    returns {source, spec, num_prior_trials}.
    """
    if method not in METHOD_SPECS:
        raise KeyError(f"unknown method {method}; available: {list(METHOD_SPECS.keys())}")

    base_space: Dict[str, Tuple] = METHOD_SPECS[method]["base_search_space"]  # B8
    fallback = {"source": "fallback", "spec": base_space, "num_prior_trials": 0}

    winners_file = output_dir.parent / f"winners.{exp}.yaml"
    if not winners_file.exists():
        return fallback

    winners = yaml.safe_load(winners_file.read_text()) or {}
    if method not in winners.get("methods", {}):
        return fallback

    broad_dir = output_dir / "broad"
    refined_dir = output_dir / "refined"
    results: List[Dict[str, Any]] = _load_trial_results(broad_dir)
    if refined_dir.exists():
        results += _load_trial_results(refined_dir)

    if not results:
        return fallback

    top_10 = results[:10]
    narrowed: Dict[str, Tuple] = {}
    for param, base_spec in base_space.items():
        values = [t["hyperparams"][param] for t in top_10 if "hyperparams" in t
                  and param in t["hyperparams"]]
        narrowed[param] = narrow_spec(base_spec, values) if values else base_spec

    # B7: tmp+replace inside flock; pid-namespaced tmp
    recal_dir = _ensure_dir(output_dir.parent / "recalibrated_specs")
    recal_file = recal_dir / f"{exp}.yaml"
    lock_path = recal_file.with_suffix(".lock")
    with open(lock_path, "a") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            existing = (yaml.safe_load(recal_file.read_text())
                        if recal_file.exists() else {}) or {}
            existing.setdefault("methods", {})[method] = {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in narrowed.items()
            }
            existing["workflow_version"] = "v1"
            tmp_path = recal_dir / f"{recal_file.name}.tmp.{os.getpid()}"
            with open(tmp_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False, sort_keys=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, recal_file)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)

    return {"source": "winners", "spec": narrowed, "num_prior_trials": len(top_10)}


# ---------------------------------------------------------------------------
# stage 2: broad
# ---------------------------------------------------------------------------

def broad(method: str, exp: str, adapter, n: int = 200, seed: int = 1729,
          output_dir: Path = None, queue_file: Path = None) -> Dict[str, Any]:
    """generate n trial configs from recalibrated (or base) spec; queue sbatch jobs.

    paths (B10):
      config_dir = output_dir/broad/<method>/
      results_dir = output_dir/broad/   <- trial_runner writes trial_*.json here
      cells at output_dir/broad_metadata/cells_seed<seed>.json

    returns {num_trials, config_dir, results_dir, cells_file}.
    """
    config_dir = _ensure_dir(output_dir / "broad" / method)
    results_dir = _ensure_dir(output_dir / "broad")  # B10
    metadata_dir = _ensure_dir(output_dir / "broad_metadata")
    cells_file = metadata_dir / f"cells_seed{seed}.json"

    # load spec (recalibrated or fallback)
    spec = (_load_recalibrated_spec(exp, method, output_dir)
            or METHOD_SPECS[method]["base_search_space"])  # B8

    # draw training cells and persist
    cells = draw_training_sample(
        adapter.cell_pool(), M=_training_M(adapter), seed=seed,
        stratify_fn=_resolve_stratify_fn(adapter))
    cells_manifest = {
        "eval_sample_seed": seed,
        "workflow_version": "v1",
        "cells": [list(c) for c in cells],
    }
    _atomically_write_json(cells_file, cells_manifest)

    # build registry wrapper for gen_config (expects {"search_space": ...})
    registry = {method: {"search_space": spec}}

    # idempotent: skip IDs whose result already has a finite score so we
    # neither overwrite the config nor re-queue the trial on watchdog restart.
    for trial_id in range(n):
        if _has_completed_result(results_dir / f"trial_{trial_id}.json"):
            continue
        config = gen_config(registry, method, trial_id)
        _atomically_write_json(config_dir / f"trial_{trial_id}.json", config)

    logdir = _ensure_dir(output_dir / "logs")
    for trial_id in range(n):
        if _has_completed_result(results_dir / f"trial_{trial_id}.json"):
            continue
        config_file = config_dir / f"trial_{trial_id}.json"
        sbatch_cmd = _build_sbatch_cmd(
            exp, method, config_file, cells_file, output_dir, "broad", logdir)
        _append_queue_line(queue_file, method, "broad", sbatch_cmd)

    return {
        "num_trials": n,
        "config_dir": config_dir,
        "results_dir": results_dir,
        "cells_file": cells_file,
    }


# ---------------------------------------------------------------------------
# stage 3: refined
# ---------------------------------------------------------------------------

def refined(method: str, exp: str, adapter, n: int = 49, seed: int = 1729,
            output_dir: Path = None, queue_file: Path = None) -> Optional[Dict[str, Any]]:
    """narrow from top-5 broad trials and generate refined configs.

    gate (B3, B4):
      - empty broad results -> return None (not raise).
      - r = top1 / median; if median == 0 -> r = 1.0 (flat -> skip).
      - if r >= 0.85 -> return {skipped: True, ...}.

    returns {skipped, num_trials, top1_score, median_score, config_dir, results_dir}
    or None on empty broad.
    """
    broad_dir = output_dir / "broad"
    trial_results = _load_trial_results(broad_dir)

    if not trial_results:
        return None  # B3: None not raise

    # refined-only profile (e.g. refined24) has n=0; skip narrowing entirely
    # so we don't trip narrow_spec on heterogeneous-typed HPs (None vs numeric)
    # arising from mid-campaign spec changes.
    if n == 0:
        return {
            "skipped": True,
            "reason": "refined budget is 0",
            "num_trials": 0,
            "config_dir": None,
            "results_dir": None,
        }

    scores = [float(t["score"]) for t in trial_results]
    top1 = min(scores)
    med = float(median(scores))
    r = top1 / med if med > 0.0 else 1.0  # B4: zero-median -> flat -> skip

    if r >= 0.85:
        return {
            "skipped": True,
            "top1_score": top1,
            "median_score": med,
            "reason": "flat",
            "num_trials": 0,
            "config_dir": None,
            "results_dir": None,
        }

    # narrow from top-5
    top_5 = trial_results[:5]
    base_space: Dict[str, Tuple] = METHOD_SPECS[method]["base_search_space"]  # B8
    narrowed: Dict[str, Tuple] = {}
    for param, base_spec in base_space.items():
        values = [t["hyperparams"][param] for t in top_5 if "hyperparams" in t
                  and param in t["hyperparams"]]
        narrowed[param] = narrow_spec(base_spec, values) if values else base_spec

    config_dir = _ensure_dir(output_dir / "refined" / method)
    results_dir = _ensure_dir(output_dir / "refined")  # B10

    registry = {method: {"search_space": narrowed}}
    # idempotent: skip IDs whose result already has a finite score.
    for trial_id in range(n):
        if _has_completed_result(results_dir / f"trial_{trial_id}.json"):
            continue
        config = gen_config(registry, method, trial_id)
        _atomically_write_json(config_dir / f"trial_{trial_id}.json", config)

    # reuse or regenerate cells
    broad_cells_file = output_dir / "broad_metadata" / f"cells_seed{seed}.json"
    if broad_cells_file.exists():
        cells_raw = json.loads(broad_cells_file.read_text())
        cells = coerce_cells_from_json(cells_raw["cells"])
    else:
        cells = draw_training_sample(
        adapter.cell_pool(), M=_training_M(adapter), seed=seed,
        stratify_fn=_resolve_stratify_fn(adapter))

    refined_meta_dir = _ensure_dir(output_dir / "refined_metadata")
    reused_manifest = {
        "eval_sample_seed": seed,
        "workflow_version": "v1",
        "reused_from": str(broad_cells_file),
        "cells": [list(c) for c in cells],
    }
    _atomically_write_json(refined_meta_dir / f"cells_reused_seed{seed}.json",
                           reused_manifest)

    logdir = _ensure_dir(output_dir / "logs")
    for trial_id in range(n):
        if _has_completed_result(results_dir / f"trial_{trial_id}.json"):
            continue
        config_file = config_dir / f"trial_{trial_id}.json"
        sbatch_cmd = _build_sbatch_cmd(
            exp, method, config_file, broad_cells_file, output_dir, "refined", logdir)
        _append_queue_line(queue_file, method, "refined", sbatch_cmd)

    return {
        "skipped": False,
        "num_trials": n,
        "top1_score": top1,
        "median_score": med,
        "config_dir": config_dir,
        "results_dir": results_dir,
    }


# ---------------------------------------------------------------------------
# stage 4: holdout
# ---------------------------------------------------------------------------

def holdout(method: str, exp: str, adapter, output_dir: Path = None,
            queue_file: Path = None) -> Dict[str, Any]:
    """select winner from broad+refined; draw holdout cells; queue one sbatch job.

    winner = argmin(score) across all valid broad+refined trial JSONs.
    source_stage read from pilot_variant field (B9: not pilot_tag).
    config and cells written to output_dir/holdout/ (B10).

    returns {winner_trial_id, winner_score, source_stage, config_file, holdout_cells_file}.
    raises ValueError if no trial results found.
    """
    broad_dir = output_dir / "broad"
    refined_dir = output_dir / "refined"
    all_results = _load_trial_results(broad_dir)
    if refined_dir.exists():
        all_results += _load_trial_results(refined_dir)

    if not all_results:
        raise ValueError("no trial results to select winner")

    winner = all_results[0]  # ascending sort -> argmin
    winner_trial_id = winner["trial_id"]
    winner_score = float(winner["score"])
    source_stage = winner.get("pilot_variant", "broad")  # B9: pilot_variant field

    # load training cells for exclusion
    broad_cells_file = output_dir / "broad_metadata" / "cells_seed1729.json"
    if broad_cells_file.exists():
        cells_raw = json.loads(broad_cells_file.read_text())
        training_cells = coerce_cells_from_json(cells_raw["cells"])
    else:
        training_cells = []

    # draw holdout cells (clamped)
    pool = adapter.cell_pool()
    M = _holdout_M(adapter)
    holdout_cells, actual_M = draw_holdout_sample_clamped(
        pool, exclude=training_cells, M=M, seed=4096
    )

    # write winner config to B10 path
    config_dir = _ensure_dir(output_dir / "holdout")
    winner_config = {
        "trial_id": 0,
        "method": method,
        "hyperparams": winner["hyperparams"],
        "_meta": {
            "source_trial_id": winner_trial_id,
            "source_stage": source_stage,
            "training_score": winner_score,
        },
    }
    config_file = config_dir / "trial_0_winner.json"
    _atomically_write_json(config_file, winner_config)

    # write holdout cells manifest
    holdout_meta_dir = _ensure_dir(output_dir / "holdout_metadata")
    holdout_cells_file = holdout_meta_dir / "cells_seed4096.json"
    cells_manifest = {
        "eval_sample_seed": 4096,
        "workflow_version": "v1",
        "cells": [list(c) for c in holdout_cells],
    }
    _atomically_write_json(holdout_cells_file, cells_manifest)

    logdir = _ensure_dir(output_dir / "logs")
    sbatch_cmd = _build_sbatch_cmd(
        exp, method, config_file, holdout_cells_file, output_dir, "holdout", logdir)
    _append_queue_line(queue_file, method, "holdout", sbatch_cmd)

    return {
        "winner_trial_id": winner_trial_id,
        "winner_score": winner_score,
        "source_stage": source_stage,
        "config_file": config_file,
        "holdout_cells_file": holdout_cells_file,
    }


# ---------------------------------------------------------------------------
# stage 5: persist
# ---------------------------------------------------------------------------

N_ALPHAS = 4


def persist(method: str, exp: str, holdout_result_file: Path, output_dir: Path,
            training_result_file: Optional[Path] = None) -> Dict[str, Any]:
    """read holdout + training results; merge into winners.<exp>[suffix].yaml.

    B2: writes both new-schema winners["methods"][method] and legacy
        winners[method][alpha_idx] for downstream readers.
    M3: output_dir explicit; no .parent chain.

    env var `DPE_WINNERS_SUFFIX` (e.g. ".refined24") is inserted between the
    experiment name and the .yaml extension so a parallel mini-campaign does
    not clobber the canonical winners file. unset -> default winners.<exp>.yaml.

    returns {winners_file, entry_written}.
    """
    suffix = os.environ.get("DPE_WINNERS_SUFFIX", "")
    winners_file = output_dir.parent / f"winners.{exp}{suffix}.yaml"  # M3

    holdout_result = json.loads(holdout_result_file.read_text())
    holdout_score = float(holdout_result["score"])

    # infer training result file from _meta if not provided
    if training_result_file is None:
        meta = holdout_result.get("_meta", {})
        source_trial_id = meta["source_trial_id"]
        source_stage = meta["source_stage"]
        training_result_file = output_dir / source_stage / f"trial_{source_trial_id}.json"  # M3 + B10
    else:
        meta = holdout_result.get("_meta", {})
        source_stage = meta.get("source_stage", "broad")
        source_trial_id = meta.get("source_trial_id", 0)

    training_result = json.loads(training_result_file.read_text())
    training_score = float(training_result["score"])
    training_trial_id = training_result.get("trial_id", source_trial_id)
    training_stage = source_stage

    entry: Dict[str, Any] = {
        "hyperparams": holdout_result["hyperparams"],
        "score": {
            "training_stage": training_stage,
            "training_score": training_score,
            "training_trial_id": training_trial_id,
            "training_seed": 1729,
            "holdout_score": holdout_score,
            "holdout_seed": 4096,
        },
    }

    # load or init winners
    winners: Dict[str, Any] = {}
    if winners_file.exists():
        winners = yaml.safe_load(winners_file.read_text()) or {}

    # new-schema write
    winners.setdefault("methods", {})[method] = entry

    # B2: legacy shim for step2_run_algorithms.py (reads winners[method][alpha_idx])
    for alpha_idx in range(N_ALPHAS):
        winners.setdefault(method, {})[alpha_idx] = {
            "hyperparams": entry["hyperparams"],
            "mae_median": float(entry["score"]["holdout_score"]),
            "source": entry["score"]["training_stage"],
            "trial_id": entry["score"]["training_trial_id"],
        }

    if "provenance" not in winners:
        winners["provenance"] = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "workflow_version": "v1",
        }

    _atomically_write_yaml(winners_file, winners)
    return {"winners_file": winners_file, "entry_written": True}


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------

def main() -> int:
    """hpo workflow orchestrator.

    --stage all chains recalibrate -> broad -> refined -> holdout via afterok
    sbatch deps (B11). PERSIST MUST BE INVOKED SEPARATELY by the user or by
    launcher.py with afterok dep on holdout. this avoids the workflow cpu job
    needing to wait on holdout trial completion.

    M1: asserts stage_budget("broad") + stage_budget("refined") +
        stage_budget("holdout") == args.budget at startup.
    """
    parser = argparse.ArgumentParser(description="HPO workflow orchestrator: 5 stages")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--stage", type=str, default="all",
                        choices=["recalibrate", "broad", "refined",
                                 "holdout", "persist", "all"])
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--queue-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    # persist-specific args
    parser.add_argument("--holdout-result-file", type=Path, default=None)
    parser.add_argument("--training-result-file", type=Path, default=None)
    args = parser.parse_args()

    # M1: budget sanity check
    assert (stage_budget("broad") + stage_budget("refined") + stage_budget("holdout")
            == args.budget), \
        (f"budget mismatch: stage sum "
         f"{stage_budget('broad') + stage_budget('refined') + stage_budget('holdout')} "
         f"!= {args.budget}")

    if args.method not in METHOD_SPECS:
        print(f"error: unknown method {args.method}; available: {list(METHOD_SPECS.keys())}",
              file=sys.stderr)
        return 1

    # resolve output_dir
    if args.output_dir is None:
        data_root = os.environ.get("DPE_DATA_ROOT")
        if not data_root:
            print("error: --output-dir not set and DPE_DATA_ROOT not in environ",
                  file=sys.stderr)
            return 1
        args.output_dir = Path(data_root) / args.experiment / args.method

    if args.queue_file is None:
        args.queue_file = args.output_dir.parent / "watchdog_queue.txt"

    # load adapter (only needed for stages that draw cells)
    adapter = None
    if args.stage in {"broad", "refined", "holdout", "all"}:
        if get_adapter is None:
            print("error: experiments.utils.hpo.adapters not available", file=sys.stderr)
            return 1
        try:
            adapter = get_adapter(args.experiment)
        except (ImportError, KeyError) as e:
            print(f"error: adapter for {args.experiment} not found: {e}", file=sys.stderr)
            return 1
        if not adapter.is_ready():
            print(f"warning: adapter for {args.experiment} not ready; skipping")
            return 0

    try:
        if args.stage == "all":
            recalibrate(args.method, args.experiment, args.output_dir)
            broad_result = broad(args.method, args.experiment, adapter, n=stage_budget("broad"),
                                 seed=args.seed, output_dir=args.output_dir, queue_file=args.queue_file)
            # poll for broad trials to finish before refined reads them (B12 fix)
            _wait_for_trials(args.output_dir / "broad", n=stage_budget("broad"),
                             timeout_sec=4*3600, poll_sec=30)
            refined_result = refined(args.method, args.experiment, adapter,
                                     n=stage_budget("refined"), seed=args.seed,
                                     output_dir=args.output_dir,
                                     queue_file=args.queue_file)
            if refined_result and not refined_result.get("skipped"):
                _wait_for_trials(args.output_dir / "refined", n=stage_budget("refined"),
                                 timeout_sec=2*3600, poll_sec=30)
            holdout(args.method, args.experiment, adapter,
                    output_dir=args.output_dir, queue_file=args.queue_file)
            _wait_for_trials(args.output_dir / "holdout", n=1, timeout_sec=3600, poll_sec=30)
            # persist auto-runs in --stage all once holdout result is on disk
            holdout_files = list((args.output_dir / "holdout").glob("trial_*.json"))
            if holdout_files:
                persist(args.method, args.experiment, holdout_files[0], args.output_dir)
                print(f"--stage all: winners.{args.experiment}.yaml written")
            else:
                print("--stage all: holdout result missing; skipping persist", file=sys.stderr)

        elif args.stage == "recalibrate":
            result = recalibrate(args.method, args.experiment, args.output_dir)
            print(json.dumps({k: str(v) if isinstance(v, Path) else v
                               for k, v in result.items() if k != "spec"}))

        elif args.stage == "broad":
            result = broad(args.method, args.experiment, adapter,
                           n=stage_budget("broad"), seed=args.seed,
                           output_dir=args.output_dir, queue_file=args.queue_file)
            print(json.dumps({k: str(v) for k, v in result.items()}))

        elif args.stage == "refined":
            result = refined(args.method, args.experiment, adapter,
                             n=stage_budget("refined"), seed=args.seed,
                             output_dir=args.output_dir, queue_file=args.queue_file)
            print(json.dumps({k: str(v) if isinstance(v, Path) else v
                               for k, v in (result or {}).items()}))

        elif args.stage == "holdout":
            result = holdout(args.method, args.experiment, adapter,
                             output_dir=args.output_dir, queue_file=args.queue_file)
            print(json.dumps({k: str(v) if isinstance(v, Path) else v
                               for k, v in result.items()}))

        elif args.stage == "persist":
            if args.holdout_result_file is None:
                print("error: --holdout-result-file required for persist stage",
                      file=sys.stderr)
                return 1
            result = persist(args.method, args.experiment, args.holdout_result_file,
                             args.output_dir,
                             training_result_file=args.training_result_file)
            print(json.dumps({k: str(v) if isinstance(v, Path) else v
                               for k, v in result.items()}))

    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"stage {args.stage} completed for {args.method}/{args.experiment}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
