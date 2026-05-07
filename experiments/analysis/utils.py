"""Shared utilities for loading trials, splitting cells, and neighbourhood search."""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Trial filtering
# ---------------------------------------------------------------------------

# VFM / CTSM / FMDRE family (basic or Triangular prefix, including FMDRE_S2)
_VCF_RE = re.compile(r"^(Triangular)?(VFM|CTSM|FMDRE)", re.IGNORECASE)


def filter_trials(trials: list[dict], method: str) -> list[dict]:
    """Drop trials that exceed compute-budget thresholds.

    Rule — VFM / CTSM / FMDRE family (basic or Triangular, incl. FMDRE_S2):
        drop if hyperparams['integration_steps'] > 1750
    """
    if not _VCF_RE.match(method):
        return trials
    out = []
    for t in trials:
        steps = t.get("hyperparams", {}).get("integration_steps")
        if steps is not None and steps > 1750:
            continue
        out.append(t)
    return out


def filter_trials_adaptive(trials: list[dict], method: str) -> list[dict]:
    """Like filter_trials, but falls back to unfiltered if filter empties the pool."""
    filtered = filter_trials(trials, method)
    return filtered if filtered else trials


# ---------------------------------------------------------------------------
# Trial loading
# ---------------------------------------------------------------------------

def load_trials(method_dir: Path, variant: str) -> list[dict]:
    """Load all trial_*.json from *method_dir*/*variant*/.

    Skips files that are malformed, have non-finite score, or are dummy runs.
    """
    results_dir = method_dir / variant
    if not results_dir.exists():
        return []
    trials: list[dict] = []
    for f in sorted(results_dir.glob("trial_*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("_dummy", False):
            continue
        score = d.get("score")
        if score is None or not math.isfinite(float(score)):
            continue
        trials.append(d)
    return trials


def is_method_dir(path: Path) -> bool:
    """True if *path* is a method results directory (has broad/ or refined/)."""
    return path.is_dir() and ((path / "broad").exists() or (path / "refined").exists())


# ---------------------------------------------------------------------------
# Cell handling
# ---------------------------------------------------------------------------

def cell_to_key(cell: tuple | list) -> str:
    """(0, 1, 4) → '0:1:4'."""
    return ":".join(str(x) for x in cell)


def split_cells(
    cells: list[tuple],
    n_train: int,
    seed: int = 42,
) -> tuple[list[tuple], list[tuple]]:
    """Deterministically split *cells* into (train_cells, holdout_cells).

    The split is reproducible via *seed* and stable across Python versions
    because we sort before shuffling.
    """
    cells_sorted = sorted(tuple(c) for c in cells)
    rng = random.Random(seed)
    shuffled = cells_sorted[:]
    rng.shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:]


def metric_on_cells(
    trial: dict,
    cells: list[tuple],
    metric_key: str,
) -> float | None:
    """Mean per-cell metric for *trial* restricted to *cells*.

    Returns None if any requested cell is missing from the trial or has a
    non-finite value.
    """
    scores_dict: dict[str, float] = trial.get(metric_key, {})
    vals: list[float] = []
    for c in cells:
        key = cell_to_key(c)
        if key not in scores_dict:
            return None
        v = scores_dict[key]
        if not math.isfinite(v):
            return None
        vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


def variance_on_cells(
    trial: dict,
    cells: list[tuple],
    metric_key: str,
) -> float | None:
    """Population std of per-cell metric values for *trial* on *cells*.

    Returns None if any cell is missing or non-finite.  Returns 0.0 for a
    single-cell evaluation (variance is trivially zero).
    """
    scores_dict: dict[str, float] = trial.get(metric_key, {})
    vals: list[float] = []
    for c in cells:
        key = cell_to_key(c)
        if key not in scores_dict:
            return None
        v = scores_dict[key]
        if not math.isfinite(v):
            return None
        vals.append(v)
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))


def rank_pool(
    pool: list[dict],
    cells: list[tuple],
    metric_key: str,
) -> list[tuple[float, float, dict]]:
    """Score each trial by average per-cell MAE rank across *cells*.

    For each cell, all trials in *pool* are ranked by their MAE (rank 1 =
    lowest MAE = best).  Each trial's average rank across all cells is its
    score.  Trials missing any cell are excluded.

    Returns list of (avg_rank, mean_mae, trial) sorted by avg_rank ascending.
    mean_mae is included so callers can still report absolute MAE values.
    """
    # keep only trials that have finite scores for all cells
    complete: list[tuple[dict[str, float], float, dict]] = []
    for t in pool:
        scores_dict = t.get(metric_key, {})
        vals: dict[str, float] = {}
        ok = True
        for c in cells:
            key = cell_to_key(c)
            v = scores_dict.get(key)
            if v is None or not math.isfinite(v):
                ok = False
                break
            vals[key] = v
        if ok:
            mean = sum(vals.values()) / len(vals)
            complete.append((vals, mean, t))

    if not complete:
        return []

    cell_keys = [cell_to_key(c) for c in cells]
    n = len(complete)
    rank_sums = [0.0] * n

    for ck in cell_keys:
        indexed = sorted(range(n), key=lambda i: complete[i][0][ck])
        for rank, idx in enumerate(indexed, 1):
            rank_sums[idx] += rank

    n_cells = len(cell_keys)
    result = [
        (rank_sums[i] / n_cells, complete[i][1], complete[i][2])
        for i in range(n)
    ]
    result.sort(key=lambda x: x[0])
    return result


def dominant_categoricals(trials: list[dict], threshold: float = 0.50) -> dict:
    """Return HPs whose single value appears in > *threshold* fraction of trials.

    Checks all hyperparameter keys.  For each key, if one value accounts for
    more than *threshold* of the trials that have that key, it is reported.

    Returns {hp_name: {"value": v, "fraction": f, "count": n, "total": t}}.
    """
    if not trials:
        return {}
    from collections import Counter
    # gather values per HP key
    hp_vals: dict[str, list] = {}
    for t in trials:
        for k, v in t.get("hyperparams", {}).items():
            hp_vals.setdefault(k, []).append(v)
    result: dict = {}
    for k, vals in hp_vals.items():
        counts = Counter(vals)
        top_val, top_cnt = counts.most_common(1)[0]
        frac = top_cnt / len(vals)
        if frac > threshold:
            result[k] = {
                "value": top_val,
                "fraction": round(frac, 3),
                "count": top_cnt,
                "total": len(vals),
            }
    return result


# ---------------------------------------------------------------------------
# Hyperparameter distance and neighbourhood
# ---------------------------------------------------------------------------

def _rel_diff(a: float, b: float, eps: float = 1e-8) -> float:
    """Symmetric relative difference: |a−b| / max(|a|, |b|, eps)."""
    return abs(a - b) / max(abs(a), abs(b), eps)


def hp_distance(hp1: dict, hp2: dict) -> float:
    """Max relative difference across all shared numeric hyperparams.

    Returns inf if key sets differ or any categorical value mismatches.
    """
    if set(hp1.keys()) != set(hp2.keys()):
        return float("inf")
    dists: list[float] = []
    for k in hp1:
        v1, v2 = hp1[k], hp2[k]
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            dists.append(_rel_diff(float(v1), float(v2)))
        else:
            if v1 != v2:
                return float("inf")   # categorical mismatch
    return max(dists) if dists else 0.0


def find_neighbors(
    candidates: list[dict],
    reference_hps: list[dict],
    tol: float,
) -> list[dict]:
    """Return candidates whose hyperparams are within *tol* of ANY reference."""
    return [
        t for t in candidates
        if any(hp_distance(t["hyperparams"], ref) <= tol for ref in reference_hps)
    ]


def neighbor_search(
    top_trials: list[dict],
    candidate_pool: list[dict],
    start_tol: float = 0.10,
    step: float = 0.10,
    max_tol: float = 0.60,
    min_matches: int = 3,
) -> tuple[list[dict], float]:
    """Find neighbours of *top_trials* in *candidate_pool*.

    Starts at *start_tol* and relaxes by *step* until ≥ *min_matches* found
    or *max_tol* is reached.

    Returns (matched_trials, tolerance_used).
    """
    ref_hps = [t["hyperparams"] for t in top_trials]
    tol = start_tol
    matched: list[dict] = []
    while tol <= max_tol + 1e-9:
        matched = find_neighbors(candidate_pool, ref_hps, tol)
        if len(matched) >= min_matches:
            return matched, round(tol, 10)
        tol = round(tol + step, 10)
    return matched, round(min(tol - step, max_tol), 10)
