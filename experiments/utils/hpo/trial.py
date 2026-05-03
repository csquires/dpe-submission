"""shared hpo trial runner.

per-experiment hpo_trial.py supplies an `eval_cell(cell) -> metric` closure that
loads data, builds an estimator, fits, predicts, and returns one scalar metric
per cell. this module owns the cell-iteration loop: per-cell seeding,
exception/NaN skipping, aggregation, atomic JSON write.

contract:
  cell is a tuple of ints (the eval-cell coordinates). cell shape is
  experiment-specific: 1-tuple for eig (design_idx,), 2-tuple for elbo
  (alpha_idx, design_idx), 3-tuple for smodice/pendulum (k1, k2, seed).

  eval_cell raises (FileNotFoundError, ValueError, RuntimeError, ...) or returns
  a non-finite float to signal a per-cell skip; we log and continue.
"""

import json
import logging
import math
import os
import random
import time
from typing import Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


def parse_cells(s: str) -> list[tuple[int, ...]]:
    """parse 'a:b:c,d:e:f' or '0,40,80' into [(a,b,c), ...] / [(0,), (40,), (80,)]."""
    return [tuple(int(x) for x in cell.split(":")) for cell in s.split(",")]


def cell_id(cell: tuple[int, ...]) -> str:
    """inverse: (0,1,2) -> '0:1:2', (5,) -> '5'."""
    return ":".join(str(x) for x in cell)


def run_trial(
    *,
    experiment: str,
    method: str,
    trial_id: int,
    hyperparams: dict,
    eval_cells: list[tuple[int, ...]],
    eval_cell: Callable[[tuple[int, ...]], float],
    output_dir: str,
    metric_key: str = "per_cell_metric",
) -> dict:
    """run one trial: per-cell seed -> eval_cell -> aggregate -> atomic-write json.

    args:
        experiment: tag for the seed hash (e.g. 'eig', 'elbo'); avoids cross-experiment
            collisions when the same trial_id is reused.
        method, trial_id, hyperparams: copied verbatim into the result json.
        eval_cells: list of int-tuples; the loop iterates these in order.
        eval_cell: closure mapping cell -> scalar metric. raise or return non-finite to skip.
        output_dir: written as <output_dir>/trial_<trial_id>.json (created if missing).
        metric_key: name of the per-cell metric dict in the result json
            (e.g. 'per_design_eig_abs_err', 'per_cell_ldr_mae').

    returns: the result dict that was written to disk.

    note: per-cell seeding covers torch, numpy, and python random; subprocess workers
        inherit the parent's PRNG state but may diverge if they fork. for distributed
        multi-worker reproducibility, seed workers explicitly in their initialization code.
    """
    per_cell: dict[str, float] = {}
    t0 = time.perf_counter()

    for cell in eval_cells:
        cs = cell_id(cell)
        seed_int = hash((experiment, trial_id, cs)) & 0xFFFFFFFF
        torch.manual_seed(seed_int)
        np.random.seed(seed_int)
        random.seed(seed_int)
        try:
            metric = eval_cell(cell)
        except FileNotFoundError as e:
            logger.warning("cell %s: data missing: %s", cs, e)
            continue
        except (RuntimeError, OSError, ImportError, MemoryError):
            # infrastructure failure (cuda init, broken nfs, oom): propagate
            # so the process exits non-zero, slurm marks failed, orphan scan
            # re-queues. distinct from numerical divergence (returns non-finite).
            raise
        except Exception as e:
            logger.warning("cell %s: %s: %s", cs, type(e).__name__, e)
            continue
        if metric is None or not math.isfinite(float(metric)):
            logger.warning("cell %s: non-finite metric (%s); skipping", cs, metric)
            continue
        per_cell[cs] = float(metric)
        logger.info("cell %s: %.6f", cs, metric)

    elapsed = time.perf_counter() - t0
    mean_metric = float(np.mean(list(per_cell.values()))) if per_cell else float("nan")

    result = {
        "method": method,
        "trial_id": trial_id,
        "hyperparams": hyperparams,
        metric_key: per_cell,
        "mean_metric": mean_metric,
        "elapsed_seconds": elapsed,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/trial_{trial_id}.json"
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)

    logger.info(
        "trial %s done: %d/%d cells, mean=%.6f, %.2fs -> %s",
        trial_id, len(per_cell), len(eval_cells), mean_metric, elapsed, out_path,
    )
    return result
