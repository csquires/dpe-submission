import random
from collections import defaultdict
from logging import getLogger
from typing import Callable, Hashable, List, Optional, Tuple

logger = getLogger(__name__)

DEFAULT_TRAINING_SEED: int = 1729
DEFAULT_HOLDOUT_SEED: int = 4096
DEFAULT_M: int = 32


def draw_training_sample(
    pool: List[Tuple],
    M: int = DEFAULT_M,
    seed: int = DEFAULT_TRAINING_SEED,
    stratify_fn: Optional[Callable[[Tuple], Hashable]] = None,
) -> List[Tuple]:
    """draw M cells without replacement from pool. silently dedup via set.

    if stratify_fn is None: un-stratified random sample (current default).
    if stratify_fn is provided: groups pool by stratify_fn(cell) and samples
    K_per_stratum = max(1, M // n_groups) cells per group. total returned ≈
    n_groups * K_per_stratum (may exceed M if M doesn't divide n_groups evenly;
    we accept this rather than dropping cells from any stratum).

    args:
      pool: list of cell tuples.
      M: sample size (>= 1). becomes a target rather than exact under stratification.
      seed: RNG seed for reproducibility (per-stratum RNG derived deterministically).
      stratify_fn: optional callable cell -> hashable group key. None for v1 behavior.

    returns:
      list of cell tuples in draw order (un-stratified) or grouped order
      (stratified: all cells from group_0, then group_1, etc., sorted by group key).
    """
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    pool_set = set(pool)
    if len(pool_set) == 0:
        raise ValueError("pool is empty after dedup")
    if stratify_fn is None:
        if len(pool_set) < M:
            raise ValueError(
                f"need M={M} cells, only {len(pool_set)} distinct in pool")
        pool_list = sorted(pool_set)
        rng = random.Random(seed)
        return rng.sample(pool_list, M)
    # stratified path
    groups: dict = defaultdict(list)
    for cell in pool_set:
        groups[stratify_fn(cell)].append(cell)
    n_groups = len(groups)
    K_per = max(1, M // n_groups)
    sample: List[Tuple] = []
    for g_key in sorted(groups.keys()):
        g_cells = sorted(groups[g_key])
        if len(g_cells) < K_per:
            raise ValueError(
                f"stratum {g_key!r} has only {len(g_cells)} cells; need K_per={K_per}")
        # derive per-stratum seed deterministically from base seed + group key hash
        g_seed = seed ^ (hash(g_key) & 0xFFFFFFFF)
        rng = random.Random(g_seed)
        sample.extend(rng.sample(g_cells, K_per))
    if len(sample) != n_groups * K_per:
        # defensive: should not happen
        raise RuntimeError(f"stratified draw produced {len(sample)} cells, expected {n_groups * K_per}")
    if len(sample) != M:
        logger.info(
            f"stratified sample size {len(sample)} (n_groups={n_groups} × K_per={K_per}) "
            f"differs from requested M={M}")
    return sample


def draw_holdout_sample(
    pool: List[Tuple],
    exclude: List[Tuple],
    M: int = DEFAULT_M,
    seed: int = DEFAULT_HOLDOUT_SEED
) -> List[Tuple]:
    """sample M cells from dedup(pool) - set(map(tuple, exclude)). silently ignore
    excluded cells not in pool. raise ValueError if |avail| < M or M < 1.

    args:
      pool: list of all candidate cell tuples.
      exclude: list of cells to exclude (may contain cells not in pool).
      M: sample size (>= 1).
      seed: RNG seed.

    returns:
      M-element list of tuples disjoint from exclude, in draw order.
    """
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    pool_set = set(pool)
    excl_set = set(map(tuple, exclude))
    avail_set = pool_set - excl_set
    if len(avail_set) < M:
        raise ValueError(f"need M={M} cells, only {len(avail_set)} available after excluding {len(excl_set)} cells")
    avail_list = sorted(avail_set)
    rng = random.Random(seed)
    sample = rng.sample(avail_list, M)
    return sample


def draw_holdout_sample_clamped(
    pool: List[Tuple],
    exclude: List[Tuple],
    M: int = DEFAULT_M,
    seed: int = DEFAULT_HOLDOUT_SEED
) -> Tuple[List[Tuple], int]:
    """draw holdout but clamp M if pool insufficient. returns (cells, actual_M).
    logs warning if actual_M < requested M. returns ([], 0) if nothing available.

    args:
      pool: list of all candidate cell tuples.
      exclude: list of cells to exclude.
      M: requested sample size.
      seed: RNG seed.

    returns:
      (cells, actual_M): cells is list of tuples disjoint from exclude; actual_M <= M.
    """
    available = list(set(map(tuple, pool)) - set(map(tuple, exclude)))
    actual_M = min(M, len(available))
    if actual_M < M:
        logger.warning(f"holdout pool clamped: requested {M}, available {actual_M}")
    if actual_M == 0:
        return [], 0
    rng = random.Random(seed)
    return rng.sample(available, actual_M), actual_M


def coerce_cells_from_json(loaded: List) -> List[Tuple]:
    """canonicalize JSON-loaded cells: [[0,0],[1,2]] -> [(0,0), (1,2)].
    handles arity 1, 2, 3 cells (recursive via tuple()). used at every JSON
    read boundary by trial_runner, workflow, launcher.

    args:
      loaded: list of lists (from json.load).

    returns:
      list of tuples, same length as loaded.
    """
    return [tuple(cell) for cell in loaded]
