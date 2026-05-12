"""
cell-level stratified train/holdout split utilities.

deterministic, deduplicated splitting with optional stratification by key.
handles cell-level splits only; within-cell sample splits are in eval_split.py.
"""
import logging
import random
from typing import Callable, Hashable


def stratified_split(
    pool: list[tuple],
    stratify_fn: Callable[[tuple], Hashable | None] | None,
    train_ratio: float,
    seed: int,
) -> tuple[list[tuple], list[tuple]]:
    """
    split pool into train and holdout, optionally stratified by stratify_fn.

    if stratify_fn is None or all keys are None:
        do unstratified random split
    else:
        group cells by stratify_fn(cell)
        for each group:
            shuffle and split at int(train_ratio * len(group))
        concatenate per-group results

    return (sorted(train), sorted(holdout))

    Args:
        pool: list of hashable tuples (cells). duplicates are deduplicated.
        stratify_fn: optional callable mapping cell → hashable key or None.
                     if None, performs unstratified split.
                     if all return None, falls back to unstratified.
        train_ratio: float in (0, 1). fraction allocated to train.
        seed: int. master seed for deterministic RNG.

    Returns:
        (train_cells, holdout_cells): tuple of sorted lists, disjoint,
        union equals deduplicated pool.

    Raises:
        ValueError: if pool is empty, train_ratio out of bounds,
                    or mix of None and non-None keys.
    """
    # validate train_ratio bounds
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    # validate and deduplicate
    if not pool:
        raise ValueError("pool cannot be empty")

    pool_sorted = sorted(set(pool))

    # early exit: no stratification (stratify_fn is None)
    if stratify_fn is None:
        rng = random.Random(seed)
        n_train = max(1, int(len(pool_sorted) * train_ratio))
        train = sorted(rng.sample(pool_sorted, n_train))
        holdout = sorted(set(pool_sorted) - set(train))
        return (train, holdout)

    # stratify: group cells by key
    strata = {}
    none_stratum = []
    for cell in pool_sorted:
        key = stratify_fn(cell)
        if key is None:
            none_stratum.append(cell)
        else:
            if key not in strata:
                strata[key] = []
            strata[key].append(cell)

    # if all keys are None, fall back to unstratified mode
    if not strata:
        rng = random.Random(seed)
        n_train = max(1, int(len(none_stratum) * train_ratio))
        train = sorted(rng.sample(none_stratum, n_train))
        holdout = sorted(set(none_stratum) - set(train))
        return (train, holdout)

    # check for mix of None and non-None: raise error (all-or-nothing)
    if none_stratum and strata:
        raise ValueError(
            "stratify_fn returned mix of None and non-None keys; "
            "require all cells to be stratifiable or all unstratifiable"
        )

    # split each non-None stratum deterministically
    train_all = []
    holdout_all = []

    logger = logging.getLogger(__name__)

    for key in sorted(strata.keys()):
        cells_in_group = strata[key]
        n_group = len(cells_in_group)
        n_train_group = max(1, int(n_group * train_ratio))

        # seed per-stratum RNG: seed XOR (hash(key) & 0xFFFFFFFF)
        per_group_seed = seed ^ (hash(key) & 0xFFFFFFFF)
        rng_group = random.Random(per_group_seed)

        if n_train_group >= n_group:
            # log warning: entire stratum goes to train
            logger.warning(
                f"stratum key {key!r}: group size {n_group} yields "
                f"n_train {n_train_group}; assigning all to train"
            )
            train_all.extend(cells_in_group)
        else:
            # sample without replacement
            train_subset = sorted(rng_group.sample(cells_in_group, n_train_group))
            holdout_subset = sorted(set(cells_in_group) - set(train_subset))
            train_all.extend(train_subset)
            holdout_all.extend(holdout_subset)

    return (sorted(train_all), sorted(holdout_all))


def validate_disjoint_union(train: list, holdout: list, full_pool: list) -> None:
    """
    validate that train and holdout are disjoint and their union equals full_pool.

    check: train ∩ holdout = ∅
    check: train ∪ holdout = full_pool (as sets)
    raise ValueError on mismatch.

    Args:
        train: list of cells in train split.
        holdout: list of cells in holdout split.
        full_pool: list of all cells.

    Raises:
        ValueError: if sets are not disjoint or union does not match pool.

    Returns:
        None on success (no-op).
    """
    train_set = set(train)
    holdout_set = set(holdout)
    pool_set = set(full_pool)

    # check disjointness
    intersection = train_set & holdout_set
    if intersection:
        raise ValueError(f"train and holdout are not disjoint; overlap: {intersection}")

    # check union coverage
    union = train_set | holdout_set
    if union != pool_set:
        missing = pool_set - union
        extra = union - pool_set
        raise ValueError(
            f"union of train and holdout != pool. missing: {missing}, extra: {extra}"
        )
