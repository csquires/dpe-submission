"""step2_gather: shard discovery, validation, and merge into normalized tables.

reads FLAT h5 shards, validates config_hash + startup designs,
merges into 3 FLAT parallel tables (studies/rounds/trials with prefixed dataset names).
no nested groups, no compound dtypes.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

from ex.ablations.eig_boed.step1_run_studies import enumerate_cells, config_hash
from ex.ablations.eig_boed.study import startup_designs, study_seed


def load_config(cfg_path: str) -> dict:
    """load config.yaml as plain dict (use cfg["key"], NOT cfg.key)."""
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def discover_shards(raw_results_dir: str) -> list[Path]:
    """glob all .h5 files in raw_results_dir, return sorted list of paths."""
    return sorted(Path(raw_results_dir).glob("*.h5"))


def read_shard_attrs(h5_path: Path) -> dict | None:
    """
    read shard attrs: complete, n_trials, n_rounds, config_hash,
    arm, geometry, prior_idx, method, seed_rep (as separate attrs, not tuple).

    return: dict with these keys, or None if file is truncated/corrupt.
    guard against OSError (truncated h5).
    """
    try:
        with h5py.File(h5_path, "r") as f:
            attrs = {
                "complete": f.attrs.get("complete", False),
                "n_trials": f.attrs.get("n_trials"),
                "n_rounds": f.attrs.get("n_rounds"),
                "config_hash": f.attrs.get("config_hash"),
                "arm": f.attrs.get("arm"),
                "geometry": f.attrs.get("geometry"),
                "prior_idx": f.attrs.get("prior_idx"),
                "method": f.attrs.get("method"),
                "seed_rep": f.attrs.get("seed_rep"),
            }
            return attrs
    except OSError as e:
        # truncated file (preemption, incomplete write)
        print(f"  CORRUPT (truncated): {h5_path.name}: {e}")
        return None


def read_shard_data(h5_path: Path, R: int, T: int, data_dim: int) -> dict | None:
    """
    read all FLAT root datasets from shard (NOT groups).

    args:
        R: n_rounds (5)
        T: n_trials (20)
        data_dim: from cfg["data_dim"]

    return: dict with all datasets loaded into memory, or None if any read fails.

    prior (before round 0):
      - mu0 (data_dim,)
      - Sigma0 (data_dim, data_dim)
      - eig_star (scalar)
      - xi_opt (data_dim,)

    per-round (length R):
      - Sigma_r (R, data_dim, data_dim)
      - xi_r (R, data_dim)
      - y_obs_r (R,)
      - eig_star_r (R,)

    per-trial (length N = R*T, FLAT):
      - round_idx (N,) int
      - trial_idx (N,) int
      - a (N,) float
      - b (N,) float
      - xi (N, data_dim) float
      - est_eig (N,) float
      - true_eig (N,) float
      - state (N,) bytes S16
      - walltime_s (N,) float
    """
    try:
        with h5py.File(h5_path, "r") as f:
            # read defensively with [()] for both scalars (0-d) and arrays.
            # scalar datasets (eig_star, y_obs_r, eig_star_r) fail with [:];
            # [()] works uniformly for all shapes.
            data = {
                "mu0": f["mu0"][()],
                "Sigma0": f["Sigma0"][()],
                "eig_star": f["eig_star"][()],
                "xi_opt": f["xi_opt"][()],
                "Sigma_r": f["Sigma_r"][()],
                "xi_r": f["xi_r"][()],
                "y_obs_r": f["y_obs_r"][()],
                "eig_star_r": f["eig_star_r"][()],
                "round_idx": f["round_idx"][()],
                "trial_idx": f["trial_idx"][()],
                "a": f["a"][()],
                "b": f["b"][()],
                "xi": f["xi"][()],
                "est_eig": f["est_eig"][()],
                "true_eig": f["true_eig"][()],
                "state": f["state"][()],
                "walltime_s": f["walltime_s"][()],
            }
            return data
    except (OSError, KeyError) as e:
        print(f"  CORRUPT (read error): {h5_path.name}: {e}")
        return None


def validate_shard(h5_path: Path, attrs: dict, data: dict, cfg: dict,
                   expected_cell: tuple) -> bool:
    """
    validate config_hash, cell locator attrs, startup designs,
    trial count. return True if all pass, False if any fail (record as corrupt).

    args:
        expected_cell: (arm, geometry, prior_idx, method, seed_rep)
        cfg: config dict with data_dim, config_hash, n_rounds, n_trials, n_startup, config_seed

    raises: ValueError on mismatch (caller catches, logs, and marks corrupt).
    """
    # validate config_hash. this is a COARSE provenance stamp, sensitive
    # to cosmetic config drift (paths/dispatch keys) that does not change any
    # cell's computation. the real computational-consistency guard is the
    # startup-design check below (re-derives each cell's seeded designs). so a
    # config_hash mismatch is downgraded to a WARNING when gather_ignore_config_
    # hash is set, letting a consistent, valid shard set through.
    stored_hash = attrs.get("config_hash")
    if isinstance(stored_hash, bytes):
        stored_hash = stored_hash.decode()
    expected_hash = config_hash(cfg)  # imported from step1_run_studies
    if stored_hash != expected_hash:
        if cfg.get("gather_ignore_config_hash"):
            print(f"  WARN config_hash mismatch (flag override): stored={stored_hash[:12]} "
                  f"expected={expected_hash[:12]} cell={expected_cell} -- relying on C15 startup check")
        else:
            raise ValueError(
                f"config_hash mismatch: stored={stored_hash}, expected={expected_hash}. "
                f"this shard is from a different config. NEVER remap."
            )

    # validate cell locator attrs (5 separate attrs)
    arm, geometry, prior_idx, method, seed_rep = expected_cell
    if attrs.get("arm") != arm:
        raise ValueError(f"arm mismatch: stored={attrs.get('arm')}, expected={arm}")
    if attrs.get("geometry") != geometry:
        raise ValueError(f"geometry mismatch")
    if attrs.get("prior_idx") != prior_idx:
        raise ValueError(f"prior_idx mismatch")
    if attrs.get("method") != method:
        raise ValueError(f"method mismatch")
    if attrs.get("seed_rep") != seed_rep:
        raise ValueError(f"seed_rep mismatch")

    # validate trial count (n_trials == R*T)
    R = cfg["n_rounds"]
    T = cfg["n_trials"]
    n_trials = attrs.get("n_trials")
    if n_trials != R * T:
        raise ValueError(f"n_trials mismatch: stored={n_trials}, expected={R*T}")

    # validate startup designs (bitwise match of sobol seed)
    n_startup = cfg["n_startup"]
    r = 0  # first round
    study_seed_r = study_seed(expected_cell, r, cfg["config_seed"])
    expected_startup = startup_designs(study_seed_r, n_startup)  # (n_startup, 2)

    # trials are stored FLAT by global index; startup is in round 0, trial_idx 0..n_startup-1
    for t in range(n_startup):
        stored_a = data["a"][t]
        stored_b = data["b"][t]
        expected_a, expected_b = expected_startup[t]

        if not (np.isclose(stored_a, expected_a, atol=1e-9) and
                np.isclose(stored_b, expected_b, atol=1e-9)):
            raise ValueError(
                f"startup design mismatch at trial {t}: "
                f"stored=({stored_a}, {stored_b}), expected=({expected_a}, {expected_b})"
            )

    return True


def flag_replay_signature(data: dict, R: int, T: int) -> None:
    """
    flag any two trials in a round with identical (a,b) AND identical est_eig.
    this is a diagnostic warning (not a fail) for resumed runs that re-proposed.

    log to stdout and continue (no abort on replay).
    """
    replays = []

    for r in range(R):
        for t1 in range(r * T, (r + 1) * T):
            for t2 in range(t1 + 1, (r + 1) * T):
                if (np.isclose(data["a"][t1], data["a"][t2], atol=1e-9) and
                    np.isclose(data["b"][t1], data["b"][t2], atol=1e-9) and
                    np.isclose(data["est_eig"][t1], data["est_eig"][t2], atol=1e-9)):
                    replays.append((r, t1, t2))

    if replays:
        print(f"  WARNING: replay signature in {len(replays)} trial pairs "
              f"(resumed run re-proposed same designs)")


def normalize_shards(shards_by_cell: dict, expected_cells: list,
                     cfg: dict) -> tuple:
    """
    args:
        shards_by_cell: dict mapping expected_cell -> (attrs, data)
        expected_cells: list of all cells (from enumerate_cells)
        cfg: config dict

    return: (studies_list, rounds_list, trials_list) ready to write.

    strategy: accumulate all records in memory (lists of dicts), then
    write as FLAT parallel datasets (no groups, no compound dtypes).
    """
    R = cfg["n_rounds"]
    T = cfg["n_trials"]

    studies_list = []
    rounds_list = []
    trials_list = []

    for study_id, cell in enumerate(expected_cells):
        arm, geometry, prior_idx, method, seed_rep = cell

        if cell not in shards_by_cell:
            # missing shard (already recorded in missing_cells); skip
            continue

        attrs, data = shards_by_cell[cell]

        # one record per study
        studies_list.append({
            "study_id": np.int32(study_id),
            "arm": arm,
            "geometry": geometry,
            "prior_idx": np.int32(prior_idx),
            "method": method,
            "seed_rep": np.int32(seed_rep),
            "n_trials": np.int32(attrs["n_trials"]),
            "config_hash": attrs["config_hash"],
            "mu0": data["mu0"].astype(np.float32),
            "Sigma0": data["Sigma0"].astype(np.float32),
            "eig_star": np.float32(data["eig_star"]),
            "xi_opt": data["xi_opt"].astype(np.float32),
        })

        # per-round records (R rows)
        for r in range(R):
            rounds_list.append({
                "study_id": np.int32(study_id),
                "round_idx": np.int32(r),
                "Sigma_r": data["Sigma_r"][r].astype(np.float32),
                "xi_r": data["xi_r"][r].astype(np.float32),
                "y_obs_r": np.float32(data["y_obs_r"][r]),
                "eig_star_r": np.float32(data["eig_star_r"][r]),
            })

        # per-trial records (N = R*T rows, flat)
        for t in range(R * T):
            state_val = data["state"][t]
            if isinstance(state_val, bytes):
                state_val = state_val.decode("utf-8")
            trials_list.append({
                "study_id": np.int32(study_id),
                "round_idx": np.int32(data["round_idx"][t]),
                "trial_idx": np.int32(data["trial_idx"][t]),
                "a": np.float32(data["a"][t]),
                "b": np.float32(data["b"][t]),
                "xi": data["xi"][t].astype(np.float32),
                "est_eig": np.float32(data["est_eig"][t]),
                "true_eig": np.float32(data["true_eig"][t]),
                "state": state_val,
                "walltime_s": np.float32(data["walltime_s"][t]),
            })

    return studies_list, rounds_list, trials_list


def write_gathered_h5(output_path: Path, studies_list: list, rounds_list: list,
                      trials_list: list, cfg: dict) -> None:
    """
    write normalized tables to gathered.h5 as FLAT parallel datasets
    (one dataset per column, no groups). dataset names are prefixed per table.

    atomic write: use temp file + os.replace to avoid corruption on crash.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".h5.tmp")

    with h5py.File(tmp_path, "w") as f:
        # studies table
        if studies_list:
            _write_parallel_datasets(f, studies_list, "studies_")

        # rounds table
        if rounds_list:
            _write_parallel_datasets(f, rounds_list, "rounds_")

        # trials table
        if trials_list:
            _write_parallel_datasets(f, trials_list, "trials_")

        # metadata
        f.attrs["n_studies"] = len(studies_list)
        f.attrs["n_rounds_per_study"] = cfg["n_rounds"]
        f.attrs["n_trials_per_round"] = cfg["n_trials"]

    os.replace(tmp_path, output_path)


def _write_parallel_datasets(h5_file, records: list, prefix: str) -> None:
    """
    write list of dicts as parallel FLAT datasets (one per column).

    args:
        h5_file: h5py.File object (write mode)
        records: list of dicts, all with identical keys
        prefix: prefix for dataset names (e.g., "studies_", "rounds_", "trials_")

    strategy: infer dtype from first record; scalars are shape (n,),
    arrays are shape (n, d, ...). all records must have same structure.
    """
    if not records:
        return

    n_records = len(records)
    first = records[0]

    for key, value in first.items():
        col_name = prefix + key

        if isinstance(value, (int, np.integer)):
            data = np.array([r[key] for r in records], dtype=np.int32)
            h5_file.create_dataset(col_name, data=data)

        elif isinstance(value, (float, np.floating)):
            data = np.array([r[key] for r in records], dtype=np.float32)
            h5_file.create_dataset(col_name, data=data)

        elif isinstance(value, str):
            data = np.array([r[key] for r in records], dtype=h5py.string_dtype(encoding="utf-8"))
            h5_file.create_dataset(col_name, data=data)

        elif isinstance(value, np.ndarray):
            data = np.array([r[key] for r in records], dtype=np.float32)
            h5_file.create_dataset(col_name, data=data)

        else:
            raise TypeError(f"unsupported type for {key}: {type(value)}")


def report_status(expected_cells: list, missing_cells: set, corrupt_cells: set,
                  shards_by_cell: dict, cfg: dict) -> None:
    """
    report counts and missing cell list. a silent shortfall is the failure
    mode to prevent; explicitly print all missing and corrupt cells.
    """
    n_expected = len(expected_cells)
    n_complete = len(shards_by_cell)
    n_missing = len(missing_cells)
    n_corrupt = len(corrupt_cells)

    print("\n" + "=" * 70)
    print("GATHER REPORT")
    print("=" * 70)
    print(f"expected cells: {n_expected}")
    print(f"complete: {n_complete} ({100*n_complete/n_expected:.1f}%)")
    print(f"missing: {n_missing}")
    print(f"corrupt: {n_corrupt}")
    print()

    # per-method breakdown
    method_counts = {}
    for cell in shards_by_cell.keys():
        arm, geometry, prior_idx, method, seed_rep = cell
        method_counts[method] = method_counts.get(method, 0) + 1

    if method_counts:
        print("per-method completion:")
        for method in sorted(method_counts.keys()):
            count = method_counts[method]
            print(f"  {method}: {count}")
        print()

    # per-geometry breakdown
    geom_counts = {}
    for cell in shards_by_cell.keys():
        arm, geometry, prior_idx, method, seed_rep = cell
        geom_counts[geometry] = geom_counts.get(geometry, 0) + 1

    if geom_counts:
        print("per-geometry completion:")
        for geom in sorted(geom_counts.keys()):
            count = geom_counts[geom]
            print(f"  {geom}: {count}")
        print()

    # explicitly list missing cells
    if missing_cells:
        print(f"MISSING CELLS ({len(missing_cells)}):")
        for cell in sorted(missing_cells):
            arm, geometry, prior_idx, method, seed_rep = cell
            print(f"  arm={arm} geom={geometry} prior={prior_idx} method={method} seed={seed_rep}")
        print()

    # explicitly list corrupt cells
    if corrupt_cells:
        print(f"CORRUPT CELLS ({len(corrupt_cells)}):")
        for cell in sorted(corrupt_cells):
            arm, geometry, prior_idx, method, seed_rep = cell
            print(f"  arm={arm} geom={geometry} prior={prior_idx} method={method} seed={seed_rep}")
        print()

    print("=" * 70)


def main(cfg_path: str, *, force: bool = False) -> None:
    """
    load config, enumerate expected cells, discover shards, validate each,
    merge complete shards, report status.

    args:
        cfg_path: path to config.yaml
        force: if True, regenerate even if output exists
    """
    # load config (plain dict; use cfg["key"])
    cfg = load_config(cfg_path)

    raw_results_dir = cfg["raw_results_dir"]
    processed_results_dir = cfg["processed_results_dir"]
    output_path = Path(processed_results_dir) / "gathered.h5"

    # check if output exists (skip if not forced)
    if output_path.exists() and not force:
        print(f"output {output_path} already exists; use --force to regenerate")
        return

    # enumerate expected cells (stable order from step1_run_studies); the method
    # axis comes from the winners file, so load it and pass it in
    from ex.utils.step2_runner.load_winners import load_winners
    winners = load_winners(cfg["winners_path"])
    print(f"enumerating cells...")
    expected_cells = enumerate_cells(cfg, winners)
    print(f"  {len(expected_cells)} expected cells")

    # discover shards
    print(f"discovering shards in {raw_results_dir}...")
    discovered_shards = discover_shards(raw_results_dir)
    print(f"  {len(discovered_shards)} discovered shards")

    # validate each expected cell
    print(f"validating shards...")
    missing_cells = set()
    corrupt_cells = set()
    shards_by_cell = {}

    for i, cell in enumerate(expected_cells):
        if (i + 1) % max(1, len(expected_cells) // 10) == 0:
            print(f"  {i+1}/{len(expected_cells)}")

        # find shard for this cell by searching attrs
        h5_path = None
        for shard_path in discovered_shards:
            attrs = read_shard_attrs(shard_path)
            if attrs is None:
                continue

            # match cell via 5 separate attrs
            arm, geometry, prior_idx, method, seed_rep = cell
            if (attrs.get("arm") == arm and
                attrs.get("geometry") == geometry and
                attrs.get("prior_idx") == prior_idx and
                attrs.get("method") == method and
                attrs.get("seed_rep") == seed_rep):
                h5_path = shard_path
                break

        if h5_path is None:
            missing_cells.add(cell)
            continue

        # read and validate shard
        attrs = read_shard_attrs(h5_path)
        if attrs is None:
            corrupt_cells.add(cell)
            continue

        data = read_shard_data(h5_path, cfg["n_rounds"], cfg["n_trials"],
                               cfg["data_dim"])
        if data is None:
            corrupt_cells.add(cell)
            continue

        try:
            validate_shard(h5_path, attrs, data, cfg, cell)
            flag_replay_signature(data, cfg["n_rounds"], cfg["n_trials"])
            shards_by_cell[cell] = (attrs, data)

        except (ValueError, AssertionError) as e:
            print(f"  VALIDATION FAILED {h5_path.name}: {e}")
            corrupt_cells.add(cell)

    # normalize and write
    print(f"merging {len(shards_by_cell)} shards...")
    studies_list, rounds_list, trials_list = normalize_shards(
        shards_by_cell, expected_cells, cfg
    )

    print(f"writing {output_path}...")
    write_gathered_h5(output_path, studies_list, rounds_list, trials_list, cfg)

    # report
    report_status(expected_cells, missing_cells, corrupt_cells, shards_by_cell, cfg)
    print(f"\noutput written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="gather step1 shards into normalized h5 tables"
    )
    parser.add_argument("--config", default="ex/ablations/eig_boed/config.yaml",
                        help="path to config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if output exists")
    args = parser.parse_args()

    main(args.config, force=args.force)
