"""step2_gather: shard discovery, validation, and merge into normalized tables.

reads FLAT h5 shards, validates config_hash + startup designs,
merges into 4 FLAT parallel tables (studies/rounds/trials/alpha_trials with
prefixed dataset names). no nested groups, no compound dtypes.

thin near-copy of eig_elbo_boed/step2_gather.py: elbo_boed uses
design_channel=analytic, so cfg["n_trials"] = cfg["n_startup"] = 0 and every
design-trial-indexed array (length R*T) is empty. all analytic-channel
guards below (startup-check skip, n_trials==R*T==0, range(R*T)==range(0))
already exist unchanged in the sibling and hold for T=0.
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

from ex.ablations.elbo_boed.step1_run_studies import enumerate_cells, config_hash
from ex.ablations.eig_boed.study import startup_designs, study_seed


def _decode(val):
    """decode bytes attrs/dataset values to str; pass through non-bytes unchanged."""
    return val.decode("utf-8") if isinstance(val, bytes) else val


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
            return {
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
    except OSError as e:
        # truncated file (preemption, incomplete write)
        print(f"  CORRUPT (truncated): {h5_path.name}: {e}")
        return None


def read_shard_data(h5_path: Path, R: int, T: int, T_alpha: int, data_dim: int) -> dict | None:
    """
    read all FLAT root datasets from shard (NOT groups).

    args:
        R: n_rounds, from cfg["n_rounds"]
        T: n_trials per round (design channel), from cfg["n_trials"]. for
            elbo_boed (design_channel=analytic) T=0, so the design-trial
            datasets (round_idx, trial_idx, a, b, xi, est_eig, true_eig,
            state, walltime_s) are length-0 arrays, not absent.
        T_alpha: n_trials_alpha per round (alpha BO channel), from cfg["n_trials_alpha"]
        data_dim: from cfg["data_dim"]

    return: dict with all datasets loaded into memory, or None if any read fails.

    prior (before round 0): mu0, Sigma0, eig_star, xi_opt.
    per-round (length R): Sigma_r, Sigma_exact_r, xi_r, y_obs_r, eig_star_r,
    alpha_r, fell_back_r, post_kl_r.
    per-trial design channel (length R*T, flat): round_idx, trial_idx, a, b,
    xi, est_eig, true_eig, state, walltime_s.
    per-alpha-trial channel (length R*T_alpha, flat): alpha_round_idx,
    alpha_trial_idx, alpha_val, elbo_est, alpha_state, alpha_walltime_s.

    read defensively with [()] for both scalars (0-d) and arrays.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            return {
                "mu0": f["mu0"][()],
                "Sigma0": f["Sigma0"][()],
                "eig_star": f["eig_star"][()],
                "xi_opt": f["xi_opt"][()],
                "Sigma_r": f["Sigma_r"][()],
                "Sigma_exact_r": f["Sigma_exact_r"][()],
                "xi_r": f["xi_r"][()],
                "y_obs_r": f["y_obs_r"][()],
                "eig_star_r": f["eig_star_r"][()],
                "alpha_r": f["alpha_r"][()],
                "fell_back_r": f["fell_back_r"][()],
                "post_kl_r": f["post_kl_r"][()],
                "round_idx": f["round_idx"][()],
                "trial_idx": f["trial_idx"][()],
                "a": f["a"][()],
                "b": f["b"][()],
                "xi": f["xi"][()],
                "est_eig": f["est_eig"][()],
                "true_eig": f["true_eig"][()],
                "state": f["state"][()],
                "walltime_s": f["walltime_s"][()],
                "alpha_round_idx": f["alpha_round_idx"][()],
                "alpha_trial_idx": f["alpha_trial_idx"][()],
                "alpha_val": f["alpha_val"][()],
                "elbo_est": f["elbo_est"][()],
                "alpha_state": f["alpha_state"][()],
                "alpha_walltime_s": f["alpha_walltime_s"][()],
            }
    except (OSError, KeyError) as e:
        print(f"  CORRUPT (read error): {h5_path.name}: {e}")
        return None


def validate_shard(h5_path: Path, attrs: dict, data: dict, cfg: dict,
                   expected_cell: tuple) -> bool:
    """
    validate config_hash, cell locator attrs, startup designs, trial count.
    return True if all pass, False if any fail (record as corrupt).

    args:
        expected_cell: (arm, geometry, prior_idx, method, seed_rep)
        cfg: config dict with data_dim, config_hash, n_rounds, n_trials, n_startup, config_seed

    config_hash behavior: cfg has no gather_ignore_config_hash key, so
    cfg.get(...) is always falsy and a mismatch hard-rejects
    unconditionally (mirrors eig_boed's fallback path,
    which the missing key always routes into the raise branch).

    trial count check compares against R*cfg["n_trials"]; elbo_boed's
    analytic design channel sets cfg["n_trials"]=0, so this reduces to
    n_trials==0 and passes for any shard with an empty design-trial channel.

    startup design check: the design channel has n_startup=0 designs (all
    proposal is analytic, not sobol-seeded), so the len(a_arr) >= n_startup
    guard (0 >= 0) lets an empty a_arr through into range(n_startup)=range(0),
    which is a no-op loop; no bitwise check is ever performed.

    raises: ValueError on any check failure (caller catches, logs, marks corrupt).
    """
    stored_hash = _decode(attrs.get("config_hash"))
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
    if _decode(attrs.get("arm")) != arm:
        raise ValueError(f"arm mismatch: stored={attrs.get('arm')}, expected={arm}")
    if _decode(attrs.get("geometry")) != geometry:
        raise ValueError("geometry mismatch")
    if attrs.get("prior_idx") != prior_idx:
        raise ValueError("prior_idx mismatch")
    if _decode(attrs.get("method")) != method:
        raise ValueError("method mismatch")
    if attrs.get("seed_rep") != seed_rep:
        raise ValueError("seed_rep mismatch")

    # validate trial count (n_trials == R*T, design channel; T=0 here)
    R = cfg["n_rounds"]
    T = cfg["n_trials"]
    n_trials = attrs.get("n_trials")
    if n_trials != R * T:
        raise ValueError(f"n_trials mismatch: stored={n_trials}, expected={R*T}")

    # validate startup designs (bitwise match of sobol seed).
    # design channel only; guard against empty/missing design-trial arrays so
    # an analytic channel (no per-trial records, n_startup=0) does not crash.
    n_startup = cfg["n_startup"]
    a_arr = data.get("a")
    if a_arr is not None and len(a_arr) >= n_startup:
        study_seed_r = study_seed(expected_cell, 0, cfg["config_seed"])  # round 0
        expected_startup = startup_designs(study_seed_r, n_startup)  # (n_startup, 2)

        # trials are stored FLAT by global index; startup is round 0, trial_idx 0..n_startup-1
        for t in range(n_startup):
            stored_a, stored_b = data["a"][t], data["b"][t]
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
    design channel only; with T=0 (analytic) the trial loop is a no-op.

    log to stdout and continue (replay does not abort the gather).
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
        cfg: config dict with n_rounds, n_trials, n_trials_alpha

    return: (studies_list, rounds_list, trials_list, alpha_trials_list) ready to write.

    strategy: accumulate all records in memory (lists of dicts), then
    write as FLAT parallel datasets (no groups, no compound dtypes).
    with cfg["n_trials"]=0 (analytic channel) range(R*T)==range(0), so
    trials_list stays empty and only rounds_list/alpha_trials_list get rows.
    """
    R = cfg["n_rounds"]
    T = cfg["n_trials"]
    T_alpha = cfg["n_trials_alpha"]

    studies_list = []
    rounds_list = []
    trials_list = []
    alpha_trials_list = []

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
            "config_hash": _decode(attrs["config_hash"]),
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
                "Sigma_exact_r": data["Sigma_exact_r"][r].astype(np.float32),
                "xi_r": data["xi_r"][r].astype(np.float32),
                "y_obs_r": np.float32(data["y_obs_r"][r]),
                "eig_star_r": np.float32(data["eig_star_r"][r]),
                "alpha_r": np.float32(data["alpha_r"][r]),
                "fell_back_r": np.int32(data["fell_back_r"][r]),
                "post_kl_r": np.float32(data["post_kl_r"][r]),
            })

        # per-trial records (N_d = R*T rows, design channel, flat; empty when T=0)
        for t in range(R * T):
            trials_list.append({
                "study_id": np.int32(study_id),
                "round_idx": np.int32(data["round_idx"][t]),
                "trial_idx": np.int32(data["trial_idx"][t]),
                "a": np.float32(data["a"][t]),
                "b": np.float32(data["b"][t]),
                "xi": data["xi"][t].astype(np.float32),
                "est_eig": np.float32(data["est_eig"][t]),
                "true_eig": np.float32(data["true_eig"][t]),
                "state": _decode(data["state"][t]),
                "walltime_s": np.float32(data["walltime_s"][t]),
            })

        # per-alpha-trial records (N_a = R*T_alpha rows, alpha BO channel, flat)
        for ta in range(R * T_alpha):
            alpha_trials_list.append({
                "study_id": np.int32(study_id),
                "alpha_round_idx": np.int32(data["alpha_round_idx"][ta]),
                "alpha_trial_idx": np.int32(data["alpha_trial_idx"][ta]),
                "alpha_val": np.float32(data["alpha_val"][ta]),
                "elbo_est": np.float32(data["elbo_est"][ta]),
                "alpha_state": _decode(data["alpha_state"][ta]),
                "alpha_walltime_s": np.float32(data["alpha_walltime_s"][ta]),
            })

    return studies_list, rounds_list, trials_list, alpha_trials_list


def write_gathered_h5(output_path: Path, studies_list: list, rounds_list: list,
                      trials_list: list, alpha_trials_list: list, cfg: dict) -> None:
    """
    write normalized tables to gathered.h5 as FLAT parallel datasets
    (one dataset per column, no groups). datasets PREFIXED per table.

    atomic write: use temp file + os.replace to avoid corruption on crash.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".h5.tmp")

    with h5py.File(tmp_path, "w") as f:
        # studies, rounds, trials, alpha_trials tables (each self-guards on empty)
        _write_parallel_datasets(f, studies_list, "studies_")
        _write_parallel_datasets(f, rounds_list, "rounds_")
        _write_parallel_datasets(f, trials_list, "trials_")
        _write_parallel_datasets(f, alpha_trials_list, "alpha_trials_")

        # metadata
        f.attrs["n_studies"] = len(studies_list)
        f.attrs["n_rounds_per_study"] = cfg["n_rounds"]
        f.attrs["n_trials_per_round"] = cfg["n_trials"]
        f.attrs["n_trials_alpha_per_round"] = cfg["n_trials_alpha"]

    os.replace(tmp_path, output_path)


def _write_parallel_datasets(h5_file, records: list, prefix: str) -> None:
    """
    write list of dicts as parallel FLAT datasets (one per column).

    args:
        h5_file: h5py.File object (write mode)
        records: list of dicts, all with identical keys
        prefix: prefix for dataset names (e.g., "studies_", "rounds_", "trials_", "alpha_trials_")

    strategy: infer dtype from first record; scalars are shape (n,),
    arrays are shape (n, d, ...). all records must have same structure.
    """
    if not records:
        return

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
            if (_decode(attrs.get("arm")) == arm and
                _decode(attrs.get("geometry")) == geometry and
                attrs.get("prior_idx") == prior_idx and
                _decode(attrs.get("method")) == method and
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
                               cfg["n_trials_alpha"], cfg["data_dim"])
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
    studies_list, rounds_list, trials_list, alpha_trials_list = normalize_shards(
        shards_by_cell, expected_cells, cfg
    )

    print(f"writing {output_path}...")
    write_gathered_h5(output_path, studies_list, rounds_list, trials_list,
                      alpha_trials_list, cfg)

    # report
    report_status(expected_cells, missing_cells, corrupt_cells, shards_by_cell, cfg)
    print(f"\noutput written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="gather step1 shards into normalized h5 tables"
    )
    parser.add_argument("--config", default="ex/ablations/elbo_boed/config.yaml",
                        help="path to config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if output exists")
    args = parser.parse_args()

    main(args.config, force=args.force)
