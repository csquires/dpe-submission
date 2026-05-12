#!/usr/bin/env python
"""
build_task_manifest.py

Generate task manifest for SMODICE pipeline SLURM array job.

Reads config.yaml, iterates (k1_idx, k2_idx) cells, computes n_seeds per cell
(40 for hard corners, 20 elsewhere), and writes a task manifest.

Manifest format (one task per line, space-separated):
  <task_id> <k1_idx> <k2_idx> <seed_offset>

Usage:
  python ex/synth/occupancy/build_task_manifest.py
  (prints "wrote N tasks to ..." and SLURM array range)
"""

import argparse
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="ex/synth/occupancy/config.yaml",
        help="Path to config.yaml (default: ex/synth/occupancy/config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ex/synth/occupancy/task_manifest.txt",
        help="Path to output manifest (default: ex/synth/occupancy/task_manifest.txt)",
    )
    args = parser.parse_args()

    # load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # extract kl targets and seed allocation params
    kl_targets = config["kl_targets"]
    k1_values = kl_targets["k1_values"]
    k2_values = kl_targets["k2_values"]
    hard_corner_threshold = kl_targets.get("hard_corner_threshold", 1.0)
    seeds_default = kl_targets.get("seeds_default", 20)
    seeds_hard = kl_targets.get("seeds_hard", 40)

    # build task list: [(task_id, k1_idx, k2_idx, seed_offset), ...]
    tasks = []
    task_id = 0

    for k1_idx, k1_val in enumerate(k1_values):
        for k2_idx, k2_val in enumerate(k2_values):
            # determine n_seeds for this cell
            is_hard = k1_val >= hard_corner_threshold and k2_val >= hard_corner_threshold
            n_seeds = seeds_hard if is_hard else seeds_default

            # add one task per seed offset (absolute seed = config["seed"] + offset
            # is computed inside step1_create_data.py per spec; manifest writes offset only)
            for seed_offset in range(n_seeds):
                tasks.append((task_id, k1_idx, k2_idx, seed_offset))
                task_id += 1

    # write manifest
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for task_id, k1_idx, k2_idx, seed in tasks:
            f.write(f"{task_id} {k1_idx} {k2_idx} {seed}\n")

    total_tasks = len(tasks)
    print(f"wrote {total_tasks} tasks to {output_path}")
    print(f"SLURM array range: --array=0-{total_tasks - 1}")


if __name__ == "__main__":
    main()
