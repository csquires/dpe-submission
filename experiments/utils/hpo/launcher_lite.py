"""lite launcher: matrix orchestration for general-partition hpo.

drops cpu_array drain and legacy per-method controllers. just:
  1. resolve experiments + methods, build (method, exp) matrix.
  2. write workflow_pairs JSON (NFS).
  3. submit one watchdog_lite via submit_watchdog_lite.sh, forwarding
     --workflow-pairs so the lite watchdog ticks each pair internally.

mirrors launcher.py steps 1-6 + a slimmer step 7 + step 9.
"""

import argparse
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from experiments.utils.hpo import launcher as _omnibus
from experiments.utils.walltime_caps import speed_rank

logger = logging.getLogger(__name__)

_WORKDIR = os.environ.get("DPE_WORKDIR") or os.getcwd()


def _resolve_watchdog_lite_script() -> Path:
    p = Path(_WORKDIR) / "experiments" / "utils" / "submit_watchdog_lite.sh"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return p


def submit_watchdog_lite(queue_file: Path, my_cap: int, total_cap: int,
                         extra_args: Optional[list[str]] = None) -> str:
    """invoke submit_watchdog_lite.sh; parse jid from `jobid: <JID>` line."""
    script = _resolve_watchdog_lite_script()
    cmd = ["bash", str(script), "--reset", str(queue_file),
           str(my_cap), str(total_cap), "60"]
    if extra_args:
        cmd.append("--")
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(
            f"sbatch watchdog_lite failed (rc={result.returncode}): "
            f"cmd={cmd!r} stderr={result.stderr[:400]}"
        )
    for line in result.stdout.splitlines():
        if line.startswith("jobid:"):
            jid = line.split(":", 1)[1].strip()
            if jid:
                return jid
    raise ValueError(f"could not parse jid from output: {result.stdout[:400]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="hpo lite launcher (general partition)")
    p.add_argument("--experiments", default="all",
                   help='CSV list of experiment names, or "all"')
    p.add_argument("--methods", default="all",
                   help='CSV list of method names, or "all"')
    p.add_argument("--queue-file", type=Path, default=None,
                   help="watchdog queue file; default $DPE_DATA_ROOT/<run>_lite_queue.txt")
    p.add_argument("--my-cap", type=int, default=45,
                   help="per-user general-partition job cap; <=50 to fit "
                        "normal qos MaxSubmitJobsPerUser=50 with headroom")
    p.add_argument("--total-cap", type=int, default=1000,
                   help="total general-partition job cap; effectively "
                        "unbounded since per-user QoS limits dominate")
    p.add_argument("--budget", type=int, default=250,
                   help="trial budget per (method, exp) workflow")
    p.add_argument("--force", action="store_true",
                   help="overwrite non-empty queue_file without error")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve matrix and print plan; no submissions")
    p.add_argument("--output-suffix", type=str, default="",
                   help="suffix appended to per-pair output_dir leaf "
                        "(data_root/<exp>/<method><suffix>); empty by default. "
                        "useful to keep a lite run disjoint from a preempt run")
    p.add_argument("--skip-pairs-file", type=Path, default=None,
                   help="optional JSON file listing (method, experiment) pairs "
                        "to drop from the matrix. format: list of "
                        '{"method": ..., "experiment": ...} objects. used to '
                        "exclude clear-winner cells whose HPs are already "
                        "pinned downstream.")
    p.add_argument("--partition", default="general",
                   help="slurm partition for trial sbatch + cap polling")
    p.add_argument("--run-id", default=None,
                   help="override run_id (default: timestamp)")
    return p.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args is None:
        args = parse_args()

    exps = _omnibus.resolve_experiments(args.experiments)
    methods = _omnibus.resolve_methods(args.methods, {})
    adapter_map, adapter_skipped = _omnibus.load_adapters(exps)
    valid_pairs, pair_skipped = _omnibus.build_matrix(methods, adapter_map)
    # honor the user's experiment ordering (first arg wins). within an
    # experiment, sort methods slow-first so heavy work front-loads.
    exp_order = {e: i for i, e in enumerate(exps)}
    valid_pairs.sort(key=lambda p: (exp_order.get(p[1], 1_000_000),
                                   speed_rank(p[0]), p[0]))

    # drop pairs the caller has flagged as already-pinned (clear winners)
    skip_set: set[tuple[str, str]] = set()
    if args.skip_pairs_file is not None:
        skip_data = json.loads(args.skip_pairs_file.read_text())
        skip_set = {(d["method"], d["experiment"]) for d in skip_data}
    if skip_set:
        before = len(valid_pairs)
        valid_pairs = [p for p in valid_pairs if p not in skip_set]
        logger.info("skip-pairs: dropped %d pairs from matrix (%d remain)",
                    before - len(valid_pairs), len(valid_pairs))

    all_skipped: dict = {**{k: [v] for k, v in adapter_skipped.items()}, **pair_skipped}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"lite_{timestamp}"

    if args.queue_file is None:
        data_root_s = os.environ.get("DPE_DATA_ROOT", ".")
        args.queue_file = Path(data_root_s) / f"{run_id}_watchdog_queue.txt"
    _omnibus.validate_queue_file(args.queue_file, force=args.force)

    logdir = _omnibus.init_watchdog_logdir(run_id)
    _omnibus.record_skipped(all_skipped, run_id)

    if args.dry_run:
        print(f"[dry-run] run_id={run_id} valid_pairs={len(valid_pairs)} "
              f"partition={args.partition}")
        for m, e in valid_pairs:
            print(f"  {m} x {e}")
        if all_skipped:
            print(f"[dry-run] skipped: {json.dumps(all_skipped, indent=2)}")
        return

    data_root = Path(os.environ["DPE_DATA_ROOT"])
    suffix = args.output_suffix or ""
    workflow_pairs_data = [
        {"method": m, "experiment": e,
         "output_dir": str(data_root / e / f"{m}{suffix}"),
         "budget": args.budget}
        for m, e in valid_pairs
    ]
    workflow_pairs_file = data_root / f"workflow_pairs_{run_id}.json"
    workflow_pairs_file.write_text(json.dumps(workflow_pairs_data, indent=2))
    logger.info("wrote workflow pairs to %s (n=%d)",
                workflow_pairs_file, len(workflow_pairs_data))

    exps_in_run = sorted({e for _, e in valid_pairs})
    _omnibus.stage_recalibrated_specs(exps_in_run, data_root)

    wd_extra = [
        "--workflow-pairs", str(workflow_pairs_file),
        "--partition", args.partition,
    ]
    watchdog_jid = submit_watchdog_lite(
        args.queue_file, my_cap=args.my_cap, total_cap=args.total_cap,
        extra_args=wd_extra,
    )
    logger.info("watchdog_lite submitted: jid=%s", watchdog_jid)

    _omnibus.emit_summary(run_id, watchdog_jid, [], args.queue_file,
                          logdir, all_skipped, timestamp, cpu_record=None)


if __name__ == "__main__":
    main()
