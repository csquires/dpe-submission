"""preempt-partition keeper for optuna hpo -- companion to the array baseline.

a long-lived cpu-partition job. each cycle it counts terminal trials
(COMPLETE + PRUNED) in every (experiment, method) study of a StudyConfig and,
for any study still short of `target_trials`, tops its preempt-partition worker
population up to `jobs_per_method`, subject to user/cluster queue caps. exits
once every study has reached its target.

idempotent under its own preemption: per-cycle state is recomputed from the
journals and from `squeue`, so a slurm requeue of the keeper loses nothing --
there is no local bookkeeping to corrupt. this is what lets the optuna stack
do a double-ended (array + preempt) drain without the legacy per-trial
`missing_array` machinery: every dispatched worker is fungible and cooperates
through the shared journal.

cli:
  python -m ex.utils.hpo.optuna.keeper --config <dotted.module> [flags]
"""
import argparse
import logging
import os
import subprocess
import sys
import time

from optuna.trial import TrialState

from ex.utils.hpo.optuna.study_config import load_config
from ex.utils.hpo.optuna.storage import create_or_load

logger = logging.getLogger(__name__)

# trials that consumed budget and inform TPE; FAIL/RUNNING do not count.
_TERMINAL = (TrialState.COMPLETE, TrialState.PRUNED)


def count_terminal(experiment: str, method: str) -> int:
    """count COMPLETE + PRUNED trials in a study's journal."""
    study = create_or_load(experiment, method)
    return sum(1 for t in study.trials if t.state in _TERMINAL)


def squeue_count(
    partition: str, *, user: str | None = None, name: str | None = None
) -> int:
    """count queued+running slurm jobs matching the given filters."""
    cmd = ["squeue", "-h", "-p", partition]
    if user:
        cmd += ["-u", user]
    if name:
        cmd += ["-n", name]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return sum(1 for line in out.splitlines() if line.strip())


def job_name(experiment: str, method: str) -> str:
    """slurm job-name for a keeper-dispatched preempt worker."""
    return f"optk_{experiment}_{method}"


def dispatch(config_module: str, combo_index: int, experiment: str,
             method: str, args: argparse.Namespace) -> str:
    """sbatch one preempt-partition optuna worker; return its jobid.

    the worker is the ordinary `ex.utils.hpo.optuna.submit` entrypoint pinned
    to one combo via --combo-index; it is preempt-safe on its own (journal +
    cleanup_zombies + SIGTERM handler).
    """
    name = job_name(experiment, method)
    wrap = (
        f"source ~/.bashrc && conda activate fac && cd {args.workdir} && "
        f"python -m ex.utils.hpo.optuna.submit "
        f"--config {config_module} --combo-index {combo_index}"
    )
    cmd = [
        "sbatch", "--parsable",
        "--job-name", name,
        "--partition", args.preempt_partition,
        "--qos", args.preempt_qos,
        "--time", args.worker_walltime,
        "--cpus-per-task", str(args.cpus),
        "--mem", args.mem,
        "--gpus", str(args.gpus),
        "--requeue",
        "--output", os.path.join(args.log_dir, f"{name}_%j.out"),
        "--wrap", wrap,
    ]
    if args.dry_run:
        logger.info("DRY-RUN dispatch %s: %s", name, " ".join(cmd))
        return "dry-run"
    jid = subprocess.run(
        cmd, capture_output=True, text=True, check=True
    ).stdout.strip()
    logger.info("dispatched %s -> jobid %s", name, jid)
    return jid


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="preempt keeper for optuna hpo")
    p.add_argument("--config", required=True,
                   help="dotted StudyConfig module path")
    p.add_argument("--jobs-per-method", type=int, default=4,
                   help="target concurrent preempt workers per under-target study")
    p.add_argument("--my-cap", type=int, default=80,
                   help="max own preempt jobs (headroom under MaxJobSubmitPU)")
    p.add_argument("--total-cap", type=int, default=200,
                   help="max total preempt jobs (all users) before pausing")
    p.add_argument("--poll-interval", type=int, default=60,
                   help="seconds between cycles")
    p.add_argument("--max-dispatch-per-cycle", type=int, default=10,
                   help="cap submits per cycle to bound burstiness")
    p.add_argument("--max-cycles", type=int, default=0,
                   help="stop after N cycles (0 = run until all studies reach "
                        "target); use --max-cycles 1 with --dry-run for a "
                        "one-shot check")
    p.add_argument("--worker-walltime", default="3:00:00",
                   help="--time for each dispatched preempt worker")
    p.add_argument("--preempt-partition", default="preempt")
    p.add_argument("--preempt-qos", default="preempt_qos")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--mem", default="32G")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--workdir", default=os.getcwd())
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--dry-run", action="store_true",
                   help="print sbatch lines instead of submitting")
    return p.parse_args()


def main() -> int:
    """keeper main loop.

    per cycle: (1) count terminal trials per study; if all >= target_trials,
    exit. (2) compute preempt queue headroom under the caps. (3) for each
    under-target study, top up its named preempt jobs toward jobs_per_method,
    bounded by headroom and max_dispatch_per_cycle.
    """
    # configure the keeper logger explicitly (single handler, no propagation)
    # so import-time logging setup elsewhere cannot double or drop our lines.
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [keeper] %(levelname)s: %(message)s")
    )
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    args = _parse_args()

    if "DPE_DATA_ROOT" not in os.environ:
        logger.error("DPE_DATA_ROOT not set")
        return 1
    os.makedirs(args.log_dir, exist_ok=True)

    config = load_config(args.config)
    user = os.environ.get("USER")
    logger.info(
        "keeper start: experiment=%s methods=%s target_trials=%d "
        "jobs_per_method=%d",
        config.experiment, config.methods, config.target_trials,
        args.jobs_per_method,
    )

    cycle = 0
    while True:
        cycle += 1
        if args.max_cycles and cycle > args.max_cycles:
            logger.info("reached max_cycles=%d; exiting (studies may still be "
                        "below target)", args.max_cycles)
            break

        # 1. count terminal trials; collect studies still below target.
        pending: list[tuple[int, str, int]] = []
        for i, method in enumerate(config.methods):
            try:
                n = count_terminal(config.experiment, method)
            except Exception as e:
                logger.warning("cycle %d: count failed for %s: %s",
                                cycle, method, e)
                continue
            if n < config.target_trials:
                pending.append((i, method, n))
        if not pending:
            logger.info("cycle %d: all %d studies reached target_trials=%d; "
                        "done", cycle, len(config.methods),
                        config.target_trials)
            break

        # 2. preempt queue headroom under the caps.
        try:
            my_pending = squeue_count(args.preempt_partition, user=user)
            total_pending = squeue_count(args.preempt_partition)
        except Exception as e:
            logger.warning("cycle %d: squeue failed: %s; sleeping", cycle, e)
            time.sleep(args.poll_interval)
            continue
        headroom = min(
            args.my_cap - my_pending,
            args.total_cap - total_pending,
            args.max_dispatch_per_cycle,
        )
        logger.info(
            "cycle %d: %d study(ies) below target; preempt my=%d total=%d "
            "headroom=%d", cycle, len(pending), my_pending, total_pending,
            headroom,
        )
        if headroom <= 0:
            time.sleep(args.poll_interval)
            continue

        # 3. top each under-target study up toward jobs_per_method.
        dispatched = 0
        for (i, method, n) in pending:
            if dispatched >= headroom:
                break
            try:
                running = squeue_count(
                    args.preempt_partition, user=user,
                    name=job_name(config.experiment, method),
                )
            except Exception as e:
                logger.warning("squeue by name failed for %s: %s", method, e)
                continue
            deficit = args.jobs_per_method - running
            for _ in range(deficit):
                if dispatched >= headroom:
                    break
                dispatch(args.config, i, config.experiment, method, args)
                dispatched += 1
        logger.info("cycle %d: dispatched %d preempt worker(s) "
                    "(%d done this run)", cycle, dispatched, dispatched)
        time.sleep(args.poll_interval)

    logger.info("keeper exiting normally after %d cycle(s)", cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
