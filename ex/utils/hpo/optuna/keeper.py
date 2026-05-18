"""multi-lane keeper for optuna hpo -- drains all configured lanes.

a long-lived cpu-partition job. each cycle it counts terminal trials
(COMPLETE + PRUNED) in every (experiment, method) study of a StudyConfig and,
for any study still short of `target_trials`, tops its worker population
across all lanes in config.lanes, subject to per-lane caps and cluster safety
limits. exits once every study has reached its target.

each lane's max_concurrent is the per-lane total running cap; the keeper splits
it evenly across the studies still under target. the array lane gets ONE slurm
--array job per under-target study (sized to that study's share); preempt/cpu/
general lanes dispatch individual per-worker sbatch jobs.

idempotent under preemption: per-cycle state is recomputed from journals and
squeue, so a keeper requeue loses nothing -- no local bookkeeping. this enables
fungible workers across lanes cooperating through the shared journal without
legacy per-trial machinery.

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

from ex.utils.hpo.optuna.lanes import get_lane, LaneProfile
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


def job_name(experiment: str, method: str, lane_name: str) -> str:
    """slurm job-name for a keeper-dispatched worker, embedding the lane.

    format: optk_{experiment}_{method}_{lane_name}. the lane is included so
    per-lane squeue -n counts never collide across lanes.
    """
    return f"optk_{experiment}_{method}_{lane_name}"


def dispatch(config_module: str, combo_index: int, experiment: str,
             method: str, lane_name: str, lane: LaneProfile, workdir: str,
             log_dir: str, dry_run: bool) -> str:
    """sbatch one optuna worker on a non-array lane (preempt/cpu/general).

    the worker is the ordinary ex.utils.hpo.optuna.submit entrypoint pinned
    to one combo via --combo-index; it is preemption-safe on its own
    (journal + cleanup_zombies + SIGTERM handler).

    args:
      config_module: dotted module path to StudyConfig.
      combo_index: combo index (method ordinal) to pass to submit.
      experiment: experiment name.
      method: method name.
      lane_name: logical lane name (e.g. "preempt", "cpu", "general").
      lane: LaneProfile for this dispatch.
      workdir: working directory for submit.
      log_dir: directory for sbatch stdout/stderr.
      dry_run: if True, log the sbatch command and return "dry-run".

    returns: job ID (str) if submitted; "dry-run" if dry_run=True.
    """
    name = job_name(experiment, method, lane_name)
    wrap = (
        f"source ~/.bashrc && conda activate fac && cd {workdir} && "
        f"python -m ex.utils.hpo.optuna.submit "
        f"--config {config_module} --lane {lane_name} --combo-index {combo_index}"
    )
    cmd = [
        "sbatch", "--parsable",
        "--job-name", name,
        "--partition", lane.partition,
        "--time", lane.worker_walltime,
        "--cpus-per-task", str(lane.cpus_per_task),
        "--mem", lane.mem,
        "--requeue",
        "--output", os.path.join(log_dir, f"{name}_%j.out"),
        "--wrap", wrap,
    ]
    # include --gpus only if lane requires gpus.
    if lane.gpus > 0:
        cmd += ["--gpus", str(lane.gpus)]
    # include --qos only if lane specifies a qos.
    if lane.qos:
        cmd += ["--qos", lane.qos]

    if dry_run:
        logger.info("DRY-RUN dispatch %s: %s", name, " ".join(cmd))
        return "dry-run"

    try:
        jid = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        ).stdout.strip()
        logger.info("dispatched %s -> jobid %s", name, jid)
        return jid
    except subprocess.CalledProcessError as e:
        logger.error("dispatch failed for %s: %s", name, e.stderr)
        raise


def dispatch_array(config_module: str, combo_index: int, experiment: str,
                   method: str, lane: LaneProfile, n_workers: int,
                   workdir: str, log_dir: str, dry_run: bool) -> str:
    """sbatch one array job for a (experiment, method) study.

    the array has size N = throttle K = n_workers -- this study's share of the
    array lane's total cap (lane.max_concurrent // studies still under target).
    all elements pin to the same combo and cooperate through the journal, so
    they are fungible workers each running submit on the same study.

    args:
      config_module: dotted module path to StudyConfig.
      combo_index: combo index (method ordinal) to pass to submit.
      experiment: experiment name.
      method: method name.
      lane: LaneProfile for the array lane.
      n_workers: array size and %throttle for this study.
      workdir: working directory for submit.
      log_dir: directory for sbatch stdout/stderr.
      dry_run: if True, log the sbatch command and return "dry-run".

    returns: job ID (str) if submitted; "dry-run" if dry_run=True.

    note: the array lane ALWAYS has gpus=0 and qos=""; no --gpus or --qos flags.
    """
    name = job_name(experiment, method, "array")
    n = max(1, n_workers)
    array_spec = f"0-{n - 1}%{n}"

    wrap = (
        f"source ~/.bashrc && conda activate fac && cd {workdir} && "
        f"python -m ex.utils.hpo.optuna.submit "
        f"--config {config_module} --lane array --combo-index {combo_index}"
    )

    cmd = [
        "sbatch", "--parsable",
        f"--array={array_spec}",
        "--job-name", name,
        "--partition", "array",
        "--time", lane.worker_walltime,
        "--cpus-per-task", str(lane.cpus_per_task),
        "--mem", lane.mem,
        "--requeue",
        "--output", os.path.join(log_dir, f"{name}_%A_%a.out"),
        "--wrap", wrap,
    ]

    if dry_run:
        logger.info("DRY-RUN dispatch_array %s: %s", name, " ".join(cmd))
        return "dry-run"

    try:
        jid = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        ).stdout.strip()
        logger.info("dispatched array %s (size=%d) -> jobid %s", name, n, jid)
        return jid
    except subprocess.CalledProcessError as e:
        logger.error("dispatch_array failed for %s: %s", name, e.stderr)
        raise


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="multi-lane keeper for optuna hpo")
    p.add_argument("--config", required=True,
                   help="dotted StudyConfig module path")
    p.add_argument("--poll-interval", type=int, default=60,
                   help="seconds between cycles")
    p.add_argument("--max-dispatch-per-cycle", type=int, default=10,
                   help="cap submits per cycle to bound burstiness")
    p.add_argument("--max-cycles", type=int, default=0,
                   help="stop after N cycles (0 = run until all studies reach "
                        "target); use --max-cycles 1 with --dry-run for a "
                        "one-shot check")
    p.add_argument("--workdir", default=os.getcwd())
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--dry-run", action="store_true",
                   help="print sbatch lines instead of submitting")
    return p.parse_args()


def main() -> int:
    """keeper main loop.

    per cycle: (1) count terminal trials per study; if all >= target_trials,
    exit. (2) for each lane in config.lanes, split lane.max_concurrent (the
    per-lane total running cap) evenly across the studies still under target
    and top each up to its share; array lane uses one --array job per study,
    per-worker lanes use individual sbatch jobs. (3) sleep and repeat.
    """
    # configure keeper logger.
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [keeper] %(levelname)s: %(message)s")
    )
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    args = _parse_args()

    # check env and prepare.
    if "DPE_DATA_ROOT" not in os.environ:
        logger.error("DPE_DATA_ROOT not set")
        return 1
    os.makedirs(args.log_dir, exist_ok=True)

    config = load_config(args.config)
    user = os.environ.get("USER")

    # warn if keeper was requeued/restarted.
    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    if restart_count > 0:
        logger.warning(
            "keeper restarted (restart_count=%d); resuming drain idempotently",
            restart_count,
        )

    logger.info(
        "keeper start: experiment=%s methods=%s target_trials=%d lanes=%s",
        config.experiment, config.methods, config.target_trials, config.lanes,
    )

    cycle = 0
    while True:
        cycle += 1
        if args.max_cycles and cycle > args.max_cycles:
            logger.info(
                "reached max_cycles=%d; exiting (studies may still be below target)",
                args.max_cycles,
            )
            break

        # (1) count terminal trials; collect studies still below target.
        pending: list[tuple[int, str, int]] = []  # (combo_index, method, n_terminal)
        for i, method in enumerate(config.methods):
            try:
                n = count_terminal(config.experiment, method)
            except Exception as e:
                logger.warning("cycle %d: count failed for %s: %s", cycle, method, e)
                continue
            if n < config.target_trials:
                pending.append((i, method, n))

        if not pending:
            logger.info(
                "cycle %d: all %d studies reached target_trials=%d; done",
                cycle, len(config.methods), config.target_trials,
            )
            break

        # (2) for each lane: split lane.max_concurrent (the per-lane total
        # running cap) evenly across the studies still under target.
        n_pending = len(pending)
        cycle_dispatched = 0
        for lane_name in config.lanes:
            try:
                lane = get_lane(lane_name)
            except Exception as e:
                logger.warning(
                    "cycle %d: failed to get lane %s: %s; skipping",
                    cycle, lane_name, e,
                )
                continue

            # this study's share of the lane's total running cap.
            per_study = max(1, lane.max_concurrent // n_pending)

            if lane_name == "array":
                # array lane: one --array job per under-target study, sized to
                # that study's share; skip if a job is already alive.
                for (combo_index, method, n) in pending:
                    try:
                        count = squeue_count(
                            partition="array", user=user,
                            name=job_name(config.experiment, method, "array"),
                        )
                    except Exception as e:
                        logger.warning("squeue failed for array %s: %s", method, e)
                        continue

                    if count == 0:
                        try:
                            dispatch_array(
                                args.config, combo_index, config.experiment,
                                method, lane, per_study, args.workdir,
                                args.log_dir, args.dry_run,
                            )
                            cycle_dispatched += 1
                        except Exception as e:
                            logger.warning("dispatch_array failed for %s: %s", method, e)

            else:
                # per-worker lanes (preempt, cpu, general): top each study up to
                # its per_study share, bounded by the per-cycle burst cap.
                for (combo_index, method, n) in pending:
                    if cycle_dispatched >= args.max_dispatch_per_cycle:
                        break

                    try:
                        running = squeue_count(
                            partition=lane.partition, user=user,
                            name=job_name(config.experiment, method, lane_name),
                        )
                    except Exception as e:
                        logger.warning(
                            "squeue by name failed for %s on %s: %s",
                            method, lane_name, e,
                        )
                        continue

                    deficit = per_study - running
                    deficit = min(deficit, config.target_trials - n)
                    deficit = min(
                        deficit, args.max_dispatch_per_cycle - cycle_dispatched
                    )

                    for _ in range(max(0, deficit)):
                        try:
                            dispatch(
                                args.config, combo_index, config.experiment,
                                method, lane_name, lane, args.workdir,
                                args.log_dir, args.dry_run,
                            )
                            cycle_dispatched += 1
                        except Exception as e:
                            logger.warning(
                                "dispatch failed for %s on %s: %s",
                                method, lane_name, e,
                            )
                            break

        logger.info(
            "cycle %d: %d study(ies) below target; dispatched %d worker(s)",
            cycle, len(pending), cycle_dispatched,
        )
        time.sleep(args.poll_interval)

    logger.info("keeper exiting normally after %d cycle(s)", cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
