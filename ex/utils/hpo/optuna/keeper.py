"""multi-lane keeper for optuna hpo -- drains all configured lanes.

a long-lived cpu-partition job. each cycle it counts COMPLETE
trials in every (experiment, method) study of a StudyConfig and,
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
from typing import Hashable

from optuna.trial import TrialState

from ex.utils.hpo.optuna.cores_registry import get_cores_for_method, needs_gpu
from ex.utils.hpo.optuna.storage import complete_counter_key, study_prefix, redis_url
from ex.utils.hpo.optuna.lanes import get_lane, LaneProfile
from ex.utils.hpo.optuna.study_config import load_config
from ex.utils.hpo.optuna.storage import create_or_load, _serialize_slice

logger = logging.getLogger(__name__)


def _env_export() -> str:
    """re-export the keeper's DPE roots inside the worker wrap.

    the wrap does `source ~/.bashrc`, which resets DPE_DATA_ROOT/DPE_CKPT_ROOT to
    the account default -- wrong when the keeper runs against an isolated root
    (e.g. a fresh-journal campaign under $DPE_DATA_ROOT/dpe-ns). baking the
    keeper's own values after the source keeps workers pointed at the same redis
    endpoint + data tree as the keeper. no-op when they already match bashrc.
    """
    parts = []
    for var in ("DPE_DATA_ROOT", "DPE_CKPT_ROOT"):
        val = os.environ.get(var)
        if val:
            parts.append(f"export {var}={val} && ")
    return "".join(parts)


# states that count toward target_trials; matches the MaxTrialsCallback in
# worker.py so the keeper and the workers agree on when a study is done.
_BUDGET_STATES = (TrialState.COMPLETE,)


def _read_complete_counter(experiment: str, method: str, slice=None) -> int:
    """read the redis O(1) COMPLETE-trial counter for one study.

    returns 0 on error. counter is incremented by worker callbacks and
    counts only trials completed since the counter was introduced.
    """
    try:
        import redis as _redis_pkg
        url = redis_url().removeprefix("redis://")
        host, port = url.split(":")
        r = _redis_pkg.Redis(host=host, port=int(port), socket_timeout=3)
        prefix = study_prefix(experiment, method, slice=slice)
        val = r.get(complete_counter_key(prefix))
        return int(val) if val is not None else 0
    except Exception:
        return 0


# cached optuna.Study handles per (experiment, method, slice). reusing the
# same handle across keeper cycles is the speed fix: optuna's JournalStorage
# tracks _last_log_id and replays only new entries on subsequent .trials
# fetches, so cycle N>=2 cost is proportional to NEW journal growth, not
# total journal size. before this, every count_complete call rebuilt the
# study from scratch and reparsed the full journal -- occupancy keeper got
# stuck in its first scan because each .trials call took 30+ s on a 300k+
# entry journal, blowing well past a single cycle's budget.
_study_handle_cache: dict[tuple, "optuna.Study"] = {}

# last successful COMPLETE count per (experiment, method, slice). used as a
# fallback when a fresh count_complete call times out. survives across
# cycles in-process; reset on keeper restart.
_count_cache: dict[tuple, int] = {}

# soft per-call ceiling. some occupancy studies have journals so large
# that one full create_or_load + study.trials can run for many minutes per
# study, freezing a whole cycle. when this fires we return the last-known
# count (or 0 if none) so the keeper still progresses through pending.
_COUNT_COMPLETE_TIMEOUT_S = 30


def _count_complete_timeout_handler(signum, frame):
    raise _CountTimeout()


class _CountTimeout(Exception):
    pass


def count_complete(experiment: str, method: str, slice=None, attempts: int = 3) -> int:
    """count COMPLETE trials in a study (the target_trials metric).

    PRUNED/FAIL trials do not count: target_trials is a budget of full-length
    completions, enforced worker-side by MaxTrialsCallback. on first call the
    study is loaded and cached; subsequent calls reuse the handle so optuna's
    journal storage only replays new log entries. a transient read error
    drops the cache entry and retries.

    when the underlying journal load + trial scan exceeds
    ``_COUNT_COMPLETE_TIMEOUT_S``, returns the last-known count (or 0 if
    no prior count cached) so the keeper does not stall on one slow study.
    """
    import signal
    key = (experiment, method, slice)
    last_err: Exception | None = None
    for attempt in range(attempts):
        # install timeout for this attempt only. signal.SIGALRM works on
        # the keeper's single-threaded main loop where count_complete is
        # called; no nested alarms exist in the dispatch path.
        old_handler = signal.signal(signal.SIGALRM, _count_complete_timeout_handler)
        signal.alarm(_COUNT_COMPLETE_TIMEOUT_S)
        try:
            study = _study_handle_cache.get(key)
            if study is None:
                study = create_or_load(experiment, method, slice=slice)
                _study_handle_cache[key] = study
            n = sum(1 for t in study.trials if t.state in _BUDGET_STATES)
            _count_cache[key] = n
            return n
        except _CountTimeout:
            cached = _count_cache.get(key)
            # try the redis counter fast-path — worker callbacks INCR it on
            # each COMPLETE, so the counter reflects trials completed since
            # its introduction. add to `cached` (which holds our last-known
            # journal-derived count, incl. pre-counter history) if present.
            counter_val = _read_complete_counter(experiment, method, slice)
            merged = (cached if cached is not None else 0) + counter_val
            logger.warning(
                "count_complete timed out for %s/%s/%s after %ds; "
                "returning cached=%s + counter=%s = %s",
                experiment, method, slice, _COUNT_COMPLETE_TIMEOUT_S,
                cached if cached is not None else "0",
                counter_val, merged,
            )
            # drop the optuna handle so next attempt re-creates it; the
            # _CountTimeout often fires inside JournalStorage internals
            # where the storage state may now be inconsistent.
            _study_handle_cache.pop(key, None)
            return merged
        except Exception as e:  # torn read; drop cache + retry
            _study_handle_cache.pop(key, None)
            last_err = e
            if attempt < attempts - 1:
                time.sleep(2)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    raise last_err


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


def job_name(experiment: str, method: str, lane_name: str, slice=None) -> str:
    """slurm job-name for a keeper-dispatched worker, embedding the lane and slice.

    format without slice: optk_{experiment}_{method}_{lane_name}.
    format with slice: optk_{experiment}_{method}_slice_{serialize(slice)}_{lane_name}.
    the lane is included so per-lane squeue -n counts never collide across lanes.
    """
    if slice is None:
        return f"optk_{experiment}_{method}_{lane_name}"
    serialized = _serialize_slice(slice)
    return f"optk_{experiment}_{method}_slice_{serialized}_{lane_name}"


def lane_b(lane: LaneProfile, method: str) -> int:
    """concurrent trials per worker task on `lane` for `method`.

    pinned `lane.batch_size`; else derived as cpus_per_task // cores, where
    cores is the lane-pinned `cores_per_trial` or the cores_registry default.
    """
    if lane.batch_size is not None:
        return max(1, lane.batch_size)
    cores = lane.cores_per_trial or get_cores_for_method(method)
    return max(1, lane.cpus_per_task // cores)


def estimate_in_flight(
    experiment: str, method: str, lanes: list[str], user: str | None, slice=None
) -> int:
    """estimate concurrent trials in the journal for (experiment, method[, slice]).

    sum lane_b(lane) * squeue_count(lane, name) across `lanes`. drops the
    contributions of lanes whose squeue lookup fails so a single transient
    error doesn't unblock dispatch beyond the cap.
    """
    total = 0
    for lane_name in lanes:
        try:
            lane = get_lane(lane_name)
            n_jobs = squeue_count(
                partition=lane.partition, user=user,
                name=job_name(experiment, method, lane_name, slice=slice),
            )
        except Exception:
            continue
        total += n_jobs * lane_b(lane, method)
    return total


def cull_study_jobs(experiment: str, method: str, lanes: list[str],
                    user: str | None, dry_run: bool, slice=None) -> int:
    """scancel all in-flight worker jobs for an at-target study, across lanes.

    once a study reaches target_trials the keeper stops dispatching to it, but
    its already-dispatched workers run their in-flight trials to completion --
    overshooting the target and holding lane slots the still-pending studies
    need. cancelling by job-name frees those slots immediately. cancelled
    in-flight trials land as orphaned RUNNING entries (reap_stale_trials cleans
    them) and do not count toward target, so accounting is unaffected.

    stateless and idempotent: gated on a live-job squeue check, so once a
    study's jobs are gone subsequent cycles are no-ops. returns the number of
    lanes for which a scancel was issued.
    """
    n = 0
    for lane_name in lanes:
        name = job_name(experiment, method, lane_name, slice=slice)
        try:
            partition = get_lane(lane_name).partition
            live = squeue_count(partition, user=user, name=name)
        except Exception as e:
            logger.warning("cull: squeue/lane lookup failed for %s: %s", name, e)
            continue
        if live == 0:
            continue
        cmd = ["scancel", "-n", name]
        if user:
            cmd += ["-u", user]
        if dry_run:
            logger.info("cull (dry-run): %s (%d live)", " ".join(cmd), live)
            n += 1
            continue
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("cull: scancel %s (%d in-flight job(s); study at target)",
                        name, live)
            n += 1
        except subprocess.CalledProcessError as e:
            logger.warning("cull: scancel failed for %s: %s", name, e.stderr.strip())
    return n


def dispatch(config_module: str, combo_index: int, experiment: str,
             method: str, lane_name: str, lane: LaneProfile, workdir: str,
             log_dir: str, dry_run: bool, slice=None) -> str:
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
      slice: optional slice identifier (recovered on submit side via combo_index).

    returns: job ID (str) if submitted; "dry-run" if dry_run=True.
    """
    name = job_name(experiment, method, lane_name, slice=slice)
    wrap = (
        f"source ~/.bashrc && conda activate ${{DPE_CONDA_ENV:-fac}} && {_env_export()}cd {workdir} && "
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
    # include --constraint if pinned (e.g. preempt lane pinned to newer gpus).
    if lane.constraint:
        cmd += ["--constraint", lane.constraint]

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
                   workdir: str, log_dir: str, dry_run: bool, slice=None) -> str:
    """sbatch one array job for a (experiment, method[, slice]) study.

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
      slice: optional slice identifier (recovered on submit side via combo_index).

    returns: job ID (str) if submitted; "dry-run" if dry_run=True.

    note: the array lane ALWAYS has gpus=0 and qos=""; no --gpus or --qos flags.
    """
    name = job_name(experiment, method, "array", slice=slice)
    n = max(1, n_workers)
    array_spec = f"0-{n - 1}%{n}"

    wrap = (
        f"source ~/.bashrc && conda activate ${{DPE_CONDA_ENV:-fac}} && {_env_export()}cd {workdir} && "
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


def dispatch_array_gpu(config_module: str, combo_index: int, experiment: str,
                       method: str, lane_name: str, lane: LaneProfile,
                       n_workers: int, workdir: str, log_dir: str,
                       dry_run: bool, slice=None) -> str:
    """sbatch one GPU array job for a study on a gpu-on-array lane.

    like dispatch_array but for a lane with gpus>0 on the array partition
    (e.g. "array_gpu"): emits --gpus and, when set, --constraint / --qos, and
    threads lane_name through so submit resolves the right lane. kept SEPARATE
    from dispatch_array so the cpu "array" lane's dispatch path is untouched.

    args mirror dispatch_array plus lane_name (the gpu-on-array lane name).
    returns: job ID (str) if submitted; "dry-run" if dry_run=True.
    """
    name = job_name(experiment, method, lane_name, slice=slice)
    n = max(1, n_workers)
    array_spec = f"0-{n - 1}%{n}"

    wrap = (
        f"source ~/.bashrc && conda activate ${{DPE_CONDA_ENV:-fac}} && {_env_export()}cd {workdir} && "
        f"python -m ex.utils.hpo.optuna.submit "
        f"--config {config_module} --lane {lane_name} --combo-index {combo_index}"
    )

    cmd = [
        "sbatch", "--parsable",
        f"--array={array_spec}",
        "--job-name", name,
        "--partition", lane.partition,
        "--time", lane.worker_walltime,
        "--cpus-per-task", str(lane.cpus_per_task),
        "--mem", lane.mem,
        "--gpus", str(lane.gpus),
        "--requeue",
        "--output", os.path.join(log_dir, f"{name}_%A_%a.out"),
        "--wrap", wrap,
    ]
    if lane.constraint:
        cmd += ["--constraint", lane.constraint]
    if lane.qos:
        cmd += ["--qos", lane.qos]

    if dry_run:
        logger.info("DRY-RUN dispatch_array_gpu %s: %s", name, " ".join(cmd))
        return "dry-run"

    try:
        jid = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        ).stdout.strip()
        logger.info("dispatched gpu array %s (size=%d) -> jobid %s", name, n, jid)
        return jid
    except subprocess.CalledProcessError as e:
        logger.error("dispatch_array_gpu failed for %s: %s", name, e.stderr)
        raise


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="multi-lane keeper for optuna hpo")
    p.add_argument("--config", required=True,
                   help="dotted StudyConfig module path")
    p.add_argument("--poll-interval", type=int, default=60,
                   help="seconds between cycles")
    p.add_argument("--max-dispatch-per-cycle", type=int, default=30,
                   help="cap submits per cycle to bound burstiness. raised to "
                        "30 (from 10) so a single cycle can fill both general "
                        "and preempt lanes; the per-lane iteration is "
                        "sequential (general first, then preempt), so a cap "
                        "of 10 was exhausted before preempt ever saw a slot.")
    p.add_argument("--max-cycles", type=int, default=0,
                   help="stop after N cycles (0 = run until all studies reach "
                        "target); use --max-cycles 1 with --dry-run for a "
                        "one-shot check")
    p.add_argument("--workdir", default=os.getcwd())
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--dry-run", action="store_true",
                   help="print sbatch lines instead of submitting")
    p.add_argument("--cull-on-target", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="scancel a study's in-flight workers once it reaches "
                        "target_trials, freeing lane slots for pending studies "
                        "(default: on; --no-cull-on-target to disable)")
    return p.parse_args()


def main() -> int:
    """keeper main loop.

    per cycle: (1) count COMPLETE trials per study; if all >= target_trials,
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
        "keeper start: experiment=%s methods=%s target_trials=%s lanes=%s",
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
        # iterate the (method x slice) product via the shared `resolve_combo`
        # contract from study_config.py; when config.slices is None, slices
        # degrades to [None] and behavior is byte-identical to today.
        pending: list[tuple[int, str, Hashable | None, int]] = []  # (combo_index, method, slice, n_terminal)
        count_errors = 0

        slices = config.slices if config.slices is not None else [None]
        combos = [(m, s) for m in config.methods for s in slices]

        for combo_index, (method, slice_) in enumerate(combos):
            try:
                n = count_complete(config.experiment, method, slice=slice_)
            except Exception as e:
                logger.warning(
                    "cycle %d: count failed for %s slice %s: %s",
                    cycle, method, slice_, e,
                )
                count_errors += 1
                continue
            if n < config.target_for(method):
                pending.append((combo_index, method, slice_, n))

        # (1b) cull in-flight workers for studies already at target -- BEFORE
        # the done-check break, so the final cycle (where pending transitions
        # to empty) still scancels every method's leftover array tasks. without
        # this, the last methods to reach target leave a long tail of slurm
        # array spawns that mint orphan RUNNING trial records for hours after
        # the keeper exits.
        if args.cull_on_target:
            # collect (method, slice) pairs still pending
            pending_ms = {(m, s) for (_, m, s, _) in pending}

            # cull all (method, slice) combos not in pending
            if config.slices:
                all_ms = [(m, s) for m in config.methods for s in config.slices]
            else:
                all_ms = [(m, None) for m in config.methods]

            for method, slice_ in all_ms:
                if (method, slice_) not in pending_ms:
                    try:
                        cull_study_jobs(
                            config.experiment, method, config.lanes, user,
                            args.dry_run, slice=slice_,
                        )
                    except Exception as e:
                        logger.warning(
                            "cull failed for %s slice %s: %s",
                            method, slice_, e
                        )

        # only conclude "done" when every count SUCCEEDED and all are at target.
        # an empty `pending` caused by count failures is NOT done -- retry.
        if not pending and count_errors == 0:
            n_studies = len(config.methods) * len(config.slices or [None])
            logger.info(
                "cycle %d: all %d studies reached target; done",
                cycle, n_studies,
            )
            break
        if not pending and count_errors:
            logger.warning(
                "cycle %d: no (method, slice) studies pending but %d count(s) "
                "failed -- not concluding done; retrying next cycle",
                cycle, count_errors,
            )
            time.sleep(args.poll_interval)
            continue

        # (2) for each lane: split lane.max_concurrent (the per-lane total
        # running cap) evenly across the studies still under target. each
        # study also carries a max_in_flight cap (concurrent trials summed
        # across lanes via lane_b) that throttles dispatch when the journal
        # is loaded -- protects against array-lane fan-out (B=32) saturating
        # the journal lock at scale.
        n_pending = len(pending)
        cycle_dispatched = 0
        study_slots: dict[str, int] = {}
        for (_, method, slice_, _) in pending:
            try:
                in_flight = estimate_in_flight(
                    config.experiment, method, config.lanes, user, slice=slice_
                )
            except Exception as e:
                logger.warning(
                    "cycle %d: in_flight estimate failed for %s: %s; "
                    "treating as 0", cycle, method, e,
                )
                in_flight = 0
            study_slots[method] = max(0, config.max_in_flight - in_flight)

        # fair-share ordering: sort pending by per-method study_slots
        # descending, so under-served methods (high slots) get first crack at
        # each cycle's dispatch budget AND any qos-cap room that frees. without
        # this, the keeper iterates pending in config-methods order; once the
        # qos cap is hit, methods later in the list (e.g. FMDRE_S2 at index 10,
        # Triangular* at 11-18) are starved indefinitely because the early
        # methods refill any freed slot before iteration reaches them. ties
        # broken by combo_index for determinism within a method.
        # tie-break rotates by cycle so methods with equal study_slots
        # don't always get tried in combo_index order. without this, methods
        # late in config.methods (TriangularVFM_V*, TriangularFMDRE at indices
        # 15-17) lose every QOS race to earlier methods that refill the cap
        # before iteration reaches them.
        _n_methods = max(1, len(config.methods))
        pending.sort(key=lambda r: (
            -study_slots.get(r[1], 0),
            (r[0] + cycle) % _n_methods,
            r[0],
        ))

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

            if lane.partition == "array" and lane.gpus == 0:
                # cpu-on-array lanes ("array", "array_small", ...): the array
                # partition only accepts job arrays, so any such lane must take
                # this --array path -- matching the gpus>0 branch below. keyed on
                # the profile, not the lane name, so new cpu-array lanes work.
                # array lane: one --array job per under-target study, sized to
                # min(per_study, study_slots[method] // lane_b). a study that
                # already has a live array job is skipped here.
                for (combo_index, method, slice_, n) in pending:
                    # gate gpu-required methods off cpu/array lanes only when
                    # the config opts in. mnist needs this; pendulum/occupancy
                    # don't FAIL on cpu so they stay unrestricted.
                    if config.gate_gpu_methods and lane.gpus == 0 and needs_gpu(method):
                        continue
                    try:
                        count = squeue_count(
                            partition="array", user=user,
                            name=job_name(config.experiment, method, "array", slice=slice_),
                        )
                    except Exception as e:
                        logger.warning("squeue failed for array %s: %s", method, e)
                        continue

                    if count != 0:
                        continue
                    b = lane_b(lane, method)
                    slot_n = study_slots.get(method, 0) // b
                    array_n = min(per_study, slot_n)
                    if array_n <= 0:
                        logger.info(
                            "skip array dispatch for %s: in_flight cap "
                            "(slots=%d, B=%d)",
                            method, study_slots.get(method, 0), b,
                        )
                        continue
                    try:
                        dispatch_array(
                            args.config, combo_index, config.experiment,
                            method, lane, array_n, args.workdir,
                            args.log_dir, args.dry_run, slice=slice_,
                        )
                        cycle_dispatched += 1
                        study_slots[method] -= array_n * b
                    except Exception as e:
                        logger.warning("dispatch_array failed for %s: %s", method, e)

            elif lane.partition == "array" and lane.gpus > 0:
                # gpu-on-array lane(s), e.g. array_gpu (B=1) and array_gpu_wide
                # (B=8): one gpu --array job per under-target study, same slot
                # logic as the cpu array branch but dispatched via
                # dispatch_array_gpu (adds --gpus/--constraint). the cpu "array"
                # branch above (gpus=0) is deliberately left untouched.
                for (combo_index, method, slice_, n) in pending:
                    try:
                        count = squeue_count(
                            partition=lane.partition, user=user,
                            name=job_name(config.experiment, method, lane_name, slice=slice_),
                        )
                    except Exception as e:
                        logger.warning("squeue failed for array_gpu %s: %s", method, e)
                        continue
                    if count != 0:
                        continue
                    b = lane_b(lane, method)
                    slot_n = study_slots.get(method, 0) // b
                    array_n = min(per_study, slot_n)
                    if array_n <= 0:
                        logger.info(
                            "skip array_gpu dispatch for %s: in_flight cap "
                            "(slots=%d, B=%d)",
                            method, study_slots.get(method, 0), b,
                        )
                        continue
                    try:
                        dispatch_array_gpu(
                            args.config, combo_index, config.experiment,
                            method, lane_name, lane, array_n, args.workdir,
                            args.log_dir, args.dry_run, slice=slice_,
                        )
                        cycle_dispatched += 1
                        study_slots[method] -= array_n * b
                    except Exception as e:
                        logger.warning("dispatch_array_gpu failed for %s: %s", method, e)

            else:
                # per-worker lanes (preempt, cpu, general): top each study up to
                # its per_study share, bounded by the per-cycle burst cap.
                for (combo_index, method, slice_, n) in pending:
                    if cycle_dispatched >= args.max_dispatch_per_cycle:
                        break
                    # see array-lane block above; config-driven gate.
                    if config.gate_gpu_methods and lane.gpus == 0 and needs_gpu(method):
                        continue

                    try:
                        running = squeue_count(
                            partition=lane.partition, user=user,
                            name=job_name(config.experiment, method, lane_name, slice=slice_),
                        )
                    except Exception as e:
                        logger.warning(
                            "squeue by name failed for %s on %s: %s",
                            method, lane_name, e,
                        )
                        continue

                    b = lane_b(lane, method)
                    deficit = per_study - running
                    deficit = min(deficit, config.target_for(method) - n)
                    deficit = min(
                        deficit, args.max_dispatch_per_cycle - cycle_dispatched
                    )
                    # in_flight cap: convert remaining slots to a per-lane
                    # task count via B and clamp deficit. floor at 0.
                    deficit = min(deficit, study_slots.get(method, 0) // b)

                    for _ in range(max(0, deficit)):
                        try:
                            dispatch(
                                args.config, combo_index, config.experiment,
                                method, lane_name, lane, args.workdir,
                                args.log_dir, args.dry_run, slice=slice_,
                            )
                            cycle_dispatched += 1
                            study_slots[method] -= b
                        except Exception as e:
                            logger.warning(
                                "dispatch failed for %s on %s: %s",
                                method, lane_name, e,
                            )
                            break

        logger.info(
            "cycle %d: %d (method, slice) below target; dispatched %d worker(s)",
            cycle, len(pending), cycle_dispatched,
        )
        time.sleep(args.poll_interval)

    # done-marker written by redis_server.sh to done/<cfg>, keyed on CONFIG
    # module only, not per (method, slice). when slices are present, one slow
    # slice blocks the cfg-level holdout launch. acceptable for peak-era
    # campaign scale; flag for future scale.
    logger.info("keeper exiting normally after %d cycle(s)", cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
