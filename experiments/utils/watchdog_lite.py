"""general-partition queue keeper. lightweight cousin of watchdog.py with no
double-ended cpu_array drain. pops sbatch invocations from a unified queue
file, rewrites them to target slurm `general` (no preemption, no requeue),
and dispatches while respecting per-user cap, total-cluster cap, walltime
caps from walltime_caps.py, and slow-node exclusion.

retains the workflow-pair tick loop from the omnibus so queue lines get
appended in-band (broad -> refined -> holdout -> persist). retains optional
orphan scan. no cpu_array_relaunch, no back-pop, no per-method controllers.
"""

import argparse, collections, fcntl, json, logging, os, random, re
import signal, subprocess, sys, time
from pathlib import Path
from typing import Optional

from experiments.utils.walltime_caps import cap_for
# reuse helpers from the omnibus to avoid duplication
from experiments.utils.watchdog import (
    PopStrategy,
    parse_spec_line,
    squeue_count,
    squeue_alive_jids,
    pop_line_atomic,
    submit_sbatch,
    update_node_stats,
    detect_slow_nodes,
    append_exclude,
    read_exclude,
    scan_for_orphans,
    _collect_experiment_output_dirs,
    _log_event,
    _append_submitted_tsv,
    _queue_empty,
)

LOGGER = logging.getLogger("watchdog_lite")


GPU_CONSTRAINT = "RTX_PRO_6000|L40S|6000Ada|A100_80GB|A100_40GB"


def render_sbatch(template: str, time_str: str, exclude_str: str,
                  partition: str = "general",
                  gpu_constraint: str = GPU_CONSTRAINT) -> str:
    """expand placeholders + retarget partition + pin gpu constraint.

    {time} -> bare HH:MM:SS. {exclude} -> comma-joined nodes; if empty,
    drop the entire `--exclude=...` flag. `--partition=preempt` rewritten
    to `--partition=<partition>`. `--requeue` stripped (general jobs do
    not get preempted, requeue is a no-op). `--constraint=<gpu_constraint>`
    inserted right after the rewritten partition flag, so general-partition
    jobs only land on the supported gpu types.
    """
    out = template.replace("{time}", time_str)
    if exclude_str:
        out = out.replace("{exclude}", exclude_str)
    else:
        out = out.replace(" --exclude={exclude}", "")
        out = out.replace("--exclude={exclude} ", "")
        out = out.replace("--exclude={exclude}", "")
    out = out.replace("--partition=preempt",
                     f"--partition={partition} --constraint='{gpu_constraint}'")
    # keep --requeue when actually targeting preempt (preempt jobs need it
    # to restart after eviction); strip for general where it's a no-op.
    if partition != "preempt":
        out = out.replace(" --requeue", "")
    return out


def _load_state(state_file: Path) -> tuple[set, set, list]:
    """load state.json: submitted_jids, seen_jid_node, workflow_states.

    missing or parse error -> (set(), set(), []).
    """
    if not state_file.exists():
        return (set(), set(), [])
    try:
        data = json.loads(state_file.read_text())
        jids = set(data.get("submitted_jids", []))
        seen = {tuple(item) for item in data.get("seen_jid_node", [])}
        wf_states = data.get("workflow_states", [])
        return (jids, seen, wf_states)
    except (json.JSONDecodeError, ValueError):
        return (set(), set(), [])


def _save_state(state_file: Path, submitted_jids: set, seen_jid_node: set,
                workflow_states: Optional[list] = None) -> None:
    """atomic JSON dump: tmp, write, flush, fsync, replace."""
    data = {
        "submitted_jids": sorted(submitted_jids),
        "seen_jid_node": sorted([list(item) for item in seen_jid_node]),
        "workflow_states": workflow_states or [],
    }
    tmp_path = state_file.parent / f"{state_file.name}.tmp"
    with open(tmp_path, "w") as tmp_fd:
        json.dump(data, tmp_fd)
        tmp_fd.flush()
        os.fsync(tmp_fd.fileno())
    os.replace(tmp_path, state_file)


def main() -> None:
    """argparse + dispatch loop targeting partition `general`.

    startup:
    1. parse args.
    2. configure root logger (file INFO, stderr WARNING).
    3. install SIGTERM handler.
    4. resolve state_file, submitted_tsv defaults under <exclude_file.parent>.
    5. --reset-exclude-on-startup clears exclude file.
    6. load state.
    7. init node_stats, last_sacct_cycle, cycle, rng.
    8. emit STARTUP event.

    main loop:
    - tick workflow pairs (each appends queue lines per stage).
    - if queue empty, sleep + continue.
    - get squeue counts for `general`. if capped, sleep + continue.
    - pop headroom lines, submit each to general, log DISPATCH, append tsv.
    - every 10 cycles: sacct update, detect slow nodes, save state.
    - sleep poll_interval.

    SIGTERM/KeyboardInterrupt -> save state + exit 0.
    """
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-file", type=Path, required=True)
    parser.add_argument("--my-cap", type=int, default=80)
    parser.add_argument("--total-cap", type=int, default=200)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--slow-multiplier", type=float, default=2.0)
    parser.add_argument("--slow-min-jobs", type=int, default=3)
    parser.add_argument("--exclude-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--reset-exclude-on-startup", action="store_true")
    parser.add_argument("--sacct-window", default="2hours")
    parser.add_argument("--max-dispatch-per-cycle", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=-1)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--partition", default="general",
                        help="slurm partition for both squeue cap check and "
                             "sbatch dispatch (default: general)")
    parser.add_argument(
        "--pop-strategy", choices=["front", "random"], default="front",
        help="queue drain order: 'front' (sorted-queue bias) or 'random'"
    )
    parser.add_argument(
        "--orphan-scan-interval", type=int, default=60,
        help="cycles between orphan scans; 0 disables (default 60)"
    )
    parser.add_argument(
        "--workflow-pairs", type=Path, default=None,
        help="JSON file: list of {method, experiment, output_dir, "
             "budget?, seed?} objects. when set, watchdog ticks each "
             "pair's broad->refined->holdout->persist state machine each cycle"
    )
    parser.add_argument(
        "--workflow-tick-interval", type=int, default=1,
        help="cycles between workflow ticks (default 1)"
    )
    args = parser.parse_args()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(args.log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.WARNING)
    root_logger.addHandler(stream_handler)

    if args.state_file is None:
        args.state_file = args.exclude_file.parent / "state.json"
    submitted_tsv = args.exclude_file.parent / "submitted.tsv"

    if args.reset_exclude_on_startup:
        args.exclude_file.write_text("")

    submitted_jids, seen_jid_node, prior_workflow_states = _load_state(args.state_file)

    from experiments.utils.hpo import workflow_runner as wfr
    workflow_pairs: list = []
    if args.workflow_pairs is not None:
        workflow_pairs = wfr.load_pairs(args.workflow_pairs)
        wfr.restore_states(workflow_pairs, prior_workflow_states)
        _log_event("WORKFLOW_INIT", n_pairs=len(workflow_pairs),
                   stages={p.stage.value: 0 for p in workflow_pairs})

    node_stats = {}
    last_sacct_cycle = 0
    cycle = 0

    strategy = PopStrategy(args.pop_strategy)
    if strategy == PopStrategy.RANDOM:
        rng = random.Random(args.shuffle_seed) if args.shuffle_seed >= 0 else random.SystemRandom()
    else:
        rng = None

    _log_event("STARTUP",
               variant="lite",
               partition=args.partition,
               queue_file=str(args.queue_file),
               exclude_file=str(args.exclude_file),
               my_cap=args.my_cap,
               total_cap=args.total_cap,
               poll_interval=args.poll_interval,
               dry_run=args.dry_run,
               pop_strategy=args.pop_strategy)

    def _on_sigterm(signum, frame):
        LOGGER.info("SIGTERM received; saving state and exiting")
        _save_state(args.state_file, submitted_jids, seen_jid_node,
                   wfr.serialize_states(workflow_pairs) if workflow_pairs else None)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        empty_streak = 0
        all_terminal_logged = False
        while True:
            cycle += 1

            # tick workflow pairs first; they may append to the queue this cycle.
            if workflow_pairs and cycle % args.workflow_tick_interval == 0:
                changed_count = 0
                for p in workflow_pairs:
                    if p.tick(args.queue_file):
                        changed_count += 1
                        _log_event("WORKFLOW_TICK", method=p.method,
                                   experiment=p.experiment,
                                   stage=p.stage.value, error=p.error)
                if changed_count > 0:
                    _save_state(args.state_file, submitted_jids, seen_jid_node,
                               wfr.serialize_states(workflow_pairs))
                if (not all_terminal_logged
                        and all(p.stage.value in ("done", "error") for p in workflow_pairs)):
                    _log_event("WORKFLOW_ALL_TERMINAL",
                               done=sum(1 for p in workflow_pairs if p.stage.value == "done"),
                               error=sum(1 for p in workflow_pairs if p.stage.value == "error"))
                    all_terminal_logged = True

            if _queue_empty(args.queue_file, args.state_file.parent / "queue.lock"):
                empty_streak += 1
                if empty_streak == 1 or empty_streak % 20 == 0:
                    _log_event("QUEUE_EMPTY", cycle=cycle, streak=empty_streak)
                time.sleep(args.poll_interval)
                continue
            empty_streak = 0

            try:
                my_pending = squeue_count(user=os.environ["USER"], partition=args.partition)
                total_pending = squeue_count(partition=args.partition)
            except RuntimeError:
                _log_event("SQUEUE_FAIL")
                time.sleep(args.poll_interval)
                continue

            if my_pending >= args.my_cap or total_pending >= args.total_cap:
                _log_event("SLEEP_CAP", my=my_pending, total=total_pending)
                time.sleep(args.poll_interval)
                continue

            headroom = min(args.my_cap - my_pending,
                          args.total_cap - total_pending,
                          args.max_dispatch_per_cycle)
            excl_str = read_exclude(args.exclude_file)
            excl_n = excl_str.count(",") + 1 if excl_str else 0

            for _ in range(headroom):
                line = pop_line_atomic(
                    args.queue_file,
                    args.state_file.parent / "queue.lock",
                    strategy,
                    rng=rng,
                )
                if line is None:
                    break
                parsed = parse_spec_line(line)
                if parsed is None:
                    continue
                method, tag, raw_cmd = parsed
                # defensive: skip if a competing consumer already produced
                # the result file. atomic claim should prevent this in normal
                # flow but covers restarts / queue edits.
                m_cfg = re.search(r"trial_(\d+)\.json", raw_cmd)
                m_out = re.search(r"--output-dir\s+(\S+)", raw_cmd)
                m_stg = re.search(r'--stage\s+([^\s"]+)', raw_cmd)
                if m_cfg and m_out and m_stg:
                    expected = Path(m_out.group(1)) / m_stg.group(1) / f"trial_{m_cfg.group(1)}.json"
                    if expected.exists():
                        _log_event("DUPLICATE_SKIP", trial_id=m_cfg.group(1),
                                   method=method, stage=m_stg.group(1))
                        continue
                # use preempt cap dict — same per-method ceiling, general just
                # avoids preemption so the cap is a hard upper bound, not a
                # priority hint.
                walltime = cap_for(method, partition="preempt", pilot_tag=tag)
                cmd = render_sbatch(raw_cmd, walltime, excl_str,
                                   partition=args.partition)
                jid = submit_sbatch(cmd, dry_run=args.dry_run)
                if jid is None:
                    continue
                submitted_jids.add(jid)
                _log_event("DISPATCH", jid=jid, method=method, tag=tag,
                          walltime=walltime, partition=args.partition,
                          excl_n=excl_n)
                _append_submitted_tsv(submitted_tsv, jid, method, tag,
                                     walltime, str(excl_n))

            if cycle - last_sacct_cycle >= 10:
                update_node_stats(node_stats, seen_jid_node, window=args.sacct_window)
                new_slow = detect_slow_nodes(node_stats, args.slow_multiplier,
                                            args.slow_min_jobs)
                if new_slow:
                    append_exclude(args.exclude_file, new_slow)
                    _log_event("SLOW_NODES", nodes=sorted(new_slow))
                last_sacct_cycle = cycle
                _save_state(args.state_file, submitted_jids, seen_jid_node,
                           wfr.serialize_states(workflow_pairs) if workflow_pairs else None)

            if args.orphan_scan_interval > 0 and cycle % args.orphan_scan_interval == 0:
                output_dirs = _collect_experiment_output_dirs(args.queue_file)
                if output_dirs:
                    live_jids = submitted_jids | squeue_alive_jids(args.partition)
                    orphans = scan_for_orphans(
                        output_dirs=output_dirs,
                        queue_file=args.queue_file,
                        live_jids=live_jids,
                        lock_file=args.state_file.parent / "queue.lock",
                    )
                    if orphans:
                        _log_event("ORPHANS_REQUEUED", count=len(orphans),
                                   trial_ids=orphans[:20])

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        _save_state(args.state_file, submitted_jids, seen_jid_node,
                   wfr.serialize_states(workflow_pairs) if workflow_pairs else None)
        sys.exit(0)


if __name__ == "__main__":
    main()
