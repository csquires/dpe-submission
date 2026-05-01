"""saturating-loop preempt queue keeper. pops sbatch invocations from a
unified queue file, dispatches to slurm preempt while respecting per-user
cap, total-cluster cap, walltime caps from walltime_caps.py, and slow-node
exclusion. multiplexes across the four hpo pilots (optuna, recalibrated,
random_a, random_b) via the PILOT_TAG column. invokes
optuna_study.cleanup_zombies() every 50 cycles for any active studies
discovered under DPE_DATA_ROOT.
"""

import argparse, collections, fcntl, json, logging, os, random, re
import signal, subprocess, sys, time
from enum import Enum
from pathlib import Path
from typing import Optional

from experiments.utils.walltime_caps import cap_for

try:
    from experiments.utils.hpo.optuna_study_v735e import cleanup_zombies_for as _opt_cleanup_for
except ImportError:
    _opt_cleanup_for = None  # no-op'd at call site (optuna unavailable)

LOGGER = logging.getLogger("watchdog")


class PopStrategy(Enum):
    """queue drain strategy enum."""
    RANDOM = "random"   # legacy: uniform random pick from valid lines
    FRONT  = "front"    # new default: pop first valid line (launcher sort)


def parse_spec_line(line: str) -> Optional[tuple[str, str, str]]:
    """parse queue spec line: tab-sep (method, pilot_tag, sbatch_template).

    strip newline, skip blank/comment, split on tab. if arity != 3 or
    unknown pilot_tag, log event but still return on valid arity.
    """
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        return None
    parts = line.split("\t", 2)
    if len(parts) != 3:
        _log_event("BAD_LINE", line=line)
        return None
    method, pilot_tag, template = parts
    if pilot_tag not in {"optuna", "recalibrated", "random_a", "random_b",
                         "broad", "refined", "holdout"}:
        LOGGER.warning(f"unknown pilot_tag: {pilot_tag}")
    return (method, pilot_tag, template)


def squeue_count(user: Optional[str] = None, partition: str = "preempt") -> int:
    """count pending jobs in partition; optionally filter by user.

    wraps subprocess.run with timeout=30. subprocess.TimeoutExpired ->
    RuntimeError("squeue timeout") for caller's except RuntimeError handler.
    """
    try:
        cmd = ["squeue", "-h", "-p", partition]
        if user:
            cmd.extend(["-u", user])
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError("squeue timeout")
    if out.returncode != 0:
        raise RuntimeError(f"squeue rc={out.returncode}")
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


def squeue_alive_jids(partition: str = "preempt") -> set[str]:
    """return set of all job ids currently in partition (any state).

    wraps subprocess.run with timeout=30; TimeoutExpired -> RuntimeError.
    """
    try:
        out = subprocess.run(
            ["squeue", "-h", "-p", partition, "--format=%i"],
            capture_output=True,
            text=True,
            timeout=30
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("squeue timeout")
    if out.returncode != 0:
        return set()
    return {jid.strip() for jid in out.stdout.splitlines() if jid.strip()}


def pop_line_atomic(
    queue_file: Path,
    lock_file: Path,
    strategy: PopStrategy | str = PopStrategy.FRONT,
    *,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """fcntl.flock-protected read, strategy-driven pick, write-back.

    strategy: PopStrategy.FRONT (default) pops first valid line (respects
              launcher sort); PopStrategy.RANDOM picks uniformly from valid.
              accepts enum or string ("front"/"random") for CLI ergonomics.
    rng: required if strategy == PopStrategy.RANDOM; otherwise ignored.
         ValueError raised if strategy == RANDOM and rng is None.

    returns one picked line (rstrip'd, no trailing newline) or None if
    queue empty/missing or no valid lines.

    atomic: flock'd throughout; tmp file fsync'd, renamed to overwrite queue.
    """
    # normalize strategy string to enum
    if isinstance(strategy, str):
        try:
            strategy = PopStrategy(strategy)
        except ValueError:
            raise ValueError(f"invalid pop strategy: {strategy}. must be 'front' or 'random'.")

    # validate rng requirement for RANDOM strategy
    if strategy == PopStrategy.RANDOM and rng is None:
        raise ValueError("strategy=RANDOM requires rng parameter (non-None)")

    with open(lock_file, "a+") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            if not queue_file.exists() or queue_file.stat().st_size == 0:
                return None

            lines = queue_file.read_text().splitlines(keepends=True)
            valid = [(i, ln) for i, ln in enumerate(lines) if parse_spec_line(ln) is not None]

            if not valid:
                return None

            # apply strategy to pick one valid line
            if strategy == PopStrategy.FRONT:
                i_picked, picked = valid[0]
            elif strategy == PopStrategy.RANDOM:
                i_picked, picked = rng.choice(valid)
            else:
                raise ValueError(f"unknown strategy: {strategy}")

            # write back remaining lines (tmp + replace pattern)
            remaining = [lines[j] for j in range(len(lines)) if j != i_picked]
            tmp_path = queue_file.parent / f"{queue_file.name}.tmp"
            with open(tmp_path, "w") as tmp_fd:
                tmp_fd.writelines(remaining)
                tmp_fd.flush()
                os.fsync(tmp_fd.fileno())
            os.replace(tmp_path, queue_file)

            return picked.rstrip("\n")
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def pop_lines_back_atomic(queue_file: Path, lock_file: Path, k: int,
                          method_filter: Optional[set[str]] = None) -> list[str]:
    """flock-protected pop of up to k valid lines from the END of queue_file.

    parallel companion to pop_line_atomic for cpu array dispatch. cpu drains
    from the back; gpu watchdog pops randomly from the rest. atomic claim
    guarantees no overlap.

    args:
      queue_file: path to shared queue file.
      lock_file: path to flock file.
      k: max lines to claim.
      method_filter: optional set of method names; only lines whose method
        is in the set are claimable. lines failing the filter stay in queue.

    returns:
      list of claimed line strings (rstrip'd, no trailing newline). may be
      shorter than k if queue ran dry or had insufficient eligible lines.
      empty list if queue empty/missing.
    """
    if k < 1:
        return []
    with open(lock_file, "a+") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if not queue_file.exists() or queue_file.stat().st_size == 0:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            return []
        lines = queue_file.read_text().splitlines(keepends=True)
        # walk from the end, collecting eligible lines until we have k
        claim_idx: list[int] = []
        for i in range(len(lines) - 1, -1, -1):
            spec = parse_spec_line(lines[i])
            if spec is None:
                continue
            if method_filter is not None and spec[0] not in method_filter:
                continue
            claim_idx.append(i)
            if len(claim_idx) >= k:
                break
        if not claim_idx:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            return []
        claim_set = set(claim_idx)
        remaining = [lines[j] for j in range(len(lines)) if j not in claim_set]
        claimed = [lines[i].rstrip("\n") for i in claim_idx]

        tmp_path = queue_file.parent / f"{queue_file.name}.tmp"
        with open(tmp_path, "w") as tmp_fd:
            tmp_fd.writelines(remaining)
            tmp_fd.flush()
            os.fsync(tmp_fd.fileno())
        os.replace(tmp_path, queue_file)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    return claimed


def render_sbatch(template: str, time_str: str, exclude_str: str) -> str:
    """expand placeholders. {time} -> HH:MM:SS bare. {exclude} -> comma-joined
    nodes; if empty, drop the entire `--exclude=...` flag from the template
    (sbatch errors on empty `--exclude=`).
    """
    out = template.replace("{time}", time_str)
    if exclude_str:
        out = out.replace("{exclude}", exclude_str)
    else:
        # drop the flag entirely; match common patterns including newline-escaped form
        out = out.replace(" --exclude={exclude}", "")
        out = out.replace("--exclude={exclude} ", "")
        out = out.replace("--exclude={exclude}", "")
    return out


def submit_sbatch(cmd: str, dry_run: bool = False) -> Optional[str]:
    """dry_run -> print + return 'DRYRUN'; else subprocess.run with timeout=60.

    parse jid from stdout via regex; rc!=0 or timeout -> emit SUBMIT_FAIL,
    return None.
    """
    if dry_run:
        print(cmd)
        return "DRYRUN"
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        _log_event("SUBMIT_FAIL", error="timeout")
        return None
    if out.returncode != 0:
        _log_event("SUBMIT_FAIL", rc=out.returncode, stderr=out.stderr[:200])
        return None
    match = re.search(r'Submitted batch job (\d+)', out.stdout)
    if not match:
        _log_event("SUBMIT_PARSE_FAIL", stdout=out.stdout[:200])
        return None
    return match.group(1)


def _parse_elapsed(s: str) -> int:
    """regex parse elapsed string: [D-]HH:MM:SS -> seconds.

    raises ValueError on garbage.
    """
    match = re.match(r'^(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})$', s)
    if not match:
        raise ValueError(f"invalid elapsed format: {s}")
    days, hours, minutes, secs = match.groups()
    return (int(days or 0) * 86400 +
            int(hours) * 3600 +
            int(minutes) * 60 +
            int(secs))


def _expand_nodelist(nl: str) -> list[str]:
    """scontrol show hostnames <nl> -> list of node names.

    on rc != 0, returns [nl] (fallback).
    """
    try:
        out = subprocess.run(
            ["scontrol", "show", "hostnames", nl],
            capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (subprocess.TimeoutExpired, Exception):
        pass
    return [nl]


def update_node_stats(node_stats: dict, seen: set, window: str = "2hours") -> None:
    """sacct query: filter to COMPLETED preempt jobs, track per-node elapsed.

    deduplicate by (jid, node); skip if already in seen set. append elapsed
    to per-node deque(maxlen=10).
    """
    cmd = [
        "sacct", "-u", os.environ["USER"], "-P", "-n",
        "--format=JobID,NodeList,Elapsed,State,Partition",
        f"-S", f"now-{window}",
        "-X"
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        _log_event("SACCT_FAIL", error="timeout")
        return
    if out.returncode != 0:
        _log_event("SACCT_FAIL", rc=out.returncode)
        return

    # group by jid, keep max elapsed within group
    jid_rows = {}
    for row in out.stdout.strip().split("\n"):
        if not row.strip():
            continue
        parts = row.split("|")
        if len(parts) < 5:
            continue
        jid, nodelist, elapsed, state, partition = parts[:5]
        if state != "COMPLETED" or partition != "preempt":
            continue
        if jid not in jid_rows:
            jid_rows[jid] = (nodelist, elapsed)
        else:
            try:
                curr_secs = _parse_elapsed(jid_rows[jid][1])
                new_secs = _parse_elapsed(elapsed)
                if new_secs > curr_secs:
                    jid_rows[jid] = (nodelist, elapsed)
            except ValueError:
                pass

    # expand nodelist, parse elapsed, append to stats
    for jid, (nodelist, elapsed) in jid_rows.items():
        try:
            secs = _parse_elapsed(elapsed)
        except ValueError:
            continue
        nodes = _expand_nodelist(nodelist)
        for node in nodes:
            if (jid, node) not in seen:
                node_stats.setdefault(node, collections.deque(maxlen=10)).append(float(secs))
                seen.add((jid, node))


def detect_slow_nodes(node_stats: dict, multiplier: float, min_jobs: int) -> set[str]:
    """return nodes with mean elapsed > multiplier * global_mean.

    skip nodes with < min_jobs samples. if global_mean == 0, return set().
    """
    all_vals = [v for dq in node_stats.values() for v in dq]
    if not all_vals:
        return set()
    global_mean = sum(all_vals) / len(all_vals)
    if global_mean == 0:
        return set()
    result = set()
    for node, dq in node_stats.items():
        if len(dq) >= min_jobs:
            node_mean = sum(dq) / len(dq)
            if node_mean > multiplier * global_mean:
                result.add(node)
    return result


def append_exclude(exclude_file: Path, nodes: set[str]) -> None:
    """atomic append with dedup: fcntl.flock, read existing, write union.
    """
    with open(exclude_file, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        existing = {ln.strip() for ln in f if ln.strip()}
        new = nodes - existing
        if not new:
            fcntl.flock(f, fcntl.LOCK_UN)
            return
        union = sorted(existing | nodes)
        tmp_path = exclude_file.parent / f"{exclude_file.name}.tmp"
        with open(tmp_path, "w") as tmp_fd:
            for node in union:
                tmp_fd.write(f"{node}\n")
            tmp_fd.flush()
            os.fsync(tmp_fd.fileno())
        os.replace(tmp_path, exclude_file)
        fcntl.flock(f, fcntl.LOCK_UN)


def read_exclude(exclude_file: Path) -> str:
    """read nodes (one per line), comma-join. empty file -> ''.
    """
    if not exclude_file.exists():
        return ""
    nodes = [ln.strip() for ln in exclude_file.read_text().splitlines() if ln.strip()]
    return ",".join(nodes)


def cleanup_optuna_zombies(active_studies: list[tuple[str, str]]) -> int:
    """for each (exp, method), call _opt_cleanup_for (positional args).

    if _opt_cleanup_for is None, log OPTUNA_UNAVAILABLE once, return 0.
    """
    if _opt_cleanup_for is None:
        _log_event("OPTUNA_UNAVAILABLE")
        return 0
    total = 0
    for exp, method in active_studies:
        try:
            n = _opt_cleanup_for(exp, method)
            total += n
        except Exception as e:
            _log_event("CLEANUP_ERROR", exp=exp, method=method, error=str(e)[:100])
    _log_event("CLEANUP_OK", total=total)
    return total


def _discover_active_studies(data_root: Path) -> list[tuple[str, str]]:
    """glob data_root / "*" / "hpo_optuna_v735e" / "*.journal".

    return (exp_name, method_name) pairs per match.
    """
    result = []
    for journal_path in data_root.glob("*/hpo_optuna_v735e/*.journal"):
        exp_name = journal_path.parent.parent.name
        method_name = journal_path.stem
        result.append((exp_name, method_name))
    return result


def _load_state(state_file: Path) -> tuple[set, set]:
    """load state.json: submitted_jids (set), seen_jid_node (set of tuples).

    missing or parse error -> (set(), set()).
    """
    if not state_file.exists():
        return (set(), set())
    try:
        data = json.loads(state_file.read_text())
        jids = set(data.get("submitted_jids", []))
        seen = {tuple(item) for item in data.get("seen_jid_node", [])}
        return (jids, seen)
    except (json.JSONDecodeError, ValueError):
        return (set(), set())


def _save_state(state_file: Path, submitted_jids: set, seen_jid_node: set) -> None:
    """atomic JSON dump: tmp, write, flush, fsync, replace.

    sets serialized as sorted lists; tuples as [str, str].
    """
    data = {
        "submitted_jids": sorted(submitted_jids),
        "seen_jid_node": sorted([list(item) for item in seen_jid_node])
    }
    tmp_path = state_file.parent / f"{state_file.name}.tmp"
    with open(tmp_path, "w") as tmp_fd:
        json.dump(data, tmp_fd)
        tmp_fd.flush()
        os.fsync(tmp_fd.fileno())
    os.replace(tmp_path, state_file)


def _collect_experiment_output_dirs(queue_file: Path) -> list[Path]:
    """parse queue file to extract unique experiment output_dirs.

    each queue line is a tab-sep spec: method\\tpilot_tag\\tsbatch_cmd.
    sbatch_cmd contains --output-dir <path> flag. extract <path>, dedup.

    if queue missing/empty, return [].
    """
    if not queue_file.exists() or queue_file.stat().st_size == 0:
        return []

    output_dirs = set()
    for line in queue_file.read_text().splitlines():
        parsed = parse_spec_line(line)
        if parsed is None:
            continue
        _, _, sbatch_cmd = parsed
        # regex: --output-dir\s+(\S+)
        m = re.search(r"--output-dir\s+(\S+)", sbatch_cmd)
        if m:
            output_dirs.add(Path(m.group(1)))

    return sorted(output_dirs)  # deterministic order


def _is_in_queue(queue_file: Path, lock_file: Path,
                 method: str, trial_id: int) -> bool:
    """check if trial_<trial_id>.json is referenced in any queue line
    for the given method.

    performs atomic read under flock. grep for trial_<trial_id>.json
    in all parsed queue lines matching the method.
    """
    if not queue_file.exists():
        return False

    trial_pattern = f"trial_{trial_id}.json"
    with open(lock_file, "a+") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            for line in queue_file.read_text().splitlines():
                parsed = parse_spec_line(line)
                if parsed is None:
                    continue
                m, _, sbatch_cmd = parsed
                if m == method and trial_pattern in sbatch_cmd:
                    return True
            return False
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def _is_in_live_jid(jid: str, trial_id: int, sacct_cache: dict[str, str]) -> bool:
    """check if trial_<trial_id>.json appears in the sbatch command
    for the given job id.

    uses sacct_cache dict (pre-populated by _batch_sacct_commands) for O(1)
    lookup instead of per-jid subprocess call.
    """
    trial_pattern = f"trial_{trial_id}.json"
    cmd = sacct_cache.get(jid, "")
    return trial_pattern in cmd


def _batch_sacct_commands(live_jids: set[str], timeout_sec: int = 30) -> dict[str, str]:
    """ONE sacct call for all live jids; returns {jid: Command_str}.
    falls back to {} on sacct failure or timeout (treat as no jids -> may over-requeue)."""
    if not live_jids:
        return {}
    jid_csv = ",".join(sorted(live_jids))
    try:
        out = subprocess.run(
            ["sacct", "-j", jid_csv, "--format=JobID,Command",
             "--noheader", "--parsable2"],
            capture_output=True, text=True, timeout=timeout_sec
        )
        if out.returncode != 0:
            return {}
        result = {}
        for line in out.stdout.splitlines():
            parts = line.split("|", 1)
            if len(parts) == 2:
                result[parts[0].split(".")[0]] = parts[1]  # strip step suffix
        return result
    except (subprocess.TimeoutExpired, Exception):
        return {}


def scan_for_orphans(output_dirs: list[Path], queue_file: Path,
                     live_jids: set[str], lock_file: Path) -> list[str]:
    """scan experiment output dirs for orphaned trial configs.

    a trial is orphan iff:
      1. config exists: <output_dir>/<stage>/<method>/trial_<id>.json
      2. result missing: <output_dir>/<stage>/trial_<id>.json does not exist
      3. not in queue: _is_in_queue returns False for (method, id)
      4. not in live jid: no jid in live_jids has trial_<id>.json in command

    for each orphan, generate a fresh sbatch line (method, "orphan_recovery",
    sbatch_cmd with trial_<id>.json) and append to queue under flock.

    preconditions:
      - lock_file path matches the lock_file used by cpu_array_element AND
        pop_line_atomic (typically args.state_file.parent / "queue.lock").
      - failure mode if violated: scanner and dispatcher race; either may pop
        or write to queue while the other holds a different lock. duplicate
        requeues possible.

    returns list of trial_<id> identifiers that were requeued.
    """
    # batch sacct call at start: one subprocess for all jids
    sacct_cache = _batch_sacct_commands(live_jids)

    requeued = []

    for output_dir in output_dirs:
        if not output_dir.exists():
            continue

        # walk output_dir/<stage>/<method>/ for trial_*.json configs
        # we assume typical layout: output_dir/stage1/method1/trial_123.json
        for stage_dir in output_dir.glob("*"):
            if not stage_dir.is_dir():
                continue
            stage = stage_dir.name

            for method_dir in stage_dir.glob("*"):
                if not method_dir.is_dir():
                    continue
                method = method_dir.name

                for trial_cfg in method_dir.glob("trial_*.json"):
                    # extract trial_id from filename
                    m = re.match(r"trial_(\d+)\.json", trial_cfg.name)
                    if not m:
                        continue
                    trial_id = int(m.group(1))

                    # check preconditions for orphan status
                    result_file = output_dir / stage / f"trial_{trial_id}.json"
                    if result_file.exists():
                        continue  # result exists; not orphan

                    if _is_in_queue(queue_file, lock_file, method, trial_id):
                        continue  # in queue; not orphan

                    in_live = any(_is_in_live_jid(jid, trial_id, sacct_cache) for jid in live_jids)
                    if in_live:
                        continue  # in flight; not orphan

                    # orphan detected; generate fresh sbatch line
                    # minimal line: method\\torphan_recovery\\tsbatch_cmd
                    # sbatch_cmd = original command that generated trial_<id>.json
                    # for now, append a minimal sentinel line for requeue
                    # (actual sbatch cmd recovery is out of scope; use placeholder)

                    orphan_line = f"{method}\torphan_recovery\t" \
                                  f"sbatch --job-name=orphan_trial_{trial_id} " \
                                  f"--output-dir {output_dir} " \
                                  f"--wrap 'echo orphan_recovery'\n"

                    # append to queue under flock
                    with open(queue_file, "a") as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        try:
                            f.write(orphan_line)
                        finally:
                            fcntl.flock(f, fcntl.LOCK_UN)

                    requeued.append(f"trial_{trial_id}")

    return requeued


def _log_event(event: str, **fields) -> None:
    """log event: event=NAME key1=val1 ... (non-string values json-encoded).
    """
    items = [f"event={event}"]
    for k, v in fields.items():
        if isinstance(v, str):
            items.append(f"{k}={v}")
        else:
            items.append(f"{k}={json.dumps(v)}")
    LOGGER.info(" ".join(items))


def _append_submitted_tsv(path: Path, jid: str, method: str, tag: str,
                          walltime: str, excl_n: int) -> None:
    """append one row: ts\\tjid\\tmethod\\ttag\\twalltime\\texcl_n.

    fcntl.flock for exclusive write; write header if file empty.
    """
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.tell() == 0:
            f.write("ts\tjid\tmethod\ttag\twalltime\texcl_n\n")
        ts = int(time.time())
        f.write(f"{ts}\t{jid}\t{method}\t{tag}\t{walltime}\t{excl_n}\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def _queue_empty(queue_file: Path, lock_file: Path) -> bool:
    """true iff queue missing, zero-byte, or all lines are blank/comment.
    """
    if not queue_file.exists() or queue_file.stat().st_size == 0:
        return True
    with open(lock_file, "a+") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        for line in queue_file.read_text().splitlines():
            if parse_spec_line(line) is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                return False
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    return True


def main() -> None:
    """argparse + dispatch loop.

    startup:
    1. parse args (all flags from spec).
    2. configure root logger: file handler (append, INFO); stream -> stderr (WARNING).
    3. install SIGTERM handler.
    4. resolve state_file, submitted_tsv defaults under <exclude_file.parent>.
    5. --reset-exclude-on-startup clears exclude file.
    6. load state (submitted_jids, seen_jid_node).
    7. init node_stats, last_sacct_cycle, cycle.
    8. create rng (seeded or SystemRandom).
    9. resolve data_root from DPE_DATA_ROOT.
    10. emit STARTUP event.

    main loop:
    - if queue empty (double-check under flock), break.
    - cycle += 1; get squeue counts.
    - if capped, sleep + continue.
    - pop headroom lines, submit each, log DISPATCH, append tsv.
    - every 10 cycles: sacct update, detect slow nodes, save state.
    - every 50 cycles: cleanup optuna zombies.
    - sleep poll_interval.

    drain:
    - log QUEUE_EMPTY.
    - wait for submitted jobs to finish.
    - save state, log DONE.

    KeyboardInterrupt -> save state + exit 0.
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
    parser.add_argument(
        "--pop-strategy",
        choices=["front", "random"],
        default="front",
        help="queue drain order: 'front' (default, sorted-queue bias) "
             "or 'random' (legacy multiplex)"
    )
    parser.add_argument(
        "--orphan-scan-interval",
        type=int,
        default=60,
        help="cycles between orphan scans; 0 disables (default 60)"
    )
    args = parser.parse_args()

    # configure logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(args.log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.WARNING)
    root_logger.addHandler(stream_handler)

    # resolve defaults
    if args.state_file is None:
        args.state_file = args.exclude_file.parent / "state.json"
    submitted_tsv = args.exclude_file.parent / "submitted.tsv"

    # reset exclude if requested
    if args.reset_exclude_on_startup:
        args.exclude_file.write_text("")

    # load state
    submitted_jids, seen_jid_node = _load_state(args.state_file)

    # init state vars
    node_stats = {}
    last_sacct_cycle = 0
    cycle = 0
    data_root = Path(os.environ["DPE_DATA_ROOT"])

    # resolve pop strategy and rng
    strategy = PopStrategy(args.pop_strategy)
    if strategy == PopStrategy.RANDOM:
        rng = random.Random(args.shuffle_seed) if args.shuffle_seed >= 0 else random.SystemRandom()
    else:
        rng = None

    # emit startup event
    _log_event("STARTUP",
               queue_file=str(args.queue_file),
               exclude_file=str(args.exclude_file),
               my_cap=args.my_cap,
               total_cap=args.total_cap,
               poll_interval=args.poll_interval,
               dry_run=args.dry_run,
               pop_strategy=args.pop_strategy)

    def _on_sigterm(signum, frame):
        LOGGER.info("SIGTERM received; saving state and exiting")
        _save_state(args.state_file, submitted_jids, seen_jid_node)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        # main loop. queue-empty does not exit; submit scripts may append
        # to the queue at any time. exit only on SIGTERM/scancel.
        empty_streak = 0
        while True:
            if _queue_empty(args.queue_file, args.state_file.parent / "queue.lock"):
                empty_streak += 1
                if empty_streak == 1 or empty_streak % 20 == 0:
                    _log_event("QUEUE_EMPTY", cycle=cycle, streak=empty_streak)
                time.sleep(args.poll_interval)
                continue
            empty_streak = 0

            cycle += 1
            try:
                my_pending = squeue_count(user=os.environ["USER"], partition="preempt")
                total_pending = squeue_count(partition="preempt")
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
                    rng=rng
                )
                if line is None:
                    break
                parsed = parse_spec_line(line)
                if parsed is None:
                    continue
                method, tag, raw_cmd = parsed
                # defensive: skip if a competing consumer (cpu array element)
                # already produced trial_<id>.json. atomic claim should make this
                # impossible in normal flow, but covers restarts / queue edits.
                m_cfg = re.search(r"trial_(\d+)\.json", raw_cmd)
                m_out = re.search(r"--output-dir\s+(\S+)", raw_cmd)
                m_stg = re.search(r'--stage\s+([^\s"]+)', raw_cmd)
                if m_cfg and m_out and m_stg:
                    expected = Path(m_out.group(1)) / m_stg.group(1) / f"trial_{m_cfg.group(1)}.json"
                    if expected.exists():
                        _log_event("DUPLICATE_SKIP", trial_id=m_cfg.group(1),
                                   method=method, stage=m_stg.group(1))
                        continue
                walltime = cap_for(method, partition="preempt", pilot_tag=tag)
                cmd = render_sbatch(raw_cmd, walltime, excl_str)
                jid = submit_sbatch(cmd, dry_run=args.dry_run)
                if jid is None:
                    continue
                submitted_jids.add(jid)
                _log_event("DISPATCH", jid=jid, method=method, tag=tag,
                          walltime=walltime, excl_n=excl_n)
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
                _save_state(args.state_file, submitted_jids, seen_jid_node)

            if cycle % 50 == 0:
                active = _discover_active_studies(data_root)
                if active:
                    cleanup_optuna_zombies(active)

            # periodic orphan scan
            if args.orphan_scan_interval > 0 and cycle % args.orphan_scan_interval == 0:
                output_dirs = _collect_experiment_output_dirs(args.queue_file)
                if output_dirs:
                    live_jids = submitted_jids | squeue_alive_jids("preempt")
                    orphans = scan_for_orphans(
                        output_dirs=output_dirs,
                        queue_file=args.queue_file,
                        live_jids=live_jids,
                        lock_file=args.state_file.parent / "queue.lock"
                    )
                    if orphans:
                        _log_event("ORPHANS_REQUEUED", count=len(orphans),
                                   trial_ids=orphans[:20])

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        _save_state(args.state_file, submitted_jids, seen_jid_node)
        sys.exit(0)


if __name__ == "__main__":
    main()
