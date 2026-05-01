"""matrix orchestration and watchdog ownership for multi-experiment hpo.

orchestration jobs (broad/refined/holdout stages) run on cpu via sbatch and are
chained with afterok deps between workflow jobs. trial jobs are dispatched
asynchronously by the watchdog to the preempt partition; stage-completion waits
(barriers) are handled inside each workflow stage by polling output dirs (B12).

wave-stagger paces submission burst only; all waves may pend simultaneously in
slurm. per-user cpu cap is enforced by the slurm scheduler, not this launcher (H2).

skipped.json resolves to $DPE_CKPT_ROOT/watchdog/<run_id>/skipped.json (M2).
"""

import argparse
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import experiments.utils.hpo.adapters as adapters
from experiments.utils.hpo.registry import LEGACY_ALIASES
from experiments.utils.hpo.method_specs import METHOD_SPECS

logger = logging.getLogger(__name__)

_WORKDIR = "/home/aviamala/dpe-submission"

# --- step 1 ---

def resolve_experiments(arg: str) -> list[str]:
    """parse experiment CSV or "all" from adapter registry.

    arg="all" -> adapters.list_adapters(); else split CSV and validate each
    name exists in list_adapters(). raises ValueError on unknown name.
    """
    known = set(adapters.list_adapters())
    if arg == "all":
        return sorted(known)
    names = [s.strip() for s in arg.split(",") if s.strip()]
    for n in names:
        if n not in known:
            raise ValueError(f"unknown experiment: {n!r}; known: {sorted(known)}")
    return names


# --- step 2 ---

def resolve_methods(arg: str, tabular_support: dict[str, bool]) -> list[str]:
    """parse method CSV or "all" from registry; filter tabular_only vs adapter support.

    arg="all" -> all canonical keys from build_search_spaces (no alias duplicates).
    else split CSV, validate each name exists in METHOD_SPECS or aliases.
    tabular_support: maps method -> bool; if method is tabular_only=True and
    tabular_support[method]=False, skip with warning. if tabular_support is empty
    dict, only aliase-resolution and validation occur.
    raises ValueError on unknown method name.
    """
    # resolve alias -> canonical for validation
    alias_set = set(LEGACY_ALIASES.keys())
    canonical_set = {k for k in METHOD_SPECS if k not in alias_set}

    if arg == "all":
        candidates = list(canonical_set)
    else:
        raw = [s.strip() for s in arg.split(",") if s.strip()]
        for n in raw:
            # accept both canonical and alias names at input
            if n not in METHOD_SPECS and n not in LEGACY_ALIASES:
                raise ValueError(f"unknown method: {n!r}")
        # resolve aliases to canonical
        candidates = [LEGACY_ALIASES.get(n, n) for n in raw]

    # filter tabular_only if tabular_support provided
    result = []
    for method in candidates:
        spec = METHOD_SPECS.get(method)
        if spec is None:
            continue
        if spec.get("tabular_only", False):
            # only include if caller's tabular_support allows it
            if not tabular_support.get(method, False):
                logger.warning("skipping tabular_only method %s (not supported by experiment)", method)
                continue
        result.append(method)
    return result


# --- step 3 ---

def load_adapters(
    experiments: list[str],
) -> tuple[dict[str, object], dict[str, str]]:
    """instantiate adapters and filter by is_ready().

    returns (ready_adapters, skipped) where skipped maps exp_name -> reason str.
    experiments with is_ready()=False are logged and excluded from ready_adapters.
    """
    ready: dict[str, object] = {}
    skipped: dict[str, str] = {}
    for exp in experiments:
        try:
            adapter = adapters.get_adapter(exp)
        except (KeyError, Exception) as e:
            reason = f"get_adapter failed: {e}"
            logger.warning("skipping %s: %s", exp, reason)
            skipped[exp] = reason
            continue
        if not adapter.is_ready():
            reason = "adapter.is_ready() returned False"
            logger.warning("skipping %s: %s", exp, reason)
            skipped[exp] = reason
            continue
        ready[exp] = adapter
    return ready, skipped


# --- step 4 ---

def build_matrix(
    methods: list[str],
    adapters_map: dict[str, object],
) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    """iterate (method, exp) pairs; filter tabular_only mismatches.

    returns (valid_pairs, skipped) where skipped maps exp -> [methods skipped].
    tabular_only=True + supports_tabular()=False -> skip and record.
    """
    valid: list[tuple[str, str]] = []
    skipped: dict[str, list[str]] = {}
    for exp, adapter in adapters_map.items():
        for method in methods:
            spec = METHOD_SPECS.get(method, {})
            if spec.get("tabular_only", False) and not adapter.supports_tabular():
                skipped.setdefault(exp, []).append(method)
                logger.warning("skipping (%s, %s): tabular_only but exp lacks tabular support",
                               method, exp)
                continue
            valid.append((method, exp))
    return valid, skipped


# --- step 5 ---

def validate_queue_file(queue_file: Path, force: bool) -> None:
    """gate on non-empty queue_file unless force=True.

    creates parent dir. if file exists and is non-empty and force=False, raises
    ValueError with instruction. else touch file to ensure it exists.
    """
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    if queue_file.exists() and queue_file.stat().st_size > 0 and not force:
        raise ValueError(
            f"queue_file {queue_file} is non-empty. inspect it, then pass --force to overwrite "
            "or append, or `rm` the file and re-run."
        )
    queue_file.touch()


# --- step 6 ---

def init_watchdog_logdir(run_id: str) -> Path:
    """create $DPE_CKPT_ROOT/watchdog/<run_id> and stub files.

    stubs: exclude.txt, state.json, submitted.tsv.
    returns the logdir path.
    """
    ckpt_root = Path(os.environ.get("DPE_CKPT_ROOT", "/scratch/dpe-submission"))
    logdir = ckpt_root / "watchdog" / run_id
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "exclude.txt").touch()
    (logdir / "state.json").touch()
    (logdir / "submitted.tsv").touch()
    return logdir


def record_skipped(skipped: dict[str, list[str]], run_id: str) -> None:
    """write skipped pairs to $DPE_CKPT_ROOT/watchdog/<run_id>/skipped.json (M2).

    creates parent dir if needed. writes JSON atomically.
    """
    ckpt_root = Path(os.environ.get("DPE_CKPT_ROOT", "/scratch/dpe-submission"))
    out = ckpt_root / "watchdog" / run_id / "skipped.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(skipped, indent=2))
    tmp.replace(out)


# --- step 7 ---

def submit_watchdog(
    queue_file: Path,
    logdir: Path,
    my_cap: int,
    total_cap: int,
) -> str:
    """sbatch watchdog via submit_watchdog.sh; return jid string.

    tries submit_watchdog.sh first; falls back to submit_watchdog_v735e.sh.
    invokes script with positional args: <queue_file> <my_cap> <total_cap>
    plus --reset-exclude-on-startup flag. uses --parsable; raises ValueError
    if jid cannot be parsed or rc != 0.
    """
    script = _resolve_watchdog_script()
    # script handles its own sbatch submission internally; launcher calls it
    # directly (not via sbatch --wrap) so --parsable output comes from inside
    # the script. script echoes jid on stdout.
    cmd = [
        "bash",
        str(script),
        "--reset",           # reset exclude list on startup
        str(queue_file),
        str(my_cap),
        str(total_cap),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(
            f"sbatch watchdog failed (rc={result.returncode}): "
            f"cmd={cmd!r} stderr={result.stderr[:400]}"
        )
    # parse jid from script output: "jobid:          <JID>" line
    for line in result.stdout.splitlines():
        if line.startswith("jobid:"):
            jid = line.split(":", 1)[1].strip()
            if jid:
                return jid
    raise ValueError(
        f"could not parse jid from watchdog script output: {result.stdout[:400]}"
    )


def _resolve_watchdog_script() -> Path:
    """locate submit_watchdog.sh; fallback to submit_watchdog_v735e.sh."""
    base = Path(_WORKDIR) / "experiments" / "utils"
    primary = base / "submit_watchdog.sh"
    fallback = base / "submit_watchdog_v735e.sh"
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"watchdog submit script not found at {primary} or {fallback}"
    )


# --- step 8 ---

def submit_workflow(
    method: str,
    exp: str,
    queue_file: Path,
    watchdog_jid: str,
    budget: int = 250,
) -> str:
    """sbatch one workflow job; return jid.

    uses `--dependency=after:<watchdog_jid>` (NOT afterok). watchdog is a
    long-running daemon; afterok would deadlock since watchdog only completes
    when its queue drains, and the queue is empty until workflows append to it.
    `after:` fires when the watchdog has STARTED (running or completed), which
    is what we want.

    invokes `python -m experiments.utils.hpo.workflow --method M --experiment E
    --stage all --queue-file Q --budget B` via --wrap. raises ValueError on
    sbatch failure.
    """
    wrap_cmd = (
        f"source ~/.bashrc && conda activate fac && cd {_WORKDIR} && "
        f"python -m experiments.utils.hpo.workflow "
        f"--method {method} --experiment {exp} "
        f"--stage all --queue-file '{queue_file}' --budget {budget}"
    )
    cmd = [
        "sbatch",
        "--parsable",
        "--partition=cpu",
        "--time=4:00:00",
        "--mem=2G",
        "--cpus-per-task=1",
        f"--job-name=hpo_{method}_{exp}",
        f"--dependency=after:{watchdog_jid}",
        f"--wrap={wrap_cmd}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError(
            f"sbatch workflow failed (rc={result.returncode}): "
            f"method={method} exp={exp} stderr={result.stderr[:400]}"
        )
    return result.stdout.strip().splitlines()[0]


def submit_waves(
    pairs: list[tuple[str, str]],
    queue_file: Path,
    watchdog_jid: str,
    wave_size: int = 3,
    budget: int = 250,
) -> list[list[str]]:
    """partition pairs into waves of wave_size; submit each wave; return nested jid list.

    wave-stagger paces submission burst only (H2). all waves may pend simultaneously.
    each wave is submitted in full before moving to next.
    """
    waves_jids: list[list[str]] = []
    for i in range(0, len(pairs), wave_size):
        wave = pairs[i : i + wave_size]
        jids = [
            submit_workflow(method, exp, queue_file, watchdog_jid, budget)
            for method, exp in wave
        ]
        waves_jids.append(jids)
    return waves_jids


# --- step 9 ---

def emit_summary(
    run_id: str,
    watchdog_jid: str,
    waves_jids: list[list[str]],
    queue_file: Path,
    logdir: Path,
    skipped: dict,
    timestamp: str,
) -> None:
    """print summary + write launcher_manifest.json to logdir.

    manifest fields: run_id, watchdog_jid, workflow_jids (flat list),
    wave_jids (nested), skipped_pairs, timestamp.
    """
    all_jids = [jid for wave in waves_jids for jid in wave]
    manifest = {
        "run_id": run_id,
        "watchdog_jid": watchdog_jid,
        "workflow_jids": all_jids,
        "wave_jids": waves_jids,
        "skipped_pairs": skipped,
        "timestamp": timestamp,
    }
    manifest_path = logdir / "launcher_manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(manifest_path)

    print(f"=== hpo launcher summary ===")
    print(f"run_id:      {run_id}")
    print(f"watchdog:    {watchdog_jid}")
    print(f"workflows:   {len(all_jids)} jobs across {len(waves_jids)} waves")
    print(f"queue_file:  {queue_file}")
    print(f"logdir:      {logdir}")
    print(f"monitor:     squeue -j {watchdog_jid}")
    print(f"tail log:    tail -f {logdir}/watchdog.log")
    print(f"manifest:    {manifest_path}")
    if skipped:
        print(f"skipped:     {json.dumps(skipped, indent=2)}")


# --- CLI ---

def parse_args() -> argparse.Namespace:
    """CLI arg parser for launcher.

    flags: --experiments, --methods, --queue-file, --my-cap, --total-cap,
    --wave-size, --budget, --force, --dry-run.
    """
    p = argparse.ArgumentParser(description="hpo multi-experiment launcher")
    p.add_argument("--experiments", default="all",
                   help='CSV list of experiment names, or "all"')
    p.add_argument("--methods", default="all",
                   help='CSV list of method names, or "all"')
    p.add_argument("--queue-file", type=Path, default=None,
                   help="watchdog queue file; default $DPE_DATA_ROOT/multi_experiment_watchdog_queue.txt")
    p.add_argument("--my-cap", type=int, default=80,
                   help="per-user preempt job cap passed to watchdog")
    p.add_argument("--total-cap", type=int, default=200,
                   help="total preempt job cap passed to watchdog")
    p.add_argument("--wave-size", type=int, default=3,
                   help="workflow submissions per wave (pacing burst, not concurrency cap)")
    p.add_argument("--budget", type=int, default=250,
                   help="trial budget per (method, exp) workflow")
    p.add_argument("--force", action="store_true",
                   help="overwrite non-empty queue_file without error")
    p.add_argument("--dry-run", action="store_true",
                   help="print sbatch commands without submitting")
    return p.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    """orchestrate watchdog + workflow submissions for (method, exp) matrix.

    steps 1-9 from spec:
    1. resolve experiments
    2. resolve methods
    3. load adapters, filter is_ready()
    4. build (method, exp) matrix, filter tabular mismatches
    5. validate queue_file
    6. init watchdog logdir, record_skipped
    7. submit watchdog (singleton)
    8. wave-stagger workflow submissions
    9. emit summary
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args is None:
        args = parse_args()

    # 1. resolve experiments
    exps = resolve_experiments(args.experiments)

    # 2. resolve methods (tabular filtering done per-pair in step 4)
    methods = resolve_methods(args.methods, {})

    # 3. load adapters, filter is_ready
    adapter_map, adapter_skipped = load_adapters(exps)

    # 4. build (method, exp) matrix; filter tabular mismatches
    valid_pairs, pair_skipped = build_matrix(methods, adapter_map)

    # merge skip records: adapter-level + pair-level
    all_skipped: dict = {**{k: [v] for k, v in adapter_skipped.items()}, **pair_skipped}

    # 5. validate queue_file
    if args.queue_file is None:
        data_root = os.environ.get("DPE_DATA_ROOT", ".")
        args.queue_file = Path(data_root) / "multi_experiment_watchdog_queue.txt"
    validate_queue_file(args.queue_file, force=args.force)

    # 6. init logdir + record skipped
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = timestamp
    logdir = init_watchdog_logdir(run_id)
    record_skipped(all_skipped, run_id)

    if args.dry_run:
        print(f"[dry-run] run_id={run_id} valid_pairs={len(valid_pairs)}")
        print(f"[dry-run] would submit watchdog + {len(valid_pairs)} workflows in waves of {args.wave_size}")
        return

    # 7. submit watchdog
    watchdog_jid = submit_watchdog(
        args.queue_file, logdir,
        my_cap=args.my_cap, total_cap=args.total_cap,
    )
    logger.info("watchdog submitted: jid=%s", watchdog_jid)

    # 8. wave-stagger workflow submissions
    waves_jids = submit_waves(
        valid_pairs, args.queue_file, watchdog_jid,
        wave_size=args.wave_size, budget=args.budget,
    )

    # 9. emit summary
    emit_summary(run_id, watchdog_jid, waves_jids, args.queue_file,
                 logdir, all_skipped, timestamp)


if __name__ == "__main__":
    main()
