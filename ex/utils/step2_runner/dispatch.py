"""step2_runner dispatch: emit a watchdog queue file from (experiment, winners.yaml).

invocation:

    DPE_DATA_ROOT=... DPE_CKPT_ROOT=... \
    python -m ex.utils.step2_runner.dispatch \
        --experiment model_selection \
        --winners scratch/200broad/winners_pinned/winners.model_selection.uniform_200broad.yaml \
        --max-cells-per-job 20 \
        [--methods CTSM,VFM,FMDRE]   # default: all methods in winners

emits one TAB-separated sbatch line per (method, cell_chunk). queue file
written to <DPE_DATA_ROOT>/step2_<exp>_queue.txt by default.

flow:
  1. load adapter.list_cells(config) -> [cell_idx, ...]
  2. for each method × chunk-of-N-cells: emit a queue line.
  3. skip cells whose result file already exists.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from ex.utils.step2_runner.load_winners import load_winners, list_methods

WORKDIR = os.environ.get("DPE_WORKDIR") or os.getcwd()
CONDA_ENV = (
    os.environ.get("DPE_CONDA_ENV")
    or os.environ.get("CONDA_DEFAULT_ENV")
    or "fac"
)
DEFAULT_PARTITION = "preempt"
DEFAULT_RESOURCES = "--gpus=1 --cpus-per-task=4 --mem=24G"


def _data_root() -> Path:
    root = os.environ.get("DPE_DATA_ROOT")
    if not root:
        raise SystemExit("DPE_DATA_ROOT must be set")
    return Path(root)


def chunk(lst: list, size: int) -> list[list]:
    """split list into chunks of size `size`."""
    return [lst[i:i+size] for i in range(0, len(lst), size)]


def queue_line(experiment: str, method: str, cell_chunk: list[int],
               winners: str, output_dir: Path, config_path: str | None,
               walltime: str, resources: str, method_label: str) -> str:
    """build a single TAB-separated queue line (METHOD\\tBUCKET\\tSBATCH...)."""
    cell_str = ",".join(str(c) for c in cell_chunk)
    log_dir = output_dir / method / "logs"
    job_name = f"step2_{experiment[:18]}_{method}_c{cell_chunk[0]}-{cell_chunk[-1]}"
    cfg_arg = f"--config {config_path}" if config_path else ""
    wrap = (
        f"set +u && source ~/.bashrc && conda activate {CONDA_ENV} && set -u && "
        f"export HDF5_USE_FILE_LOCKING=FALSE && cd {WORKDIR} && "
        "python -m ex.utils.step2_runner.worker "
        f"--experiment {experiment} --method {method} "
        f"--cell-indices '{cell_str}' "
        f"--winners {winners} --output-dir {output_dir} {cfg_arg}"
    )
    template = (
        f"sbatch --partition={DEFAULT_PARTITION} --time={walltime} {resources} --requeue "
        f"--job-name={job_name} --output={log_dir}/%j.out "
        f'--wrap="{wrap}"'
    )
    return f"{method_label}\tbroad\t{template}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--winners", required=True, help="path to winners.yaml")
    p.add_argument("--max-cells-per-job", type=int, default=20)
    p.add_argument("--methods", default=None,
                   help="comma-separated method whitelist (default: all in winners)")
    p.add_argument("--config", default=None,
                   help="experiment config.yaml (default: ex/<exp>/config.yaml)")
    p.add_argument("--out", default=None,
                   help="queue file path (default: $DPE_DATA_ROOT/step2_<exp>_queue.txt)")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="omit cells with existing result h5 (default: on)")
    args = p.parse_args()

    # adapter + config
    adapter = importlib.import_module(f"ex.{args.experiment}.step2_adapter")
    config_path = args.config or f"ex/{args.experiment}/config.yaml"
    config = adapter.load_config(config_path)

    # cells
    all_cells = list(adapter.list_cells(config))
    print(f"experiment {args.experiment}: {len(all_cells)} cells")

    # winners + methods
    winners = load_winners(args.winners)
    yaml_methods = list_methods(winners)
    if args.methods:
        wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
        missing = [m for m in wanted if m not in yaml_methods]
        if missing:
            raise SystemExit(f"requested methods not in winners.yaml: {missing}")
        methods = wanted
    else:
        methods = yaml_methods
    print(f"methods to dispatch: {methods}")

    # output paths
    data_root = _data_root()
    output_dir = data_root / args.experiment / "step2_results"
    queue_path = Path(args.out) if args.out else (data_root / f"step2_{args.experiment}_queue.txt")

    # build queue lines
    front, back = [], []
    n_skipped = 0
    n_emitted = 0
    for method in methods:
        # filter out cells with existing results
        cells_remaining = []
        for c in all_cells:
            result_path = output_dir / method / f"cell_{c}.h5"
            if args.skip_existing and result_path.exists():
                n_skipped += 1
                continue
            cells_remaining.append(c)
        if not cells_remaining:
            continue
        # walltime: ask adapter for per-cell walltime, scale by chunk size
        per_cell_seconds = adapter.walltime_per_cell_seconds(method, config)
        method_label = adapter.method_label(method)
        resources = adapter.resources_for_method(method)
        chunks = chunk(cells_remaining, args.max_cells_per_job)
        for chunk_cells in chunks:
            chunk_seconds = max(60, int(per_cell_seconds * len(chunk_cells) * 1.5) + 90)  # 50% headroom + 90s startup
            walltime = _seconds_to_hms(chunk_seconds)
            (output_dir / method / "logs").mkdir(parents=True, exist_ok=True)
            line = queue_line(
                args.experiment, method, chunk_cells,
                args.winners, output_dir, args.config,
                walltime, resources, method_label,
            )
            # adapter decides front/back priority (slow → front, fast cpu-eligible → back)
            if adapter.is_cpu_eligible(method):
                back.append(line)
            else:
                front.append(line)
            n_emitted += 1

    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text("\n".join(front + back) + ("\n" if (front or back) else ""))
    msg = f"[{args.experiment}] {n_emitted} queue lines"
    if args.skip_existing:
        msg += f" (skipped {n_skipped} already-done cells)"
    msg += f" -> {queue_path}"
    print(msg)


def _seconds_to_hms(s: int) -> str:
    s = max(60, int(s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}"


if __name__ == "__main__":
    main()
