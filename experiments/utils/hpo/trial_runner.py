"""canonical adapter-driven hpo trial runner.

cli: --experiment, --method, --config-file, --eval-cells-file, --output-dir, --stage.
loads adapter, reads trial config + eval cells, defines eval_cell closure,
calls run_trial unchanged, annotates result, and writes atomically to
output_dir/<stage>/trial_<id>.json.
"""

import argparse
import json
import math
import os
import signal
from pathlib import Path

import numpy as np
import torch

from experiments.utils.hpo.trial import run_trial
from experiments.utils.hpo.registry import build_search_spaces
from experiments.utils.hpo.cell_schema import coerce_cells_from_json
from experiments.utils.hpo.adapters import get_adapter


def parse_args(args=None):
    """parse cli arguments.

    returns Namespace: experiment, method, config_file, eval_cells_file, output_dir, stage.
    """
    parser = argparse.ArgumentParser(
        description="canonical hpo trial runner: adapter-driven worker"
    )
    parser.add_argument("--experiment", required=True, help="adapter name")
    parser.add_argument("--method", required=True, help="estimation method")
    parser.add_argument("--config-file", required=True, dest="config_file",
                        help="path to trial.json")
    parser.add_argument("--eval-cells-file", required=True, dest="eval_cells_file",
                        help="path to cells.json")
    parser.add_argument("--output-dir", required=True, dest="output_dir",
                        help="output directory (stage subdir created inside)")
    parser.add_argument("--stage", default="broad",
                        choices=["broad", "refined", "holdout"],
                        help="pilot variant tag for annotation")
    return parser.parse_args(args)


def trimmed_mean(values, trim_frac=0.2):
    """robust mean: filter finite, trim fraction from each tail, return float.

    filter to finite values only. if empty, return inf.
    sort; k = round(trim_frac * n).
    core = sorted[k:n-k] (or full list if n <= 2*k).
    return float(mean(core)).
    """
    finite_vals = [v for v in values if math.isfinite(float(v))]
    if not finite_vals:
        return float("inf")
    sorted_vals = sorted(finite_vals)
    n = len(sorted_vals)
    k = round(trim_frac * n)
    core = sorted_vals if n <= 2 * k else sorted_vals[k: n - k]
    return float(np.mean(core))


def main():
    """entry point: sigterm -> parse -> adapter -> config -> cells -> eval_cell -> run_trial -> annotate -> write."""
    # step 1: restore default sigterm so watchdog kill propagates cleanly
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    # step 2: parse args
    args = parse_args()
    method = args.method

    # step 3: instantiate adapter (KeyError or AttributeError propagates on bad name)
    adapter = get_adapter(args.experiment)

    # step 4: load and validate trial config
    with open(args.config_file) as f:
        trial_config = json.load(f)

    trial_id = trial_config["trial_id"]
    assert trial_config["method"] == method, (
        f"method mismatch: config={trial_config['method']!r} vs arg={method!r}"
    )
    hyperparams = trial_config["hyperparams"]

    # step 5: load and coerce cells (B1 fix)
    with open(args.eval_cells_file) as f:
        cells_raw = json.load(f)
    cells = coerce_cells_from_json(cells_raw["cells"])
    eval_sample_seed = cells_raw.get("seed")

    if not cells:
        raise ValueError("empty cells list in eval_cells_file")

    # step 6: resolve builder + requires_pstar (H3 fix: explicit KeyError)
    search_spaces = build_search_spaces()
    if method not in search_spaces:
        raise KeyError(
            f"unknown method {method!r}; known: {sorted(search_spaces.keys())}"
        )
    entry = search_spaces[method]
    builder = entry["builder"]
    requires_pstar = entry["requires_pstar"]

    # step 7: eval_cell closure captures adapter, builder, requires_pstar, hyperparams
    def eval_cell(cell):
        """load cell data, build estimator, fit, predict, return mae scalar.

        raises FileNotFoundError if data missing (caught and logged by run_trial).
        returns float; non-finite signals skip.
        """
        data = adapter.load_cell_data(cell, device=adapter.device())
        est = builder(
            input_dim=adapter.latent_dim(),
            device=adapter.device(),
            num_waypoints=adapter.num_waypoints(),
            **hyperparams,
        )
        if requires_pstar:
            est.fit(data["p0"], data["p1"], data["pstar"])
        else:
            est.fit(data["p0"], data["p1"])
        predicted = est.predict_ldr(data["pstar"])  # (n,)
        mae = torch.abs(predicted.cpu() - data["true_ldrs"].cpu()).mean()  # scalar
        return float(mae)

    # step 8: compute output path with stage prefix
    output_path = Path(args.output_dir) / args.stage / f"trial_{trial_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # step 9: call run_trial unchanged
    result = run_trial(
        experiment=adapter.cell_seed_namespace(),
        method=method,
        trial_id=trial_id,
        hyperparams=hyperparams,
        eval_cells=cells,
        eval_cell=eval_cell,
        output_dir=str(output_path.parent),
        metric_key=adapter.metric_key(),
    )

    # step 10: annotate result with score + metadata; copy _meta (B5 fix)
    metric_values = result[adapter.metric_key()].values()
    score = trimmed_mean(metric_values, trim_frac=0.2)

    result["score"] = score
    result["pilot_variant"] = args.stage
    result["eval_sample_seed"] = eval_sample_seed
    result["training_cells"] = [list(cell) for cell in cells]  # tuples -> lists for json
    result["workflow_version"] = "v1"

    if "_meta" in trial_config:
        result["_meta"] = trial_config["_meta"]

    # atomic re-write to stage-prefixed path (tmp + fsync + replace)
    tmp_path = output_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)

    # step 11: log summary
    mean_metric = result["mean_metric"]
    elapsed = result["elapsed_seconds"]
    print(
        f"method={method}, trial_id={trial_id}, stage={args.stage}, "
        f"score={score:.6f}, elapsed={elapsed:.2f}s, path={output_path}"
    )


if __name__ == "__main__":
    main()
