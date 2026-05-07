# step2_runner

distributed step2 runner: dispatches a winners-yaml-driven evaluation across
slurm jobs, with configurable per-job cell budget. mirrors the variant_sweep
architecture (build_queues -> run_variant_sweep -> gather), and supports two
drain modes:

- **single-drain**: lite watchdog only, drains everything to preempt GPU.
- **dual-drain**: lite watchdog (front-pop -> preempt) + cpu array dispatcher
  (back-pop -> array CPU) feeding from opposite ends of the same queue. fast
  cpu-eligible methods (`adapter.is_cpu_eligible`) sit at the queue back; slow
  GPU-only methods at the front. atomic flock on the shared queue prevents
  overlap.

## components

```
experiments/utils/step2_runner/
  load_winners.py        schema-aware winners.yaml loader
  dispatch.py            emits sbatch queue file: chunks cells into N per job
  worker.py              per-job worker: fits and evals (method, cell_chunk)
  gather.py              assembles per-cell h5 fragments into results.h5

  # dual-drain (optional)
  cpu_array_element.py   per-array-element back-popper; runs worker subprocess
  cpu_dispatcher.py      submits cpu slurm array job
  submit_dual.sh         wrapper: lite watchdog + cpu_dispatcher together
```

## per-experiment adapter (required)

each experiment provides `experiments/<exp>/step2_adapter.py` exposing:

```
load_config(path)            -> dict
list_cells(config)           -> Iterable[int]    # all cell indices (flat ints)
bucket_for_cell(idx, config) -> str | None       # for winners.yaml lookup; None = use default
fit_and_eval(method, hp, cell_idx, config, device) -> dict
    must return {'est_ldrs': np.ndarray, ...}; optional 'mae_per_test_set', 'true_ldrs'
walltime_per_cell_seconds(method, config) -> int
resources_for_method(method) -> str    # sbatch flags
is_cpu_eligible(method) -> bool        # routes to back of queue (CPU array drain)
method_label(method) -> str            # for cap_for matching in watchdog
```

optional hooks consumed by gather.py:
```
gather_dataset_name(method, config) -> str   # default: 'est_ldrs_arr_<method>'
gather_output_path(config) -> str            # default: experiments/<exp>/raw_results/results.h5
```

reference adapters:
- `experiments/model_selection/step2_adapter.py`     — Pattern A, 70 cells, kl_idx bucket
- `experiments/elbo_estimation/step2_adapter.py`     — Pattern C, 24000 cells, no bucket, scalar output
- `experiments/smodice_eldr_estimation/step2_adapter.py` — Pattern C, (k1,k2,seed) tuple cells flattened to int

## winners.yaml schema

```yaml
methods:
  CTSM:
    hyperparams:                  # default broadcast to all buckets
      n_epochs: 1295
      lr: 1.79e-3
      ...
    score: { ... }                # provenance (optional)
    per_bucket:                   # optional override per bucket id
      kl_idx_3: { hyperparams: {lr: 5e-4, ...} }
provenance: { ... }
```

`load_winners.resolve_hp(winners, method, bucket_id)` resolution priority:
1. `methods[method].per_bucket[bucket_id].hyperparams` (if present)
2. `methods[method].hyperparams` (default)

legacy schema `<method>: <bucket_idx>: [{hyperparams, ...}, ...]` (top-K list) is
also supported via auto-detection.

## usage — single-drain (preempt only)

```bash
export DPE_DATA_ROOT=/data/user_data/$USER/dpe-submission
export DPE_CKPT_ROOT=/scratch/$USER/ckpt/dpe-submission

# 1. dispatch: emit queue (one line per (method, cell_chunk))
python -m experiments.utils.step2_runner.dispatch \
    --experiment model_selection \
    --winners scratch/200broad/winners_pinned/winners.model_selection.uniform_200broad.yaml \
    --max-cells-per-job 20 \
    [--methods CTSM,VFM,FMDRE]   # optional whitelist; default = all in yaml

# emits $DPE_DATA_ROOT/step2_<exp>_queue.txt

# 2. drain via watchdog_lite (existing infrastructure)
bash experiments/utils/submit_watchdog_lite.sh \
    $DPE_DATA_ROOT/step2_model_selection_queue.txt \
    22 800 60   # my-cap=22, total-cap=800, orphan-scan=60s

# 3. when results land, assemble per-cell h5 fragments into results.h5
python -m experiments.utils.step2_runner.gather \
    --experiment model_selection
```

## usage — dual-drain (preempt GPU + array CPU, both feeding shared queue)

cpu-eligible methods (`adapter.is_cpu_eligible(method) == True`) drain through
the array partition while gpu-only methods drain through preempt. throughput
roughly doubles vs single-drain when both partitions have headroom.

```bash
# dispatch the same way (queue is sorted: gpu-only at front, cpu-eligible at back)
python -m experiments.utils.step2_runner.dispatch \
    --experiment model_selection \
    --winners scratch/200broad/winners_pinned/winners.model_selection.uniform_200broad.yaml \
    --max-cells-per-job 20

# submit BOTH drains via the wrapper
bash experiments/utils/step2_runner/submit_dual.sh \
    $DPE_DATA_ROOT/step2_model_selection_queue.txt \
    22 800              # watchdog: my-cap=22, total-cap=800
    64 100 2            # cpu_array: array_size=64, concurrency=100, n_per_element=2
    BDRE,MDRE_15,CTSM   # method_filter: only these methods drain via array partition

# OR submit them separately for finer control:
bash experiments/utils/submit_watchdog_lite.sh $DPE_DATA_ROOT/step2_<exp>_queue.txt 22 800 60
python -m experiments.utils.step2_runner.cpu_dispatcher \
    --queue-file $DPE_DATA_ROOT/step2_<exp>_queue.txt \
    --array-size 64 --concurrency 100 --n-per-element 2 \
    --output-root $DPE_DATA_ROOT/<exp>/step2_dual/$(date +%Y%m%d)/cpu_array \
    --method-filter BDRE,MDRE_15,CTSM \
    --walltime 2:00:00 \
    --device cpu

# gather as usual when results land (idempotent; safe to run incrementally)
python -m experiments.utils.step2_runner.gather --experiment model_selection
```

### picking dual-drain parameters

- `array-size` ≈ ceil(num_cpu_eligible_lines / n_per_element). e.g. for 16
  cpu-eligible queue lines × 2 lines/element = 8 elements is enough; oversize
  is fine since elements that find no work just exit.
- `concurrency` ≤ array_qos `MaxJobsPU` (typically 100 on this cluster).
- `n_per_element=2` is a reasonable default; bigger amortizes startup cost
  but increases orphan-loss risk on preempt.
- `method-filter` should be the cpu-eligible subset for your experiment
  (`MDRE_15,BDRE,CTSM,TSM,...`). without it, the array elements would try to
  run gpu-only methods on cpu and crash. dispatch puts them at the back of
  the queue but doesn't enforce a hard partition.

## idempotency

- worker skips cells whose `cell_<idx>.h5` already exists; partial chunks resume cleanly.
- dispatch with `--skip-existing` (default on) omits done cells from the queue.
- gather is fully reproducible from per-cell fragments.

## per-cell output

each cell's result lands at `<DPE_DATA_ROOT>/<exp>/step2_results/<method>/cell_<idx>.h5`:

```
datasets:
  est_ldrs:           (ntest_sets, nsamples_test) float32
  mae_per_test_set:   (ntest_sets,) float32
  true_ldrs:          (ntest_sets, nsamples_test) float32   [optional, for diagnostics]
attrs:
  experiment, method, cell_idx, bucket_id, hyperparams (json string),
  elapsed_seconds, ok (bool), error (traceback string if !ok)
```

failed cells (`ok=False`) are sentineled with a single-NaN `est_ldrs` so the
worker doesn't loop on them. gather treats them as missing.

## adding a new experiment

1. write `experiments/<exp>/step2_adapter.py` implementing the contract above.
2. create a `winners.<exp>.<tag>.yaml` (any path; passed via `--winners`).
3. run the dispatch + watchdog + gather flow as above.

experiment-specific quirks (e.g., eig's TriangularTSMEIGAdapter wrappers,
smodice's SUPPORTED_ENCODINGS dispatch) belong in the per-experiment adapter,
not the shared library.
