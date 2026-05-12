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
ex/utils/step2_runner/
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

each experiment provides `ex/<exp>/step2_adapter.py` exposing:

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
gather_output_path(config) -> str            # default: ex/<exp>/raw_results/results.h5
```

## canonical winners path

all winners yamls live at `scratch/gold_winners/winners.<exp>.yaml`. see
`scratch/gold_winners/README.md` for the gold-set selection rules and per-method
score schema.

most experiments share the same short name as the directory (`model_selection`,
`elbo`, etc.) but a few experiment dirs have an `_eldr` suffix that
the winners filename omits — see the `winners.yaml` column below.

## per-experiment status (11 experiments, all ported)

| experiment dir | cell axis | n_cells | bucket axis | output ds name | gather output | winners.yaml |
|---|---|---|---|---|---|---|
| model_selection | row idx | 70 | `kl_idx_<n>` | `est_ldrs_arr_<m>` | `results.h5` | `winners.model_selection.yaml` |
| elbo | design row | 24000 | none | `est_eigs_arr_<m>` | `results_d=D,nsamples=N.h5` | `winners.elbo.yaml` |
| eig | design row | 6000 | none | `est_eigs_arr_<m>` | `results_d=D,nsamples=N.h5` | `winners.eig.yaml` (+ `true_eigs_arr` post-step via `gather_postprocess`) |
| smodice_eldr_estimation | flat (k1,k2,seed) | 480 | `k1_idx_<n>` | `est_ldrs_<m>` | `<encoding>/<sigma>/results_all_cells.h5` | `winners.smodice_eldr_estimation.yaml` |
| pendulum_eldr_estimation | flat (k1,k2,seed) | 160 | `k1_idx_<n>` | `est_ldrs_<m>` | `results_all_cells.h5` | `winners.pendulum_eldr_estimation.yaml` |
| mnist_eldr_estimation | flat (alpha,pair) | 160 | `alpha_idx_<n>` | `est_ldrs_<m>` | `results_all_cells.h5` | `winners.mnist_eldr_estimation.yaml` |
| **mnist_eldr_cond_flow** | flat (alpha,pair) | 160 | `alpha_idx_<n>` | `est_ldrs_<m>` | `results_all_cells.h5` | `winners.mnist_cond_flow.yaml` (no `_eldr` in filename) |
| **dbpedia_eldr_cond_flow** | flat (alpha,pair) | 160 | `alpha_idx_<n>` | `est_ldrs_<m>` | `results_all_cells.h5` | `winners.dbpedia_cond_flow.yaml` (no `_eldr` in filename) |
| plugin_dre | row idx | 4 | `kl_idx_<n>` | `est_ldrs_grid_<m>` | `raw_results.h5` | `winners.plugin_dre.yaml` |
| dre_sample_complexity | flat (row,ntrain_idx) | 420 | `kl_idx_<n>` | `est_ldrs_arr_<m>` | `results.h5` | `winners.dre_sample_complexity.yaml` |
| pstar_sample_complexity | flat (instance,pstar_idx) | 80 | `pstar_idx_<n>` | `est_ldrs_arr_<m>` | `results_all_cells.h5` | `winners.pstar_sample_complexity.yaml` (triangular methods only) |

quirks worth knowing for each:

- **smodice / mnist_eldr_cond_flow / dbpedia_eldr_cond_flow / pstar_sample_complexity**: the original step2 wrote one h5 per cell with multiple method datasets inside; this runner emits a single combined h5 instead. step3/4 may need a small slicing update for these.
- **pstar_sample_complexity**: no v2 winners yaml exists (no non-triangular HPO data in 200broad); pass a custom `--winners` if you have one, or add per-bucket overrides for triangular methods to a hand-written yaml.
- **plugin_dre / dre_sample_complexity**: builders take an extra `config` kwarg (`build_X(input_dim, device, config, **hp)`) — passed through automatically by these adapters.
- **eig**: writes a `true_eigs_arr` post-step via `adapter.gather_postprocess(config, out_path)` — a deterministic function of dataset (`Sigma_pi`, `design`). gather.py invokes it automatically if defined.
- **model_selection**: `ex/synth/model_selection/hpo_search_spaces.py` is broken (stale TDRE_5 reference); this adapter bypasses it and uses METHOD_SPECS directly. all other experiments use their own SEARCH_SPACES module which imports cleanly.

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
python -m ex.utils.step2_runner.dispatch \
    --experiment model_selection \
    --winners scratch/gold_winners/winners.model_selection.yaml \
    --max-cells-per-job 20 \
    [--methods CTSM,VFM,FMDRE]   # optional whitelist; default = all in yaml

# emits $DPE_DATA_ROOT/step2_<exp>_queue.txt

# 2. drain via watchdog_lite (existing infrastructure)
bash ex/utils/submit_watchdog_lite.sh \
    $DPE_DATA_ROOT/step2_model_selection_queue.txt \
    22 800 60   # my-cap=22, total-cap=800, orphan-scan=60s

# 3. when results land, assemble per-cell h5 fragments into results.h5
python -m ex.utils.step2_runner.gather \
    --experiment model_selection
```

## usage — dual-drain (preempt GPU + array CPU, both feeding shared queue)

cpu-eligible methods (`adapter.is_cpu_eligible(method) == True`) drain through
the array partition while gpu-only methods drain through preempt. throughput
roughly doubles vs single-drain when both partitions have headroom.

```bash
# dispatch the same way (queue is sorted: gpu-only at front, cpu-eligible at back)
python -m ex.utils.step2_runner.dispatch \
    --experiment model_selection \
    --winners scratch/gold_winners/winners.model_selection.yaml \
    --max-cells-per-job 20

# submit BOTH drains via the wrapper
bash ex/utils/step2_runner/submit_dual.sh \
    $DPE_DATA_ROOT/step2_model_selection_queue.txt \
    22 800              # watchdog: my-cap=22, total-cap=800
    64 100 2            # cpu_array: array_size=64, concurrency=100, n_per_element=2
    BDRE,MDRE_15,CTSM   # method_filter: only these methods drain via array partition

# OR submit them separately for finer control:
bash ex/utils/submit_watchdog_lite.sh $DPE_DATA_ROOT/step2_<exp>_queue.txt 22 800 60
python -m ex.utils.step2_runner.cpu_dispatcher \
    --queue-file $DPE_DATA_ROOT/step2_<exp>_queue.txt \
    --array-size 64 --concurrency 100 --n-per-element 2 \
    --output-root $DPE_DATA_ROOT/<exp>/step2_dual/$(date +%Y%m%d)/cpu_array \
    --method-filter BDRE,MDRE_15,CTSM \
    --walltime 2:00:00 \
    --device cpu

# gather as usual when results land (idempotent; safe to run incrementally)
python -m ex.utils.step2_runner.gather --experiment model_selection
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

1. write `ex/<exp>/step2_adapter.py` implementing the contract above.
2. create a `winners.<exp>.<tag>.yaml` (any path; passed via `--winners`).
3. run the dispatch + watchdog + gather flow as above.

experiment-specific quirks (e.g., eig's TriangularTSMEIGAdapter wrappers,
smodice's SUPPORTED_ENCODINGS dispatch) belong in the per-experiment adapter,
not the shared library.
