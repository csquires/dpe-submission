# ELDR Estimation

Code for *Estimating Implicit Expected Log Density Ratios*.

## Installation

1. Run `bash setup.sh` from the project root. This creates a `venv/` virtual environment and `pip install`s every direct dependency.
2. Activate it for subsequent shells: `source venv/bin/activate` (or `conda activate fac` if you maintain the project's conda env instead).

`setup.sh` installs `numpy`, `scipy`, `torch`, `matplotlib`, `einops`, `seaborn`, `ipython`, `tqdm`, `pyyaml`, `h5py`, plus the DBpedia conditional-flow extras (`sentence-transformers`, `datasets`, `scikit-learn`). Add the HPO stack on top:

```bash
pip install optuna joblib kaleido
```

## Environment variables

A few env vars are read at runtime.

| variable | required by | default | meaning |
| --- | --- | --- | --- |
| `DPE_DATA_ROOT` | everything | (required, must be set) | nfs-shared root for experiment data and HPO storage. the redis endpoint is published at `$DPE_DATA_ROOT/redis/endpoint`. |
| `DPE_CKPT_ROOT` | every `python -m ex.*` invocation | (required, must be set) | node-local scratch root for checkpoints and run logs. `ex/__init__.py` raises at import if unset. |
| `DPE_CONDA_ENV` | some slurm wrappers | `fac` | conda env activated inside the wrapper. |
| `DPE_REDIS_ENDPOINT_FILE` | optuna workers | unset | alternate redis endpoint file, for workers on one account against a redis on another. |
| `DPE_PREEMPT_CONSTRAINT` | `submit.sh` (preempt lane) | lane default | override of the preempt lane's gpu node constraint. |
| `SLURM_ARRAY_TASK_ID` | optuna `submit.py` | set by slurm | identifies the `(experiment, method)` combo an array element handles. use `--combo-index` outside slurm. |
| `SLURM_CONCURRENCY` | `submit.sh` | `16` | array concurrency cap (array elements in flight at once). |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | optuna worker | set by `worker.py` to `cores_per_trial` before `torch` import | per-trial BLAS thread budget. set automatically; do not export ahead of time. |

A minimal one-off setup:

```bash
bash setup.sh
source venv/bin/activate
pip install optuna joblib kaleido         # only if using HPO
export DPE_DATA_ROOT=/path/to/nfs/scratch # only if using HPO
export DPE_CKPT_ROOT=/path/to/scratch    # required by ex.*
```

### Slurm specifics

The HPO stack targets a Slurm cluster. The cluster-specific knobs to change or fix are:

- **lane profiles** (`ex/utils/hpo/optuna/lanes.py`): each lane pins `partition`, `qos`, `gpus`, `cpus_per_task`, `mem`, `worker_walltime`, `max_concurrent`, and an optional gpu `constraint`. Edit these (or add a lane) to match the target cluster's partitions and QoS limits.
- **`SLURM_ARRAY_TASK_ID`**: set by Slurm; resolves the `(experiment, method)` combo for an array element. Outside Slurm, pass `--combo-index` (see HPO).
- **`SLURM_CONCURRENCY`**: optional array concurrency cap for `submit.sh`.
- **`redis_server.sh`**: the sbatch header (partition, walltime, cpus, mem) sizes the long-lived redis/keeper job.
- **conda env**: the slurm wrappers run `source ~/.bashrc && conda activate fac` (or `$DPE_CONDA_ENV`); point them at the local environment.
- **`DPE_DATA_ROOT` / `DPE_CKPT_ROOT`**: usually exported by cluster setup (e.g. `~/.bashrc`). They are hard errors when unset; there is no silent default.

## Repository organization

- `src/` - implementations of algorithms, models, and their APIs.
  - `methods/` - density ratio estimators. Split into `cls/` (classification-based: BDRE, TDRE family, MDRE, tabular plug-in) and `reg/` (regression / score-based: TSM, CTSM, FMDRE, VFM); shared base classes and the training loop in `common/`.
  - `waypoints/` - waypoint generators for telescoping and triangular methods.
  - `sampling/` - data samplers (gibbs, frozen-flow, tabular, pendulum trajectories).
  - `models/` - neural-network backbones for classifiers, regressors, flows, VAEs.
  - `utils/` - shared utilities (i/o, gridworld, pendulum dynamics, etc.).

- `ex/` - reproducible experiment pipelines, grouped by data regime.
  - `synth/` - synthetic experiments with closed-form ground truth: `eig/`, `elbo/`, `model_selection/`, `occupancy/`.
  - `semisynth/` - semi-synthetic experiments combining real-data components with synthesized distribution structure: `mnist/`, `mnist_uncond/`, `dbpedia/`, `pendulum/`.
  - `ablations/` - secondary studies and analysis tooling: `dre_sample_complexity/`, `pstar_sample_complexity/`, `plugin_dre/`, `dre_hidden_dim_scaling/`, `hidden_dim_scaling/`, `eig_vertex_sweep/`, `analysis/` (cross-experiment aggregation).
  - `utils/hpo/` - the Optuna HPO stack and the per-experiment adapters that drive it (see "Hyperparameter optimization" below).
  - `utils/step2_runner/` - distributed post-HPO runner used by some experiments to fan winning hyperparameters across slurm jobs.

## Estimators

**DRE** (`src/methods/common/base.py`)
- `fit(samples_p0, samples_p1, *, step_cb=None, eval_data=None, step_cb_interval=50)` - train on samples from two distributions. the three keyword-only arguments are optional HPO instrumentation hooks (see HPO).
- `predict_ldr(xs)` - per-sample log density ratios at `xs`, shape `[N]`.
- `predict_eldr(xs)` - expected log density ratio: `mean(predict_ldr(xs))`. the natural scalar summary; subclasses may override for smarter reductions. used directly as the EIG estimate when `xs` are joint samples and as the ELDR estimate when `xs` are p* samples.

**ELDR** (`src/methods/common/base.py`)
- subclass of `DRE` whose `fit` also accepts `samples_pstar`. enforced via an `__init_subclass__` hook that inspects the positional-parameter prefix at class-definition time.

**EIG via density-ratio estimation** (`ex/utils/eig_ldr.py`)
- `joint_and_shuffled(theta, y)` builds the (p0, p1) pair: p0 = concat(theta, y) and p1 = independently-shuffled rows of theta and y. fitting any DRE on this pair and calling `predict_eldr(joint)` recovers the MI between theta and y.
- `true_ldrs_gaussian_linear(theta, y, mu_pi, Sigma_pi, xi)` returns the closed-form per-sample log ratio for the gaussian linear model. used as the HPO eval signal (MAE on r) for the `eig` experiment.

**Method roster**

- **BDRE**: binary classification (p0 vs p1) via a single classifier.
- **TDRE**: telescoping DRE; multiple binary classifiers, one per adjacent waypoint pair. the `MultiHeadTDRE` and `MultiHeadTriangularTDRE` variants share a backbone across heads.
- **MDRE**: multiclass classifier across all waypoints.
- **TSM**, **CTSM**: time score matching and its conditional variant.
- **FMDRE**: flow matching DRE (simulate along numerator `s1`, simulate along unconditional flow `s2`).
- **VFM**: velocity flow matching with two-phase training (velocity then denoiser).
- **Triangular variants**: `triangular_tdre`, `triangular_mdre`, `triangular_tsm`, `triangular_ctsm`, `triangular_vfm`, `triangular_fmdre`. consume a reference `samples_pstar` and decompose the ratio along p0 -> pstar -> p1.

## Experiment pipeline

Every experiment follows a numbered-step convention. Run steps as modules from the project root, in order:

```bash
# <regime> is "synth" or "semisynth"; <exp> is the experiment subdir under it.
python -m ex.<regime>.<exp>.step0_pretrain          # optional, encoder pretraining
python -m ex.<regime>.<exp>.step1_create_data       # generate per-cell h5 data
# run HPO (below) to produce the winners.yaml step2 consumes
python -m ex.<regime>.<exp>.step2_run_algorithms    # post-HPO full-budget eval
python -m ex.<regime>.<exp>.step3_process_results   # aggregate to metrics
python -m ex.<regime>.<exp>.step4_plot_results      # generate figures
```

- **step0** (optional, present in `mnist`, `mnist_uncond`, `dbpedia`): pretrain a feature extractor used downstream (conditional flow, MLM-style head, etc.).
- **step1_create_data**: build the per-cell hdf5 files that downstream steps consume. a "cell" is one evaluation unit (e.g. one (alpha, beta) pair on mnist, one (k1, k2, seed) tuple on pendulum). cells are tuples of ints; arity is per-experiment.
- **step2_adapter**: declarative adapter class used by HPO. exposes `cell_pool`, `load_cell_data`, `metric_key`, `latent_dim`, optionally `stratify_key`, and an overridable `eval_cell`. consumed by the Optuna driver; not a runnable script.
- **step2_run_algorithms**: post-HPO evaluation. reads winning hyperparameters from a `winners.yaml` (one entry per `(method, cell)` group) and runs the full-budget fit + predict across all cells. for experiments wired into the distributed runner, `ex/utils/step2_runner/` orchestrates this across a slurm array.
- **step3_process_results**: aggregate the raw per-cell results into summary metrics. writes `processed_results/metrics.h5`.
- **step4_plot_results**: render figures from `processed_results/`. plots land in `figures/`.

Raw per-cell outputs land in `ex/<regime>/<exp>/raw_results/`, aggregated metrics in `ex/<regime>/<exp>/processed_results/`, and figures in `ex/<regime>/<exp>/figures/`. All paths are configurable per-experiment via yaml.

## Hyperparameter optimization

HPO is driven by Optuna and lives under `ex/utils/hpo/`. The stack:

- **`adapters/`**: per-experiment data and metric definitions consumed by the trial loop. each adapter inherits `ExperimentAdapter` (`adapters/base.py`) and declares `cell_pool`, `load_cell_data`, `metric_key`, `latent_dim`, and optional overrides. the base class provides `train_pool` / `holdout_pool` (cell-level stratified split, see `adapters/split_utils.py`) and `split_for_eval` (within-cell paired split of `pstar` + `true_ldrs`, see `adapters/eval_split.py`).
- **`optuna/`**: the Optuna driver.
  - `storage.py` - Redis-journal-backed study storage. studies are namespaced per `(experiment, method)`.
  - `study_config.py` - `StudyConfig` dataclass + `load_config` for python-config files.
  - `objective.py` - the per-trial closure: picks a cell from `adapter.train_pool()` via `stratified_pick`, suggests hyperparameters, constructs the Hyperband pruning `step_cb`, and calls `adapter.eval_cell(...)`.
  - `worker.py` - loky worker entrypoint; sets BLAS thread env vars before `torch` import, then drives `study.optimize`.
  - `submit.py` + `submit.sh` - slurm array entrypoint. resolves `(experiment, method)` from `SLURM_ARRAY_TASK_ID`, fans out loky workers per array element.
  - `probe.py` - reconstructs the TPE Parzen posterior at a chosen budget step and returns the top-k hyperparameters by log-density.
  - `holdout.py` - re-evaluates the probe's top-k on the adapter's holdout cell pool at full budget; writes per-cell JSON and a summary CSV.
  - `configs/` - python config files defining `StudyConfig` instances per study (e.g. `bdre_pilot.py`).
- **`suggest_hp/`**: per-method `suggest_hp(trial: optuna.Trial) -> dict` plus a `METADATA` dict declaring `cores_per_trial`, `uses_pruning`, `requires_pstar`, and the builder key.
- **`builders.py`** / **`method_specs.py`**: `BUILDERS_REGISTRY` maps a method label to an estimator builder `(input_dim, device, num_waypoints, **flat_hp) -> estimator`; `METHOD_SPECS` is the canonical per-method search-space declaration, consumed by `step2_run_algorithms` and by future `suggest_hp` additions.

Every method whose `suggest_hp` declares `uses_pruning=True` invokes a `do_report` closure once per SGD step (a no-op when no `step_cb` is attached), enabling Hyperband pruning. the eval score for every method is `MAE(predict_ldr(eval_pstar), eval_true_ldrs)` on the adapter's per-trial within-cell eval split.

### Storage

Studies persist in a shared Redis journal. A long-lived redis-server (launched by `redis_server.sh`) is the only process that writes study state to disk; workers reach it over tcp. Start it before any campaign:

```bash
bash ex/utils/hpo/optuna/redis_server.sh    # sbatch it, or run directly in a
                                            # terminal (OPTREDIS_NOCHAIN=1)
```

The server publishes its address at `$DPE_DATA_ROOT/redis/endpoint`. To drive workers against a redis on another account, point `DPE_REDIS_ENDPOINT_FILE` at an alternate endpoint file.

### Define a study

Each study is a python `StudyConfig` module:

```python
# ex/utils/hpo/optuna/configs/bdre_pilot.py
from ex.utils.hpo.optuna.study_config import StudyConfig

CONFIG = StudyConfig(
    study_seed=1729,
    experiment="dre_sample_complexity",
    methods=["BDRE"],
    min_resource=100,
    max_resource=10000,
    reduction_factor=3,
    holdout_top_k=5,
    walltime_minutes=120,
    walltime_margin_minutes=10,
    resume_existing=True,
    include_tabular=False,
)
```

### Submit with slurm

```bash
export DPE_DATA_ROOT=/path/to/nfs/scratch
bash ex/utils/hpo/optuna/submit.sh \
  --config ex.utils.hpo.optuna.configs.bdre_pilot \
  --lane array
```

`submit.sh` submits one array element per `(experiment, method)` pair; `--lane` selects a compute profile (partition, qos, gpus, cpus, mem, walltime, concurrency cap) from `ex/utils/hpo/optuna/lanes.py`. `--replicas R` repeats the array (for multi-seed campaigns) and `--concurrency K` caps array elements in flight.

### Without slurm

The same entrypoint runs standalone; `--combo-index` replaces `SLURM_ARRAY_TASK_ID`:

```bash
export DPE_DATA_ROOT=/path/to/nfs/scratch
python -m ex.utils.hpo.optuna.submit \
  --config ex.utils.hpo.optuna.configs.bdre_pilot \
  --lane array --combo-index 0
```

`--lane` is still used to size batch size and cores per trial. For finer control, drive the loop directly from a script:

```python
from ex.utils.hpo.optuna.worker import run_worker

run_worker(experiment="dre_sample_complexity", method="BDRE", ...)
```

Either way, `DPE_DATA_ROOT` must be set and a redis server must be reachable.

### After a study

Once a study reaches its target, select winners:

```python
from ex.utils.hpo.optuna.probe import best_at_budget
from ex.utils.hpo.optuna.holdout import run_holdout

best_at_budget(study, budget_step=10000, k=5)   # top-k hp by TPE log-density
run_holdout(study, adapter, method, builder)    # full-budget retest on the holdout pool
```

`holdout` writes per-(hp, cell) JSON and a summary CSV. Translate the results into the `winners.yaml` that `step2_run_algorithms` consumes (schema documented in `ex/utils/step2_runner/load_winners.py`).

## Configuration

Per-experiment configuration lives in `ex/<exp>/config.yaml`. Path keys are templated with `${...}` env vars at load time (`ex/__init__.py` expands every string in the parsed yaml), so datasets live on `DPE_DATA_ROOT` rather than in the repo. Common parameters:

```yaml
data_dir: "${DPE_DATA_ROOT}/model_selection/data"
raw_results_dir: "ex/synth/model_selection/raw_results"
processed_results_dir: "ex/synth/model_selection/processed_results"
figures_dir: "ex/synth/model_selection/figures"

data_dim: 3
device: "cuda"
seed: 1729
```

Experiment-specific parameters vary by task; see each experiment's config for the full list. HPO studies are configured separately as python `StudyConfig` modules under `ex/utils/hpo/optuna/configs/` (see "Hyperparameter optimization" above).

## Tensor conventions

- samples: `[batch_size, dim]`
- waypoints: `[num_waypoints, batch_size, dim]`
- binary labels: `[batch_size, 1]` (float 0.0 or 1.0)
- multiclass labels: `[batch_size]` (long integer class indices)
- ldr outputs: `[batch_size]` (1d tensor of log density ratios)
- eval_data: `dict[str, Tensor]` with at least `"pstar"` and `"true_ldrs"` paired by row index.
