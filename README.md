# ELDR Estimation

## Installation

1. Run `bash setup.sh` from the project root. this creates a `venv/` virtual environment and `pip install`s every direct dependency.
2. Activate it for subsequent shells: `source venv/bin/activate` (or `conda activate fac` if you maintain the project's conda env instead).

dependencies installed by `setup.sh`: `numpy`, `scipy`, `torch`, `matplotlib`, `einops`, `seaborn`, `ipython` (optional, interactive), `tqdm`, `pyyaml`, `h5py`, plus the DBpedia ELDR conditional-flow extras `sentence-transformers`, `datasets`, `scikit-learn`. add `optuna`, `joblib`, and `kaleido` (optional, for plotly PNG export) on top of `setup.sh` if you intend to run the HPO stack:

```bash
pip install optuna joblib kaleido
```

## Environment

a few env vars are read at runtime.

| variable | required by | default | meaning |
| --- | --- | --- | --- |
| `DPE_DATA_ROOT` | Optuna storage, slurm submit | (required, must be set) | nfs-shared root for journal files. studies persist under `$DPE_DATA_ROOT/<experiment>/hpo_optuna/<method>.journal`. |
| `SLURM_ARRAY_TASK_ID` | `ex/utils/hpo/optuna/submit.py` | (set by slurm) | identifies the `(experiment, method)` combo this array element should handle. |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | Optuna worker | set by `worker.py` to `cores_per_trial` before `torch` import | per-trial BLAS thread budget. set automatically; do not export ahead of time. |
| `DPE_CORES_PER_NODE`, `DPE_MEM_PER_NODE` | `submit.sh` | `16`, `32G` | per-job slurm resource defaults; override via flags or env. |
| `SLURM_PARTITION`, `SLURM_TIME`, `SLURM_CONCURRENCY` | `submit.sh` | `cpu`, `06:00:00`, `16` | per-job slurm allocation; override via flags or env. |

a minimal one-off setup:

```bash
bash setup.sh
source venv/bin/activate
pip install optuna joblib kaleido         # only if using HPO
export DPE_DATA_ROOT=/path/to/nfs/scratch # only if using HPO
```

## Organization

- `src/` - implementations of algorithms, models, and their APIs.
  - `methods/` - density ratio estimators. split into `cls/` (classification-based: BDRE, TDRE family, MDRE, tabular plug-in) and `reg/` (regression / score-based: TSM, CTSM, FMDRE, VFM); shared base classes and the training loop in `common/`.
  - `waypoints/` - waypoint generators for telescoping and triangular methods.
  - `sampling/` - data samplers (gibbs, frozen-flow, tabular, pendulum trajectories).
  - `models/` - neural-network backbones for classifiers, regressors, flows, VAEs.
  - `utils/` - shared utilities (i/o, gridworld, pendulum dynamics, etc.).

- `ex/` - reproducible experiment pipelines, grouped by data regime.
  - `synth/` - synthetic experiments with closed-form ground truth: `eig/`, `elbo/`, `model_selection/`, `occupancy/`.
  - `semisynth/` - semi-synthetic experiments combining real-data components with synthesized distribution structure: `mnist/`, `mnist_uncond/`, `dbpedia/`, `pendulum/`.
  - `ablations/` - secondary studies and analysis tooling: `dre_sample_complexity/`, `pstar_sample_complexity/`, `plugin_dre/`, `dre_hidden_dim_scaling/`, `hidden_dim_scaling/`, `eig_vertex_sweep/`, `analysis/` (cross-experiment aggregation).
  - `utils/hpo/` - the Optuna HPO stack and the per-experiment adapters that drive it (see "HPO" below).
  - `utils/step2_runner/` - distributed post-HPO runner used by some experiments to fan winning hyperparameters across slurm jobs.

- `plans/` - low-level specs for major refactors. each subdirectory is one plan, one markdown file per source file touched.
- `tests/` - cross-cutting test suite for `src/`, the Optuna stack, and adapter signatures.

## Core Abstractions

**DRE** (`src/methods/common/base.py`)
- `fit(samples_p0, samples_p1, *, step_cb=None, eval_data=None, step_cb_interval=50)` - train on samples from two distributions. the three keyword-only arguments are optional HPO instrumentation hooks (see HPO).
- `predict_ldr(xs)` - per-sample log density ratios at `xs`, shape `[N]`.
- `predict_eldr(xs)` - expected log density ratio: `mean(predict_ldr(xs))`. the natural scalar summary; subclasses may override for smarter reductions. used directly as the EIG estimate when `xs` are joint samples and as the ELDR estimate when `xs` are p* samples.

**ELDR** (`src/methods/common/base.py`)
- subclass of `DRE` whose `fit` also accepts `samples_pstar`. enforced via an `__init_subclass__` hook that inspects the positional-parameter prefix at class-definition time.

**EIG via density-ratio estimation** (`ex/utils/eig_ldr.py`)
- `joint_and_shuffled(theta, y)` builds the (p0, p1) pair: p0 = concat(theta, y) and p1 = independently-shuffled rows of theta and y. fitting any DRE on this pair and calling `predict_eldr(joint)` recovers the MI between theta and y.
- `true_ldrs_gaussian_linear(theta, y, mu_pi, Sigma_pi, xi)` returns the closed-form per-sample log ratio for the gaussian linear model. used as the HPO eval signal (MAE on r) for the `eig` experiment.

## DRE methods

- **BDRE**: binary classification (p0 vs p1) via a single classifier.
- **TDRE**: telescoping DRE; multiple binary classifiers, one per adjacent waypoint pair. the `MultiHeadTDRE` and `MultiHeadTriangularTDRE` variants share a backbone across heads.
- **MDRE**: multiclass classifier across all waypoints.
- **TSM**, **CTSM**: time score matching and its conditional variant.
- **FMDRE**: flow matching DRE (single-stage `s1`, two-stage `s2`, triangular `tri`).
- **VFM**: velocity flow matching with two-phase training (velocity then denoiser).
- **Triangular variants**: `triangular_tdre`, `triangular_mdre`, `triangular_tsm`, `triangular_ctsm`, `triangular_vfm`, `triangular_fmdre`. consume a reference `samples_pstar` and decompose the ratio along p0 -> pstar -> p1.
- **TabularPluginDRE**, **SmoothedTabularPluginDRE**: oracle plug-in estimators for discrete state-action spaces.

## Experiment Pipeline

each experiment follows a numbered-step convention. run steps as modules from the project root.

```bash
# <regime> is "synth" or "semisynth"; <exp> is the experiment subdir under it.
python -m ex.<regime>.<exp>.step0_pretrain          # optional, encoder pretraining
python -m ex.<regime>.<exp>.step1_create_data       # generate per-cell h5 data
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
- **step5_compare** (present in `mnist` only): cross-method comparison plots.

raw per-cell outputs land in `ex/<regime>/<exp>/raw_results/` and aggregated metrics in `ex/<regime>/<exp>/processed_results/`. figures land in `ex/<regime>/<exp>/figures/`. all paths are configurable per-experiment via yaml.

## HPO

hyperparameter optimization is driven by Optuna and lives under `ex/utils/hpo/`. three sibling subpackages:

- **`adapters/`**: per-experiment data + metric definitions consumed by the trial loop. each adapter inherits `ExperimentAdapter` (`adapters/base.py`) and declares `cell_pool`, `load_cell_data`, `metric_key`, `latent_dim`, and optional overrides. the base class also provides `train_pool` / `holdout_pool` (cell-level stratified split, see `adapters/split_utils.py`) and `split_for_eval` (within-cell paired split of `pstar` + `true_ldrs`, see `adapters/eval_split.py`).
- **`optuna/`**: the Optuna driver.
  - `storage.py` - JournalStorage-backed study at `$DPE_DATA_ROOT/<experiment>/hpo_optuna/<method>.journal`, with `create_or_load` and `cleanup_zombies`.
  - `study_config.py` - `StudyConfig` dataclass + `load_config` for python-config files.
  - `cores_registry.py` - per-method `cores_per_trial` defaults; overridable.
  - `objective.py` - the per-trial closure. picks a cell from `adapter.train_pool()` via `stratified_pick`, suggests hyperparameters via `suggest_hp`, constructs the `step_cb` callback (Hyperband pruning), and calls `adapter.eval_cell(..., trial_number=trial.number, step_cb=step_cb, step_cb_interval=50)`.
  - `worker.py` - loky worker entrypoint; sets BLAS thread env vars before `torch` import, then drives `study.optimize` with `RetryFailedTrialCallback`.
  - `submit.py` + `submit.sh` - slurm array entrypoint. resolves `(experiment, method)` from `SLURM_ARRAY_TASK_ID`, fans out `n_jobs_per_task` loky workers via `joblib.Parallel(backend='loky')`.
  - `probe.py` - reconstructs the TPE Parzen posterior at a chosen budget step and returns the top-k hyperparameters by log-density.
  - `holdout.py` - re-evaluates the probe's top-k on the adapter's holdout cell pool at full budget; writes per-cell JSON and a summary CSV.
  - `figures.py` - optimization history, intermediate values, parallel coordinate, slice, parameter importance plots (HTML via plotly when available; PNG via matplotlib).
  - `configs/` - python config files defining `StudyConfig` instances per study (e.g. `bdre_pilot.py`).
- **`suggest_hp/`**: per-method `suggest_hp(trial: optuna.Trial) -> dict` plus a `METADATA` dict declaring `cores_per_trial`, `uses_pruning`, `requires_pstar`, and the builder key. four methods are currently registered: BDRE, MultiHeadTriangularTDRE, TriangularFMDRE, TabularPluginDRE.

**Builders and method specs.** `ex/utils/hpo/builders.py` exposes `BUILDERS_REGISTRY: dict[str, Callable]` mapping a method label to a builder that takes `(input_dim, device, num_waypoints, **flat_hp)` and returns an estimator. `ex/utils/hpo/method_specs.py` exposes `METHOD_SPECS` with the canonical per-method search-space declaration; this is the source of truth for `step2_run_algorithms` and for any future suggest_hp additions.

**Step-callback pruning.** every method whose `suggest_hp` declares `uses_pruning=True` invokes a `do_report` closure once per SGD step. the closure is bound by `src/methods/common/_report.py::_make_report` and returns `_noop` when either `step_cb` or `eval_fn` is absent, so the hot path performs zero per-step branching on the disabled case. instrumented training loops: `src/methods/reg/common/_trainer.py::train_loop`, `src/models/binary_classification/default_binary_classifier.py::fit`, `src/models/binary_classification/multi_head_binary_classifier.py::fit`. the eval score for every method is `MAE(predict_ldr(eval_pstar), eval_true_ldrs)` on the adapter's per-trial within-cell eval split.

**Submitting a study.**

```bash
export DPE_DATA_ROOT=/path/to/nfs/scratch
bash ex/utils/hpo/optuna/submit.sh \
  --config ex.utils.hpo.optuna.configs.bdre_pilot \
  --partition cpu --time 06:00:00 --cpus 16 --concurrency 16
```

a minimal `StudyConfig`:

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

after a study completes, run `probe.best_at_budget(study, budget_step=10000, k=5)` for top-k hyperparameter inspection and `holdout.run_holdout(study, adapter, method, builder)` for a held-out cell-pool retest. `holdout` writes per-(hp, cell) JSON and a summary CSV that downstream `step2_run_algorithms` can consume by translating to `winners.yaml`.

## Configuration

per-experiment configuration lives in `ex/<exp>/config.yaml`. common parameters:

```yaml
data_dir: "ex/synth/model_selection/data"
raw_results_dir: "ex/synth/model_selection/raw_results"
processed_results_dir: "ex/synth/model_selection/processed_results"
figures_dir: "ex/synth/model_selection/figures"

data_dim: 3
device: "cuda"
seed: 1729
```

experiment-specific parameters vary by task. examples:

**model_selection** ([config1.yaml](ex/synth/model_selection/config1.yaml))
```yaml
gamma: 0.05
kl_divergences: [0.5, 2, 8, 32, 128]
num_instances_per_kl: 10
nsamples_train: 2048
nsamples_test: 1024
```

**dre_sample_complexity** ([config.yaml](ex/ablations/dre_sample_complexity/config.yaml))
```yaml
nsamples_train_values: [100, 300, 900, 1800, 3600, 5400, 8100]
```

**eig** ([config1.yaml](ex/synth/eig/config1.yaml))
```yaml
eig_min: 0.5
eig_max: 2
design_eig_percentages: [0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
```

**plugin_dre** ([config.yaml](ex/ablations/plugin_dre/config.yaml))
```yaml
grid_size: 50
tdre_waypoints: [5]
mdre_waypoints: [15]
```

HPO studies are configured separately as python `StudyConfig` modules under `ex/utils/hpo/optuna/configs/` (see HPO section above).

## Tensor conventions

- samples: `[batch_size, dim]`
- waypoints: `[num_waypoints, batch_size, dim]`
- binary labels: `[batch_size, 1]` (float 0.0 or 1.0)
- multiclass labels: `[batch_size]` (long integer class indices)
- ldr outputs: `[batch_size]` (1d tensor of log density ratios)
- eval_data: `dict[str, Tensor]` with at least `"pstar"` and `"true_ldrs"` paired by row index.
