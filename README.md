# ELDR Estimation for Experimental Design

## Installation

1. Run `bash setup.sh` to create the virtual environment and install dependencies
2. Activate the virtual environment via `source venv/bin/activate`

dependencies: `numpy`, `scipy`, `torch`, `matplotlib`, `einops`, `seaborn`, `ipython` (optional), `tqdm`, `pyyaml`, `h5py`

## Organization

- `src/` - implementations of algorithms, models, and their APIs for our tasks
  - `density_ratio_estimation/` - Density Ratio Estimation methods (BDRE, TDRE, MDRE, TSM, triangular/spatial variants thereof)
  - `eldr_estimation/` - ELDR methods that accept samples from three distributions
  - `eig_estimation/` - EIG estimation APIs
  - `waypoints/` - waypoint generation for telescoping methods
  - `models/` - NN models
    - `binary_classification/` - binary classifiers for bdre, tdre
    - `multiclass_classification/` - multiclass classifiers for mdre
    - `regression/` - template for regression models, may be used to extend TSM-like methods.
    - `time_score_matching/` - time score matching models

- `experiments/` - experiment pipelines for reproduction
  - `density_ratio_estimation/` - DRE method comparison (Section 4.2)
  - `dre_sample_complexity/` - number of training samples vs DRE performance (Section 4.2)
  - `plugin_dre/` - visualization of DRE performance over input space
  - `elbo_estimation/` - ELBO estimation experiments (Section 6.1)
  - `eig_estimation/` - EIG estimation experiments (Section 6.2)

## Core Abstractions

**DensityRatioEstimator** (`src/density_ratio_estimation/base.py`)
- `fit(samples_p0, samples_p1)` - train on samples from two distributions
- `predict_ldr(xs)` - predict log density ratio at points xs

<!-- **KLEstimator** (`src/kl_estimation/base.py`)
- `estimate_kl(samples_p0, samples_p1)` - estimate kl divergence -->

**ELDREstimator** (`src/eldr_estimation/base.py`)
- `estimate_eldr(samples_base, samples_p0, samples_p1)` - estimate expected log density ratio

**EIGEstimator** (`src/eig_estimation/base.py`)
- `estimate_eig(samples_theta, samples_y)` - estimate expected information gain

**Plugin estimation** (`src/eldr_estimation/plugins.py`, `src/eig_estimation/plugin.py`): Non-triangular methods compose DRE methods for higher-level estimation
- ELDR plugin: fits any dre method on (p0, p1), evaluates ldr on samples from base distribution, returns empirical mean (monte carlo estimate of expected ldr)
- EIG plugin: constructs joint samples (theta, y) as p0 and shuffled marginals as p1, fits dre, returns mean ldr (mutual information via kl between joint and product of marginals)
- implementation: accepts any `DensityRatioEstimator` as dependency injection, enabling comparison of bdre/tdre/mdre/tsm/vfm as subroutines

## DRE methods

- **BDRE**: Binary classification-based DRE (single classifier, p0 vs p1)
- **TDRE**: Telescoping DRE (multiple binary classifiers, one per waypoint transition)
- **MDRE**: Multiclass classification-based DRE (single multiclass classifier)
- **TSM**: Time Score Matching baseline
- **Triangular variants**: `triangular_tdre`, `triangular_mdre`, `triangular_tsm`
- **VFM**: Velocity Flow Matching, various estimands possible as described in appendix but `spatial_velo_denoiser2` is sufficient to reproduce the results. 

## Running experiments

Each experiment suite follows a numbered step pipeline. Run steps as modules sequentially from project root.

**example: `experiments.density_ratio_estimation`**
```bash
python -m experiments.density_ratio_estimation.step1_create_data
python -m experiments.density_ratio_estimation.step2_run_algorithms
python -m experiments.density_ratio_estimation.step3_process_results
python -m experiments.density_ratio_estimation.step4_plot_results
```

experiment configurations are in yaml files (e.g., `experiments/density_ratio_estimation/config1.yaml`).

## Configuration

experiments are configured via yaml files. common parameters:

```yaml
# directory paths for data and results
data_dir: "experiments/density_ratio_estimation/data"
raw_results_dir: "experiments/density_ratio_estimation/raw_results"
processed_results_dir: "experiments/density_ratio_estimation/processed_results"
figures_dir: "experiments/density_ratio_estimation/figures"

# experiment parameters
data_dim: 3                                      # dimensionality of data
device: "cuda"                                   # "cuda" or "cpu"
seed: 1729                                       # random seed for reproducibility
```

Experiment-specific parameters vary by task:

**density_ratio_estimation** ([config1.yaml](experiments/density_ratio_estimation/config1.yaml))
```yaml
gamma: 0.05                                      # covariance scale parameter
kl_divergences: [0.3, 1, 3, 9, 18, 36, 54]       # kl divergences b/w p_0, p_1 to test
num_instances_per_kl: 10                         # instances per kl value
nsamples_train: 2048                             # training samples
nsamples_test: 1024                              # test samples
```

**dre_sample_complexity** ([config.yaml](experiments/dre_sample_complexity/config.yaml))
```yaml
nsamples_train_values: [100, 300, 900, 1800, 3600, 5400, 8100]  # sample sizes
```

**eig_estimation** ([config1.yaml](experiments/eig_estimation/config1.yaml))
```yaml
eig_min: 0.5                                     # minimum eig value
eig_max: 2                                       # maximum eig value
design_eig_percentages: [0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
```

**plugin_dre** ([config.yaml](experiments/plugin_dre/config.yaml))
```yaml
grid_size: 50                                    # evaluation grid resolution
tdre_waypoints: [5]                              # waypoint counts for tdre
mdre_waypoints: [15]                             # waypoint counts for mdre
```

## Tensor conventions

- samples: `[batch_size, dim]`
- waypoints: `[num_waypoints, batch_size, dim]`
- binary labels: `[batch_size, 1]` (float 0.0 or 1.0)
- multiclass labels: `[batch_size]` (long integer class indices)
- ldr outputs: `[batch_size]` (1d tensor of log density ratios)

## Testing implementations

main modules have `__main__` blocks for standalone testing:

```bash
python -m src.density_ratio_estimation.tdre
python -m src.density_ratio_estimation.mdre
python -m src.density_ratio_estimation.bdre
```
