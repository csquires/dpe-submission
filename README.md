# Deep Preemptive Exploration Code

## Organization
- `src/` contains source code, including baseline algorithms
    - `density_ratio_estimation/` contains methods for density ratio estimation
    - `eldr_estimation/` contains methods for ELDR estimation
    - `kl_estimation/` contains methods for KL estimation
    - `models/` contains models used throughout various approaches
        - `binary_classification/` contains models for binary classification
        - `multiclass_classification/` contains models for multi-class classification
        - `regression/` contains models for regression
    - `utils/` contains utilities used throughout various approaches
- `experiments/` contains code used for reproducing the experiments in the paper
    - `eldr_estimation/` contains code to reproduce the experiments on ELDR estimation (Section X.X)
    - `experimental_design/` contains code to reproduce the experiments on experimental design (Section X.X)

## Installation
1. Run `bash setup.sh` to create the virtual environment and install dependencies
2. Activate the virtual environment via `source venv/bin/activate`

## Experiment Reproduction

### Preliminary Experiments: Density Ratio Estimation
1. Configuration is described in `experiments/density_ratio_estimation/config1.yaml`
2. Run `experiments/density_ratio_estimation/step1_create_data.py`
3. Run `experiments/density_ratio_estimation/step2_run_algorithms.py`
4. Run `experiments/density_ratio_estimation/step3_process_results.py`
5. Run `experiments/density_ratio_estimation/step4_plot_results.py`

### Main Experiments: Experimental Design
1. Run `experiments/experimental_design/step1_create_data.py`
2. Run `experiments/experimental_design/step2_run_algorithms.py`
3. Run `experiments/experimental_design/step3_process_results.py`
4. Run `experiments/experimental_design/step4_plot_results.py`
