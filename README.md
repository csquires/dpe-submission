# Deep Preemptive Exploration Code

## Organization
- `src/` contains source code, including baseline algorithms
    - `density_ratio_estimation/` contains method for density ratio estimation
    - `eldr_estimation/` contains method for ELDR estimation
    - `kl_estimation/` contains method for KL estimation
    - `utils/` contains utilities used throughout various approaches
- `experiments/` contains code used for reproducing the experiments in the paper
    - `eldr_estimation/` contains code to reproduce the experiments on ELDR estimation (Section X.X)
    - `experimental_design/` contains code to reproduce the experiments on experimental design (Section X.X)

## Installation
1. Run `bash setup.sh` to create the virtual environment and install dependencies
2. Activate the virtual environment via `source venv/bin/activate`

## Experiment Reproduction

### Preliminary Experiments: Density Ratio Estimation
1. Run `experiments/density_ratio_estimation/step1_create_data.py`
2. Run `experiments/density_ratio_estimation/step2_run_compute_true_ldrs.py`
2. Run `experiments/density_ratio_estimation/step3_run_algorithms.py`
3. Run `experiments/density_ratio_estimation/step4_process_results.py`
4. Run `experiments/density_ratio_estimation/step5_plot_results.py`

### Main Experiments: Experimental Design
1. Run `experiments/experimental_design/step1_create_data.py`
2. Run `experiments/experimental_design/step2_run_algorithms.py`
3. Run `experiments/experimental_design/step3_process_results.py`
4. Run `experiments/experimental_design/step4_plot_results.py`

Other repos (remove prior to submission):
- https://github.com/csquires/deep-preemptive-exploration
- https://github.com/YizhouLu-Johnson/DRE_Eval