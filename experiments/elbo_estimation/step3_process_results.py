import os

import h5py
import numpy as np
import torch
from torch.distributions import MultivariateNormal
import yaml


config = yaml.load(open('experiments/elbo_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)

# directories
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
PROCESSED_RESULTS_DIR = config['processed_results_dir']
# dataset parameters
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']
ALPHAS = config['alphas']
DESIGN_EIG_PERCENTAGES = config['design_eig_percentages']
NUM_PRIORS = config['num_priors']
NUM_DESIGNS_PER_SETTING = config['num_designs_per_setting']

dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM},nsamples={NSAMPLES}.h5'
raw_results_filename = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM},nsamples={NSAMPLES}.h5'
processed_results_filename = f'{PROCESSED_RESULTS_DIR}/errors_d={DATA_DIM},nsamples={NSAMPLES}.h5'


def compute_true_eldr(
    mu_pi: torch.Tensor,
    Sigma_pi: torch.Tensor,
    mu_q: torch.Tensor,
    Sigma_q: torch.Tensor,
    xi: torch.Tensor,
    obs_y: torch.Tensor
) -> float:
    """
    Compute the true ELDR analytically for Gaussian case.

    ELDR = E_q[log p0(θ,y)/p1(θ,y)]

    Where:
    - p0(θ,y) = p(θ)p(y|θ,ξ) is the prior-induced joint
    - p1(θ,y) = q(θ)p(y|ξ) is q(θ) times prior predictive
    - Expectation is under q(θ) at y=obs_y

    ELDR = E_q[log p(θ)] + E_q[log p(obs_y|θ,ξ)] - E_q[log q(θ)] - log p(obs_y|ξ)
    """
    d = mu_pi.shape[0]

    # Precision matrices
    Sigma_pi_inv = torch.linalg.inv(Sigma_pi)
    Sigma_q_inv = torch.linalg.inv(Sigma_q)

    # Term 1: E_q[log p(θ)] where p(θ) = N(μ_π, Σ_π)
    # = -0.5 d log(2π) - 0.5 log|Σ_π| - 0.5 tr(Σ_π⁻¹Σ_q) - 0.5 (μ_q-μ_π)ᵀΣ_π⁻¹(μ_q-μ_π)
    log_det_Sigma_pi = torch.linalg.slogdet(Sigma_pi)[1]
    diff_pi = mu_q - mu_pi
    E_q_log_p_theta = (
        -0.5 * d * np.log(2 * np.pi)
        - 0.5 * log_det_Sigma_pi
        - 0.5 * torch.trace(Sigma_pi_inv @ Sigma_q)
        - 0.5 * diff_pi @ Sigma_pi_inv @ diff_pi
    )

    # Term 2: E_q[log q(θ)] where q(θ) = N(μ_q, Σ_q)
    # = -0.5 d log(2π) - 0.5 log|Σ_q| - 0.5 d (since trace(Σ_q⁻¹Σ_q) = d)
    log_det_Sigma_q = torch.linalg.slogdet(Sigma_q)[1]
    E_q_log_q_theta = (
        -0.5 * d * np.log(2 * np.pi)
        - 0.5 * log_det_Sigma_q
        - 0.5 * d
    )

    # Term 3: E_q[log p(obs_y|θ,ξ)] where p(y|θ,ξ) = N(ξᵀθ, 1)
    # = -0.5 log(2π) - 0.5 [(obs_y - ξᵀμ_q)² + ξᵀΣ_qξ]
    xi_flat = xi.squeeze()
    obs_y_flat = obs_y.squeeze()
    pred_mean_q = xi_flat @ mu_q
    pred_var_q = xi_flat @ Sigma_q @ xi_flat
    E_q_log_p_y_given_theta = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * ((obs_y_flat - pred_mean_q) ** 2 + pred_var_q)
    )

    # Term 4: log p(obs_y|ξ) where p(y|ξ) = N(ξᵀμ_π, ξᵀΣ_πξ + 1)
    prior_pred_mean = xi_flat @ mu_pi
    prior_pred_var = xi_flat @ Sigma_pi @ xi_flat + 1.0
    log_p_y_given_xi = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * torch.log(prior_pred_var)
        - 0.5 * ((obs_y_flat - prior_pred_mean) ** 2) / prior_pred_var
    )

    # ELDR = E_q[log p(θ)] + E_q[log p(y|θ,ξ)] - E_q[log q(θ)] - log p(y|ξ)
    eldr = E_q_log_p_theta + E_q_log_p_y_given_theta - E_q_log_q_theta - log_p_y_given_xi

    return eldr.item()


# Load dataset and compute true ELDRs
with h5py.File(dataset_filename, 'r') as f:
    nrows = f['design_arr'].shape[0]
    true_eldrs = np.zeros(nrows)

    for idx in range(nrows):
        mu_pi = torch.from_numpy(f['prior_mean_arr'][idx])
        Sigma_pi = torch.from_numpy(f['prior_covariance_arr'][idx])
        mu_q = torch.from_numpy(f['mu_q_arr'][idx])
        Sigma_q = torch.from_numpy(f['Sigma_q_arr'][idx])
        xi = torch.from_numpy(f['design_arr'][idx])
        obs_y = torch.from_numpy(f['obs_y_arr'][idx])

        true_eldrs[idx] = compute_true_eldr(mu_pi, Sigma_pi, mu_q, Sigma_q, xi, obs_y)

# Load estimated ELDRs
with h5py.File(raw_results_filename, 'r') as f:
    result_keys = [key for key in f.keys() if key.startswith('est_eldrs_arr_')]
    est_eldrs_by_alg = {key.replace('est_eldrs_arr_', ''): f[key][:] for key in result_keys}

# Compute errors for each algorithm
errors_by_alg = {}
for alg_name, est_eldrs in est_eldrs_by_alg.items():
    errors_by_alg[alg_name] = np.abs(est_eldrs - true_eldrs)

# Reshape errors by (alpha, design_eig_percentage)
# Data is organized as: for each prior, for each design_eig_percentage, for each design, for each alpha
# So the ordering is: [prior0_dep0_design0_alpha0, prior0_dep0_design0_alpha1, ..., prior0_dep0_design0_alphaN,
#                      prior0_dep0_design1_alpha0, ...]
# Shape after reshape: (NUM_PRIORS, len(DESIGN_EIG_PERCENTAGES), NUM_DESIGNS_PER_SETTING, len(ALPHAS))

errors_by_alpha_and_dep = {}
for alg_name, errors in errors_by_alg.items():
    # Reshape to (NUM_PRIORS, len(DESIGN_EIG_PERCENTAGES), NUM_DESIGNS_PER_SETTING, len(ALPHAS))
    errors_reshaped = errors.reshape(
        NUM_PRIORS,
        len(DESIGN_EIG_PERCENTAGES),
        NUM_DESIGNS_PER_SETTING,
        len(ALPHAS)
    )
    errors_by_alpha_and_dep[alg_name] = errors_reshaped

# Also save true ELDRs reshaped the same way
true_eldrs_reshaped = true_eldrs.reshape(
    NUM_PRIORS,
    len(DESIGN_EIG_PERCENTAGES),
    NUM_DESIGNS_PER_SETTING,
    len(ALPHAS)
)

# Compute mean absolute errors by (alpha, design_eig_percentage)
# Average over priors and designs: (len(DESIGN_EIG_PERCENTAGES), len(ALPHAS))
mae_by_alpha_and_dep = {}
for alg_name, errors_reshaped in errors_by_alpha_and_dep.items():
    mae = np.mean(errors_reshaped, axis=(0, 2))  # Average over priors and designs
    mae_by_alpha_and_dep[alg_name] = mae  # (len(DESIGN_EIG_PERCENTAGES), len(ALPHAS))

# Save processed results
os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)
with h5py.File(processed_results_filename, 'w') as f:
    # Save true ELDRs
    f.create_dataset('true_eldrs', data=true_eldrs)
    f.create_dataset('true_eldrs_reshaped', data=true_eldrs_reshaped)

    # Save errors for each algorithm
    for alg_name, errors_reshaped in errors_by_alpha_and_dep.items():
        f.create_dataset(f'errors_{alg_name}', data=errors_reshaped)
        f.create_dataset(f'mae_{alg_name}', data=mae_by_alpha_and_dep[alg_name])

    # Save metadata for plotting
    f.create_dataset('alphas', data=np.array(ALPHAS))
    f.create_dataset('design_eig_percentages', data=np.array(DESIGN_EIG_PERCENTAGES))

print(f"Processed results saved to {processed_results_filename}")
