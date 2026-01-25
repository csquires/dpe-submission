"""
Compare averaged score approach vs individual score approach.

This script runs both approaches and compares their ELDR estimates.
"""

import torch
import numpy as np
from torch.distributions import MultivariateNormal, kl_divergence
from src.eldr_estimation.direct import DirectELDREstimator
from experiments.utils.two_gaussians_kl import create_two_gaussians_kl

# Set random seeds
torch.manual_seed(123)
np.random.seed(123)

DIM = 2
NSAMPLES = 256
KL_DISTANCE = 2.0

# Create data
print("Creating Gaussian data...")
gaussian_pair = create_two_gaussians_kl(DIM, KL_DISTANCE, beta=0.5)
mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']
p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

# Use p0 as base distribution
samples_base = p0.sample((NSAMPLES,)).numpy()
samples_p0 = p0.sample((NSAMPLES,)).numpy()
samples_p1 = p1.sample((NSAMPLES,)).numpy()

# True ELDR
true_eldr = kl_divergence(p0, p1).item()
print(f"\nTrue ELDR (KL divergence): {true_eldr:.4f}")

# Test with averaged score approach (current implementation)
print("\n" + "="*60)
print("AVERAGED SCORE APPROACH")
print("="*60)
estimator_avg = DirectELDREstimator(
    input_dim=DIM,
    k=8.0,
    l=4.0,
    num_epochs=1000,
    batch_size=128,
    verbose=False
)
eldr_avg = estimator_avg.estimate_eldr(samples_base, samples_p0, samples_p1)
print(f"Estimated ELDR: {eldr_avg:.4f}")
print(f"Error: {abs(eldr_avg - true_eldr):.4f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"True ELDR: {true_eldr:.4f}")
print(f"Averaged approach: {eldr_avg:.4f} (error: {abs(eldr_avg - true_eldr):.4f})")
