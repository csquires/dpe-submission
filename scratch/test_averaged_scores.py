"""
Test script to verify averaged score computation.

This script shows the difference between individual scores and averaged scores.
"""

import torch
import numpy as np
from torch.distributions import MultivariateNormal
from src.eldr_estimation.direct import DirectELDREstimator
from experiments.utils.two_gaussians_kl import create_two_gaussians_kl

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
DIM = 2
NSAMPLES = 100
KL_DISTANCE = 2.0
BATCH_SIZE = 4

print("=" * 80)
print("TEST: Averaged Scores vs Individual Scores")
print("=" * 80)

# === CREATE SYNTHETIC DATA ===
print("\n[1] Creating synthetic Gaussian data...")
gaussian_pair = create_two_gaussians_kl(DIM, KL_DISTANCE, beta=0.5)
mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']
p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

# Use p0 as the base distribution p_*
samples_base = p0.sample((NSAMPLES,))
samples_p0 = p0.sample((NSAMPLES,))
samples_p1 = p1.sample((NSAMPLES,))

# === CREATE ESTIMATOR ===
print("\n[2] Creating DirectELDREstimator...")
estimator = DirectELDREstimator(
    input_dim=DIM,
    k=8.0,
    l=4.0,
    device='cpu'  # Use CPU for easier inspection
)

# === TEST SCORE COMPUTATION ===
print("\n[3] Testing score computation...")

# Sample one set of t, x0, x1
t = torch.tensor([0.5])  # Middle time point
x0 = samples_p0[0:1]  # [1, dim]
x1 = samples_p1[0:1]  # [1, dim]

print(f"\nFixed parameters:")
print(f"  t = {t[0].item():.4f}")
print(f"  x0 = {x0[0].numpy()}")
print(f"  x1 = {x1[0].numpy()}")

# Compute scores for all base samples
n_base = samples_base.shape[0]
t_expanded = t.expand(n_base)
x0_expanded = x0.expand(n_base, -1)
x1_expanded = x1.expand(n_base, -1)

scores = estimator.score(t_expanded, samples_base, x0_expanded, x1_expanded)

print(f"\n[4] Score distribution across {n_base} base samples:")
print(f"  Min: {scores.min().item():.4f}")
print(f"  Max: {scores.max().item():.4f}")
print(f"  Mean: {scores.mean().item():.4f}")
print(f"  Std: {scores.std().item():.4f}")
print(f"  Median: {scores.median().item():.4f}")

print(f"\n  First 10 scores: {scores[:10].numpy()}")

print(f"\n[5] Averaged score (what the network should learn):")
avg_score = scores.mean()
print(f"  E_x[s(x, t; x0, x1)] = {avg_score.item():.4f}")

# Now compute the true ELDR for this (t, x0, x1) configuration
# ELDR = E_{p_*}[log p0(x) - log p1(x)]
true_eldr = (p0.log_prob(samples_base) - p1.log_prob(samples_base)).mean()
print(f"\n[6] True ELDR for comparison:")
print(f"  E_p*[log p0(x) - log p1(x)] = {true_eldr.item():.4f}")

# The integral of the averaged score over t should give the ELDR
print(f"\n" + "=" * 80)
print("INTERPRETATION:")
print("=" * 80)
print(f"The network learns S_*(t) = E_x[s(x, t; x0, x1)]")
print(f"At t={t[0].item():.2f}, for this particular (x0, x1) pair:")
print(f"  - Individual scores range from {scores.min().item():.2f} to {scores.max().item():.2f}")
print(f"  - The averaged score is: {avg_score.item():.4f}")
print(f"\nThe integral of S_*(t) from 0 to 1 should approximate the ELDR.")
print("=" * 80)
