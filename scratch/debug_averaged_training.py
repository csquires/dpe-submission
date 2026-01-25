"""
Debug script to inspect averaged score training.
"""

import torch
import numpy as np
from torch.distributions import MultivariateNormal
from src.eldr_estimation.direct import DirectELDREstimator
from experiments.utils.two_gaussians_kl import create_two_gaussians_kl

torch.manual_seed(42)
np.random.seed(42)

DIM = 2
NSAMPLES = 100
KL_DISTANCE = 2.0

# Create data
gaussian_pair = create_two_gaussians_kl(DIM, KL_DISTANCE, beta=0.5)
mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']
p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

samples_base = p0.sample((NSAMPLES,)).numpy()
samples_p0 = p0.sample((NSAMPLES,)).numpy()
samples_p1 = p1.sample((NSAMPLES,)).numpy()

# Create estimator
estimator = DirectELDREstimator(
    input_dim=DIM,
    k=8.0,
    l=4.0,
    batch_size=4,
    device='cpu'
)

# Convert to tensors
samples_base_t = torch.from_numpy(samples_base).float()
samples_p0_t = torch.from_numpy(samples_p0).float()
samples_p1_t = torch.from_numpy(samples_p1).float()

n_base = samples_base_t.shape[0]
n_p0 = samples_p0_t.shape[0]
n_p1 = samples_p1_t.shape[0]

print("="*80)
print("DEBUG: Averaged Score Training")
print("="*80)

# Manually run one training iteration
print("\n[1] Sampling t values...")
t = estimator._sample_t_importance(4)
print(f"t = {t.numpy()}")

print("\n[2] Sampling x0, x1...")
idx_p0 = torch.randint(0, n_p0, (4,))
idx_p1 = torch.randint(0, n_p1, (4,))
x0 = samples_p0_t[idx_p0]
x1 = samples_p1_t[idx_p1]
print(f"x0 shape: {x0.shape}")
print(f"x1 shape: {x1.shape}")

print("\n[3] Computing averaged scores...")
avg_scores = []
for i in range(4):
    # Expand to all base samples
    t_i = t[i].expand(n_base)
    x0_i = x0[i].unsqueeze(0).expand(n_base, -1)
    x1_i = x1[i].unsqueeze(0).expand(n_base, -1)

    # Compute scores
    scores_i = estimator.score(t_i, samples_base_t, x0_i, x1_i)

    print(f"\n  t={t[i].item():.4f}:")
    print(f"    Individual scores: min={scores_i.min().item():.4f}, "
          f"max={scores_i.max().item():.4f}, "
          f"mean={scores_i.mean().item():.4f}, "
          f"std={scores_i.std().item():.4f}")

    avg_scores.append(scores_i.mean())

avg_scores = torch.stack(avg_scores)
print(f"\n[4] Averaged scores for this batch:")
print(f"  {avg_scores.numpy()}")

# Now check what these scores represent
print("\n[5] Interpretation:")
print("  Each averaged score is E_x[s(x, t; x0, x1)] for a specific (t, x0, x1)")
print("  The network should learn to predict these averaged values from t alone")
print("  But different (x0, x1) pairs will give different averages for the same t!")

# Let's verify this by checking multiple (x0, x1) pairs at the same t
print("\n[6] Testing variance at fixed t=0.5...")
t_fixed = torch.tensor([0.5])
n_trials = 10
averaged_scores_at_t05 = []

for trial in range(n_trials):
    # Sample different x0, x1
    idx = torch.randint(0, n_p0, (1,))
    x0_trial = samples_p0_t[idx]
    x1_trial = samples_p1_t[idx]

    # Compute averaged score
    t_expanded = t_fixed.expand(n_base)
    x0_expanded = x0_trial.expand(n_base, -1)
    x1_expanded = x1_trial.expand(n_base, -1)
    scores = estimator.score(t_expanded, samples_base_t, x0_expanded, x1_expanded)
    avg_score = scores.mean().item()
    averaged_scores_at_t05.append(avg_score)

print(f"  Averaged scores at t=0.5 for {n_trials} different (x0,x1) pairs:")
print(f"  Values: {averaged_scores_at_t05}")
print(f"  Mean: {np.mean(averaged_scores_at_t05):.4f}")
print(f"  Std: {np.std(averaged_scores_at_t05):.4f}")
print(f"  Range: [{np.min(averaged_scores_at_t05):.4f}, {np.max(averaged_scores_at_t05):.4f}]")

print("\n" + "="*80)
print("CONCLUSION:")
print("The averaged scores still vary with (x0, x1), but the network only sees t!")
print("This creates a noisy target that may be hard to learn.")
print("We might need to also average over (x0, x1) to get a target that depends only on t.")
print("="*80)
