"""
Debug script to inspect DirectELDREstimator training loop values.

Runs a few iterations with small batch size and prints:
- Time values t
- Sampled x0 and x1 values
- gamma(t) values
- mu(t, x0, x1) values
- Neural network predictions
- True score values
- Loss values
"""

import torch
import numpy as np
from torch.distributions import MultivariateNormal, kl_divergence
from src.eldr_estimation.direct import DirectELDREstimator
from experiments.utils.two_gaussians_kl import create_two_gaussians_kl

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
DIM = 2  # 2D Gaussians
NSAMPLES = 32
KL_DISTANCE = 5.  # Changed from 5 to 2.0
BATCH_SIZE = 4
NUM_DEBUG_ITERATIONS = 3

print("=" * 80)
print("DEBUG: DirectELDREstimator Training Loop")
print("=" * 80)

# === CREATE SYNTHETIC DATA ===
print("\n[1] Creating synthetic Gaussian data...")
gaussian_pair = create_two_gaussians_kl(DIM, KL_DISTANCE, beta=0.5)
mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']
p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

print(f"   Dimension: {DIM}")
print(f"   KL(p0 || p1): {KL_DISTANCE}")
print(f"   mu0: {mu0}")
print(f"   mu1: {mu1}")

# Use p0 as the base distribution p_*
samples_base = p0.sample((NSAMPLES,)).numpy()
samples_p0 = p0.sample((NSAMPLES,)).numpy()
samples_p1 = p1.sample((NSAMPLES,)).numpy()

# === SANITY CHECK: Marginal Distribution Score ===
print("\n" + "=" * 80)
print("SANITY CHECK: Closed-Form Marginal Score")
print("=" * 80)
print("\nThe UNCONDITIONED (marginal) interpolant distribution is:")
print("  p_t(x) = N(x; μ_t, Σ_t)")
print("where:")
print("  μ_t = α(t)*μ₀ + β(t)*μ₁")
print("  Σ_t = α²(t)*Σ₀ + β²(t)*Σ₁ + γ²(t)*I_d")
print("  α(t) = 1-t, β(t) = t, γ(t) = g(t)*h(t)")

def compute_interpolant_coeffs(t: float, k: float) -> dict:
    """
    Compute α, β, γ and their derivatives at time t.

    α(t) = 1 - t
    β(t) = t
    γ(t) = g(t) where g(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t)))
    """
    alpha = 1 - t
    beta = t
    alpha_prime = -1.0
    beta_prime = 1.0

    # g(t) and g'(t)
    exp_kt = np.exp(-k * t)
    exp_k1t = np.exp(-k * (1 - t))
    gamma = (1 - exp_kt) * (1 - exp_k1t)
    gamma_prime = k * exp_kt * (1 - exp_k1t) - k * exp_k1t * (1 - exp_kt)

    return {
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'alpha_prime': alpha_prime, 'beta_prime': beta_prime, 'gamma_prime': gamma_prime
    }

def compute_marginal_score(
    t: float,
    x: torch.Tensor,      # [batch, dim]
    mu0: torch.Tensor,    # [dim]
    Sigma0: torch.Tensor, # [dim, dim]
    mu1: torch.Tensor,    # [dim]
    Sigma1: torch.Tensor, # [dim, dim]
    k: float
) -> torch.Tensor:
    """
    Compute d/dt log p_t(x) for the MARGINAL Gaussian interpolant.

    p_t(x) = N(x; μ_t, Σ_t)
    μ_t = α(t)*μ₀ + β(t)*μ₁
    Σ_t = α²(t)*Σ₀ + β²(t)*Σ₁ + γ²(t)*I

    d/dt log p_t(x) = -1/2 tr(Σ_t⁻¹ Σ'_t)
                     + (x-μ_t)ᵀ Σ_t⁻¹ μ'_t
                     + 1/2 (x-μ_t)ᵀ Σ_t⁻¹ Σ'_t Σ_t⁻¹ (x-μ_t)

    Returns:
        Score values of shape [batch]
    """
    dim = x.shape[-1]
    coeffs = compute_interpolant_coeffs(t, k)
    alpha, beta, gamma = coeffs['alpha'], coeffs['beta'], coeffs['gamma']
    alpha_p, beta_p, gamma_p = coeffs['alpha_prime'], coeffs['beta_prime'], coeffs['gamma_prime']

    # μ_t and μ'_t
    mu_t = alpha * mu0 + beta * mu1  # [dim]
    mu_prime_t = alpha_p * mu0 + beta_p * mu1  # = μ₁ - μ₀

    # Σ_t and Σ'_t
    I_d = torch.eye(dim, dtype=Sigma0.dtype, device=Sigma0.device)
    Sigma_t = alpha**2 * Sigma0 + beta**2 * Sigma1 + gamma**2 * I_d
    Sigma_prime_t = 2*alpha*alpha_p * Sigma0 + 2*beta*beta_p * Sigma1 + 2*gamma*gamma_p * I_d

    # Σ_t⁻¹
    Sigma_t_inv = torch.linalg.inv(Sigma_t)  # [dim, dim]

    # r = x - μ_t
    r = x - mu_t  # [batch, dim]

    # Term 1: -1/2 tr(Σ_t⁻¹ Σ'_t)
    term1 = -0.5 * torch.trace(Sigma_t_inv @ Sigma_prime_t)  # scalar

    # Term 2: (x-μ_t)ᵀ Σ_t⁻¹ μ'_t = r @ Σ_t⁻¹ @ μ'_t
    term2 = r @ Sigma_t_inv @ mu_prime_t  # [batch]

    # Term 3: 1/2 (x-μ_t)ᵀ Σ_t⁻¹ Σ'_t Σ_t⁻¹ (x-μ_t)
    # = 1/2 * r @ Σ_t⁻¹ @ Σ'_t @ Σ_t⁻¹ @ rᵀ
    M = Sigma_t_inv @ Sigma_prime_t @ Sigma_t_inv  # [dim, dim]
    term3 = 0.5 * torch.einsum('bi,ij,bj->b', r, M, r)  # [batch]

    return term1 + term2 + term3  # [batch]

# Print coefficients at several t values
print("\n" + "-" * 60)
print("Interpolant coefficients at various t:")
print("-" * 60)
k_val = 8.0  # Same as in DirectELDREstimator
for t_check in [0.0, 0.25, 0.5, 0.75, 1.0]:
    c = compute_interpolant_coeffs(t_check, k_val)
    print(f"t={t_check:.2f}: α={c['alpha']:.4f}, β={c['beta']:.4f}, γ={c['gamma']:.6f}")
    print(f"        α'={c['alpha_prime']:.4f}, β'={c['beta_prime']:.4f}, γ'={c['gamma_prime']:.6f}")

# Generate samples and compute
N_SAMPLES = 1000
samples_base_check = p0.sample((N_SAMPLES,))  # x ~ p_*

# Compute true KL
true_kl = kl_divergence(p0, p1).item()
print(f"\nTrue KL(p0 || p1): {true_kl:.6f}")

# Integration over t
eps = 0.01
t_values = np.linspace(eps, 1 - eps, 101)
averaged_scores = []

print(f"\nComputing E_{{p_*}}[d/dt log p_t(x)] at {len(t_values)} time points...")
for t_val in t_values:
    scores = compute_marginal_score(t_val, samples_base_check, mu0, Sigma0, mu1, Sigma1, k_val)
    avg_score = scores.mean().item()
    averaged_scores.append(avg_score)

# Numerical integration
integrated_score = np.trapz(averaged_scores, t_values)
another_integrated = sum(averaged_scores)/len(averaged_scores)

print(f"\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"∫₀¹ E_{{p_*}}[d/dt log p_t(x)] dt = {integrated_score:.6f} ({another_integrated:.6f})")
print(f"-∫₀¹ E_{{p_*}}[d/dt log p_t(x)] dt = {-integrated_score:.6f}")
print(f"True ELDR = KL(p0 || p1)           = {true_kl:.6f}")
print(f"Difference: {abs(-integrated_score - true_kl):.6f}")

print("\n" + "=" * 80)
breakpoint()
# === CREATE ESTIMATOR ===
print("\n[2] Creating DirectELDREstimator...")
estimator = DirectELDREstimator(
    input_dim=DIM,
    k=8.0,
    l=4.0,
    num_epochs=5,  # Just a few epochs for debugging
    batch_size=BATCH_SIZE,
    hidden_dim=32,  # Smaller network for debugging
    num_layers=2,
    time_embed_size=16,  # Smaller embedding for debugging
    verbose=False
)

print(f"   Device: {estimator.device}")
print(f"   k (gamma parameter): {estimator.k}")
print(f"   l (importance sampling parameter): {estimator.l}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Time embed size: 16 (output: 32)")

# === MANUALLY RUN DEBUG TRAINING ITERATIONS ===
print("\n[3] Running debug training iterations...")
print("=" * 80)

# Move data to device
samples_base_t = torch.from_numpy(samples_base).float().to(estimator.device)
samples_p0_t = torch.from_numpy(samples_p0).float().to(estimator.device)
samples_p1_t = torch.from_numpy(samples_p1).float().to(estimator.device)

# Compute h(t) coefficients from sample statistics
estimator._compute_h_coefficients(samples_base_t, samples_p0_t, samples_p1_t)
print(f"\nh(t) coefficients for debug estimator:")
print(f"  A (Var(x0)): {estimator._A.item():.6f}")
print(f"  B (Var(x1)): {estimator._B.item():.6f}")
print(f"  C (-2*mean(x0)·mean(x)): {estimator._C.item():.6f}")
print(f"  D (-2*mean(x1)·mean(x)): {estimator._D.item():.6f}")
print(f"  E (Var(x)): {estimator._E.item():.6f}")

n_base = samples_base_t.shape[0]
n_p0 = samples_p0_t.shape[0]
n_p1 = samples_p1_t.shape[0]

# Initialize the network
estimator.score_network._reset_parameters()
estimator.score_network.train()

# Setup optimizer
optimizer = torch.optim.AdamW(
    estimator.score_network.parameters(),
    lr=estimator.learning_rate,
    weight_decay=estimator.weight_decay
)

for iteration in range(NUM_DEBUG_ITERATIONS):
    print(f"\n{'=' * 80}")
    print(f"ITERATION {iteration + 1}/{NUM_DEBUG_ITERATIONS}")
    print(f"{'=' * 80}")

    # Sample t from importance sampling distribution
    t = estimator._sample_t_importance(BATCH_SIZE)

    # Sample x0 from p0, x1 from p1
    # NOTE: We sample one (x0, x1) pair per batch element
    idx_p0 = torch.randint(0, n_p0, (BATCH_SIZE,), device=estimator.device)
    idx_p1 = torch.randint(0, n_p1, (BATCH_SIZE,), device=estimator.device)
    x0 = samples_p0_t[idx_p0]  # [batch, dim]
    x1 = samples_p1_t[idx_p1]  # [batch, dim]

    print(f"\n--- Sampled Values ---")
    print(f"t (time values): {t.cpu().numpy()}")
    print(f"  Shape: {t.shape}")

    # Print x0 samples (from p0)
    print(f"\nx0 (samples from p0):")
    print(f"  Shape: {x0.shape}")
    for i in range(min(BATCH_SIZE, 4)):  # Print all samples (up to 4)
        print(f"  Sample {i}: {x0[i].cpu().numpy()}")

    # Print x1 samples (from p1)
    print(f"\nx1 (samples from p1):")
    print(f"  Shape: {x1.shape}")
    for i in range(min(BATCH_SIZE, 4)):  # Print all samples (up to 4)
        print(f"  Sample {i}: {x1[i].cpu().numpy()}")

    # Compute gamma(t) = g(t) * h(t)
    t_exp = t.unsqueeze(-1)  # [batch, 1]
    g_t = estimator.g(t_exp)  # [batch, 1]
    h_t = estimator.h(t_exp)  # [batch, 1]
    gamma_t = estimator.gamma(t_exp)  # [batch, 1]
    print(f"\ng(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t))):")
    print(f"  Values: {g_t.squeeze(-1).cpu().numpy()}")
    print(f"\nh(t) = A*(1-t)² + B*t² + C*(1-t) + D*t + E:")
    print(f"  Values: {h_t.squeeze(-1).cpu().numpy()}")
    print(f"\ngamma(t) = g(t) * h(t):")
    print(f"  Values: {gamma_t.squeeze(-1).cpu().numpy()}")
    print(f"  Shape: {gamma_t.shape}")

    # Compute mu(t, x0, x1)
    mu_t = estimator.mu(t, x0, x1)
    print(f"\nmu(t, x0, x1) = (1-t)*x0 + t*x1:")
    print(f"  Shape: {mu_t.shape}")
    for i in range(min(BATCH_SIZE, 4)):  # Print all samples (up to 4)
        print(f"  Sample {i}: {mu_t[i].cpu().numpy()}")

    # ===== COMPUTE AVERAGED SCORES OVER ALL BASE SAMPLES =====
    print(f"\n{'=' * 60}")
    print("COMPUTING AVERAGED SCORES OVER BASE DISTRIBUTION")
    print(f"{'=' * 60}")
    print(f"\nFor each (t, x0, x1) tuple, we compute the score for ALL {n_base} base samples")
    print(f"and average them to get the expected score E_{{p_*}}[s(x,t;x0,x1)].\n")

    # Compute average scores (matching the actual training implementation)
    avg_scores = []
    for i in range(BATCH_SIZE):
        # Expand t, x0, x1 to match all base samples
        t_i = t[i].expand(n_base)  # [n_base]
        x0_i = x0[i].unsqueeze(0).expand(n_base, -1)  # [n_base, dim]
        x1_i = x1[i].unsqueeze(0).expand(n_base, -1)  # [n_base, dim]

        # Compute score for all base samples
        scores_i = estimator.score(t_i, samples_base_t, x0_i, x1_i)  # [n_base]

        print(f"Batch element {i}:")
        print(f"  t={t[i].item():.4f}, x0={x0[i].cpu().numpy()}, x1={x1[i].cpu().numpy()}")
        print(f"  Individual scores for all {n_base} base samples:")
        print(f"    Min: {scores_i.min().item():+.6f}, Max: {scores_i.max().item():+.6f}, "
              f"Mean: {scores_i.mean().item():+.6f}, Std: {scores_i.std().item():.6f}")

        # Average over all base samples
        avg_scores.append(scores_i.mean())

    avg_scores = torch.stack(avg_scores)  # [batch_size]
    print(f"\nAveraged scores (targets for neural network):")
    print(f"  Values: {avg_scores.cpu().numpy()}")
    print(f"  Shape: {avg_scores.shape}")

    print(f"\n{'=' * 60}")

    # ===== DETAILED GROUND TRUTH SCORE CALCULATION (for first base sample, first batch element) =====
    print(f"\n{'=' * 60}")
    print("DETAILED GROUND TRUTH SCORE CALCULATION")
    print(f"{'=' * 60}")
    print(f"\nShowing detailed calculation for first batch element (i=0) and first base sample.")
    print(f"This is just one of the {n_base} samples that get averaged.\n")

    # Use first batch element
    idx = 0
    t_single = t[idx:idx+1]  # [1]
    x0_single = x0[idx:idx+1]  # [1, dim]
    x1_single = x1[idx:idx+1]  # [1, dim]
    x_single = samples_base_t[0:1]  # [1, dim] - first base sample

    # Compute intermediate values for score calculation
    dim = x0.shape[-1]
    t_exp_score = t_single.unsqueeze(-1)  # [1, 1]

    # g(t), h(t), gamma(t) = g(t)*h(t), and their derivatives
    g_t_score = estimator.g(t_exp_score)  # [1, 1]
    dgdt_score = estimator.dgdt(t_exp_score)  # [1, 1]
    h_t_score = estimator.h(t_exp_score)  # [1, 1]
    dhdt_score = estimator.dhdt(t_exp_score)  # [1, 1]
    gamma_t_score = estimator.gamma(t_exp_score)  # [1, 1]
    gamma_prime_t_score = estimator.dgammadt(t_exp_score)  # [1, 1]

    print(f"t = {t_single[0].item():.4f}")
    print(f"x (first base sample) = {x_single[0].cpu().numpy()}")
    print(f"x0 = {x0_single[0].cpu().numpy()}")
    print(f"x1 = {x1_single[0].cpu().numpy()}")

    print(f"\ng(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t))):")
    print(f"  Value: {g_t_score[0, 0].item():.6f}")

    print(f"\ng'(t) = k*exp(-k*t)*(1-exp(-k*(1-t))) - k*exp(-k*(1-t))*(1-exp(-k*t)):")
    print(f"  Value: {dgdt_score[0, 0].item():.6f}")

    print(f"\nh(t) = A*(1-t)² + B*t² + C*(1-t) + D*t + E:")
    print(f"  Value: {h_t_score[0, 0].item():.6f}")

    print(f"\nh'(t) = -2*A*(1-t) + 2*B*t - C + D:")
    print(f"  Value: {dhdt_score[0, 0].item():.6f}")

    print(f"\ngamma(t) = g(t) * h(t):")
    print(f"  Value: {gamma_t_score[0, 0].item():.6f}")

    print(f"\ngamma'(t) = g'(t)*h(t) + g(t)*h'(t) (product rule):")
    print(f"  Value: {gamma_prime_t_score[0, 0].item():.6f}")

    # mu(t) and mu'(t)
    mu_t_score = estimator.mu(t_single, x0_single, x1_single)  # [1, dim]
    mu_prime_score = estimator.dmudt(x0_single, x1_single)  # [1, dim]

    print(f"\nmu(t) = (1-t)*x0 + t*x1:")
    print(f"  Value: {mu_t_score[0].cpu().numpy()}")

    print(f"\nmu'(t) = x1 - x0 (constant in t):")
    print(f"  Value: {mu_prime_score[0].cpu().numpy()}")

    # r = x - mu(t) (NOT divided by gamma)
    r = x_single - mu_t_score  # [1, dim]
    print(f"\nr = x - mu(t):")
    print(f"  Value: {r[0].cpu().numpy()}")

    # ||r||^2
    r_norm_sq = (r ** 2).sum(dim=-1, keepdim=True)  # [1, 1]
    print(f"\n||r||^2:")
    print(f"  Value: {r_norm_sq[0, 0].item():.6f}")

    # r^T * mu'(t)
    r_dot_mu_prime = (r * mu_prime_score).sum(dim=-1, keepdim=True)  # [1, 1]
    print(f"\nr^T * mu'(t):")
    print(f"  Value: {r_dot_mu_prime[0, 0].item():.6f}")

    # gamma'(t) / gamma(t)
    gamma_ratio = gamma_prime_t_score / gamma_t_score
    print(f"\ngamma'(t) / gamma(t):")
    print(f"  Value: {gamma_ratio[0, 0].item():.6f}")

    # gamma(t)^2
    gamma_sq = gamma_t_score ** 2
    print(f"\ngamma(t)^2:")
    print(f"  Value: {gamma_sq[0, 0].item():.6f}")

    # gamma(t)^3
    gamma_cubed = gamma_t_score ** 3
    print(f"\ngamma(t)^3:")
    print(f"  Value: {gamma_cubed[0, 0].item():.6f}")

    # Compute the three terms of the score formula
    # s = -dim*gamma'/gamma + ||r||^2*gamma'/gamma^3 + r^T*mu'/gamma^2
    term1 = -dim * gamma_ratio
    print(f"\nTerm 1: -dim * gamma'/gamma where dim={dim}:")
    print(f"  Value: {term1[0, 0].item():.6f}")

    term2 = r_norm_sq * (gamma_prime_t_score / gamma_cubed)
    print(f"\nTerm 2: ||r||^2 * gamma'/gamma^3:")
    print(f"  Value: {term2[0, 0].item():.6f}")

    term3 = r_dot_mu_prime / gamma_sq
    print(f"\nTerm 3: r^T*mu' / gamma^2:")
    print(f"  Value: {term3[0, 0].item():.6f}")

    # Final score: s = -dim*gamma'/gamma + ||r||^2*gamma'/gamma^3 + r^T*mu'/gamma^2
    score_manual = (term1 + term2 + term3).squeeze(-1)
    print(f"\nFinal score s = -dim*gamma'/gamma + ||r||^2*gamma'/gamma^3 + r^T*mu'/gamma^2:")
    print(f"  Value: {score_manual[0].item():.6f}")

    print(f"\n{'=' * 60}")

    # Forward pass through neural network
    predictions = estimator.score_network(t)  # [batch]
    print(f"\nNeural network predictions:")
    print(f"  Values: {predictions.detach().cpu().numpy()}")
    print(f"  Shape: {predictions.shape}")

    # MSE loss per sample (comparing to averaged scores)
    mse_per_sample = (predictions - avg_scores.detach()) ** 2
    print(f"\nMSE per sample (compared to averaged scores):")
    print(f"  Values: {mse_per_sample.detach().cpu().numpy()}")

    # Show comparison
    print(f"\nComparison:")
    for i in range(BATCH_SIZE):
        print(f"  Batch {i}: NN prediction = {predictions[i].item():+.6f}, "
              f"Averaged target = {avg_scores[i].item():+.6f}, "
              f"MSE = {mse_per_sample[i].item():.6f}")

    # Compute importance sampling weight: 1 / (f(t) * g(t))
    f_t = estimator._f_torch(t)  # [batch]
    weights = 1.0 / (f_t * g_t.squeeze(-1))  # [batch]
    print(f"\nf(t) (scaled importance distribution):")
    print(f"  Values: {f_t.cpu().numpy()}")
    print(f"\nWeights (1 / (f(t) * g(t))):")
    print(f"  Values: {weights.cpu().numpy()}")

    # Weighted loss
    weighted_losses = mse_per_sample * weights
    print(f"\nWeighted loss per sample (MSE / (f(t) * g(t))):")
    print(f"  Values: {weighted_losses.detach().cpu().numpy()}")

    loss = weighted_losses.mean()
    print(f"\nTotal loss (mean of weighted losses):")
    print(f"  Value: {loss.item():.6f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Print gradient norms
    total_grad_norm = 0.0
    for name, param in estimator.score_network.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
    total_grad_norm = np.sqrt(total_grad_norm)
    print(f"\nGradient norm before clipping: {total_grad_norm:.6f}")

    torch.nn.utils.clip_grad_norm_(estimator.score_network.parameters(), max_norm=10.0)
    optimizer.step()

    print(f"\n{'=' * 80}")

print("\n" + "=" * 80)
print("Debug session complete!")
print("=" * 80)
