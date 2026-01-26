"""
Debug script to inspect DirectELDREstimator3 training loop values.

Runs a few iterations with small batch size and prints:
- Time values t
- Sampled x0 and x1 values
- g(t), g'(t), gamma(t) values
- mu(t, x0, x1) values
- Noiser values (averaged over base samples)
- Neural network predictions
- Loss values

Key differences from debug_direct_eldr.py (for DirectELDREstimator):
- No h(t) function (direct3.py uses simpler g(t) as gamma)
- Uniform t sampling instead of importance sampling
- Different estimand: d^T * mu' * gamma + ||d||^2
- Variance normalization via v(t)
- Loss weighting: v(t)^2 / (g(t) + eps)^6
- Integration: -dim * g'/g + v(t) * NN(t) / gamma^3
"""

import torch
import numpy as np
from torch.distributions import MultivariateNormal, kl_divergence
from src.eldr_estimation.direct3 import DirectELDREstimator3
from experiments.utils.two_gaussians_kl import create_two_gaussians_kl

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
DIM = 2  # 2D Gaussians
NSAMPLES = 32
KL_DISTANCE = 5.
BATCH_SIZE = 4
NUM_DEBUG_ITERATIONS = 3

print("=" * 80)
print("DEBUG: DirectELDREstimator3 Training Loop")
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
print("  p_t(x) = N(x; mu_t, Sigma_t)")
print("where:")
print("  mu_t = (1-t)*mu0 + t*mu1")
print("  Sigma_t = (1-t)^2*Sigma0 + t^2*Sigma1 + gamma^2(t)*I_d")
print("  gamma(t) = g(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t)))")

def compute_interpolant_coeffs(t: float, k: float) -> dict:
    """
    Compute alpha, beta, gamma and their derivatives at time t.

    alpha(t) = 1 - t
    beta(t) = t
    gamma(t) = g(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t)))
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

    p_t(x) = N(x; mu_t, Sigma_t)
    mu_t = alpha(t)*mu0 + beta(t)*mu1
    Sigma_t = alpha^2(t)*Sigma0 + beta^2(t)*Sigma1 + gamma^2(t)*I

    d/dt log p_t(x) = -1/2 tr(Sigma_t^-1 Sigma'_t)
                     + (x-mu_t)^T Sigma_t^-1 mu'_t
                     + 1/2 (x-mu_t)^T Sigma_t^-1 Sigma'_t Sigma_t^-1 (x-mu_t)

    Returns:
        Score values of shape [batch]
    """
    dim = x.shape[-1]
    coeffs = compute_interpolant_coeffs(t, k)
    alpha, beta, gamma = coeffs['alpha'], coeffs['beta'], coeffs['gamma']
    alpha_p, beta_p, gamma_p = coeffs['alpha_prime'], coeffs['beta_prime'], coeffs['gamma_prime']

    # mu_t and mu'_t
    mu_t = alpha * mu0 + beta * mu1  # [dim]
    mu_prime_t = alpha_p * mu0 + beta_p * mu1  # = mu1 - mu0

    # Sigma_t and Sigma'_t
    I_d = torch.eye(dim, dtype=Sigma0.dtype, device=Sigma0.device)
    Sigma_t = alpha**2 * Sigma0 + beta**2 * Sigma1 + gamma**2 * I_d
    Sigma_prime_t = 2*alpha*alpha_p * Sigma0 + 2*beta*beta_p * Sigma1 + 2*gamma*gamma_p * I_d

    # Sigma_t^-1
    Sigma_t_inv = torch.linalg.inv(Sigma_t)  # [dim, dim]

    # r = x - mu_t
    r = x - mu_t  # [batch, dim]

    # Term 1: -1/2 tr(Sigma_t^-1 Sigma'_t)
    term1 = -0.5 * torch.trace(Sigma_t_inv @ Sigma_prime_t)  # scalar

    # Term 2: (x-mu_t)^T Sigma_t^-1 mu'_t = r @ Sigma_t^-1 @ mu'_t
    term2 = r @ Sigma_t_inv @ mu_prime_t  # [batch]

    # Term 3: 1/2 (x-mu_t)^T Sigma_t^-1 Sigma'_t Sigma_t^-1 (x-mu_t)
    # = 1/2 * r @ Sigma_t^-1 @ Sigma'_t @ Sigma_t^-1 @ r^T
    M = Sigma_t_inv @ Sigma_prime_t @ Sigma_t_inv  # [dim, dim]
    term3 = 0.5 * torch.einsum('bi,ij,bj->b', r, M, r)  # [batch]

    return term1 + term2 + term3  # [batch]

# Print coefficients at several t values
print("\n" + "-" * 60)
print("Interpolant coefficients at various t:")
print("-" * 60)
k_val = 8.0  # Same as in DirectELDREstimator3 (when k=8.0)
for t_check in [0.0, 0.25, 0.5, 0.75, 1.0]:
    c = compute_interpolant_coeffs(t_check, k_val)
    print(f"t={t_check:.2f}: alpha={c['alpha']:.4f}, beta={c['beta']:.4f}, gamma={c['gamma']:.6f}")
    print(f"        alpha'={c['alpha_prime']:.4f}, beta'={c['beta_prime']:.4f}, gamma'={c['gamma_prime']:.6f}")

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
averaged_decentered_scores = []
centering_terms = []

print(f"\nComputing E_{{p_*}}[d/dt log p_t(x)] at {len(t_values)} time points...")
print(f"Also computing decentered version: marginal_score + dim * gamma'/gamma\n")

for t_val in t_values:
    scores = compute_marginal_score(t_val, samples_base_check, mu0, Sigma0, mu1, Sigma1, k_val)
    avg_score = scores.mean().item()
    averaged_scores.append(avg_score)

    # Compute centering term: dim * gamma'/gamma
    coeffs = compute_interpolant_coeffs(t_val, k_val)
    gamma = coeffs['gamma']
    gamma_prime = coeffs['gamma_prime']
    centering = DIM * gamma_prime / gamma
    centering_terms.append(centering)

    # Decentered = marginal_score + dim * gamma'/gamma
    decentered_score = avg_score + centering
    averaged_decentered_scores.append(decentered_score)

# Numerical integration
integrated_score = np.trapz(averaged_scores, t_values)
integrated_decentered = np.trapz(averaged_decentered_scores, t_values)
integrated_centering = np.trapz(centering_terms, t_values)

print(f"\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\n--- Marginal Score (full score) ---")
print(f"integral E[d/dt log p_t(x)] dt = {integrated_score:.6f}")
print(f"-integral E[d/dt log p_t(x)] dt = {-integrated_score:.6f}")

print(f"\n--- Centering Term ---")
print(f"integral dim*gamma'/gamma dt = {integrated_centering:.6f}")
print(f"  (This should be close to 0 since gamma(eps) ~ gamma(1-eps))")

print(f"\n--- Decentered Marginal Score ---")
print(f"  decentered = marginal_score + dim*gamma'/gamma")
print(f"integral E[decentered] dt = {integrated_decentered:.6f}")
print(f"-integral E[decentered] dt = {-integrated_decentered:.6f}")

print(f"\n--- Ground Truth ---")
print(f"True ELDR = KL(p0 || p1) = {true_kl:.6f}")

print(f"\n--- Comparison ---")
print(f"|-integral(marginal)| - KL   = {abs(-integrated_score - true_kl):.6f}")
print(f"|-integral(decentered)| - KL = {abs(-integrated_decentered - true_kl):.6f}")

print("\n" + "=" * 80)
breakpoint()

# === CREATE ESTIMATOR ===
print("\n[2] Creating DirectELDREstimator3...")
estimator = DirectELDREstimator3(
    input_dim=DIM,
    k=8.0,
    eps=0.1,  # Boundary epsilon for t sampling
    num_epochs=5,  # Just a few epochs for debugging
    batch_size=BATCH_SIZE,
    hidden_dim=32,  # Smaller network for debugging
    num_layers=2,
    time_embed_size=16,  # Smaller embedding for debugging
    verbose=False
)

print(f"   Device: {estimator.device}")
print(f"   k (gamma parameter): {estimator.k}")
print(f"   eps (boundary epsilon): {estimator.eps}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Time embed size: 16 (output: 32)")

# === MANUALLY RUN DEBUG TRAINING ITERATIONS ===
print("\n[3] Running debug training iterations...")
print("=" * 80)

# Move data to device
samples_base_t = torch.from_numpy(samples_base).float().to(estimator.device)
samples_p0_t = torch.from_numpy(samples_p0).float().to(estimator.device)
samples_p1_t = torch.from_numpy(samples_p1).float().to(estimator.device)

# === COMPUTE SAMPLE STATISTICS ===
print("\n[4] Computing sample statistics...")
estimator._compute_statistics(samples_base_t, samples_p0_t, samples_p1_t)
print(f"\nSample statistics:")
print(f"  _mean_x (mean of base samples): {estimator._mean_x.cpu().numpy()}")
print(f"  _E (sum of Var(x) across dims): {estimator._E.item():.6f}")

# === COMPUTE v(t) GRID ===
print("\n[5] Computing v(t) grid...")
n_v_grid = 100
estimator._v_grid_t = torch.linspace(estimator.eps, 1 - estimator.eps, n_v_grid, device=estimator.device)
estimator._v_grid_vals = []
with torch.no_grad():
    for t_val in estimator._v_grid_t:
        v_t = estimator._compute_v(t_val, samples_p0_t, samples_p1_t)
        estimator._v_grid_vals.append(v_t)
estimator._v_grid_vals = torch.stack(estimator._v_grid_vals)

print(f"  v(t) grid range: [{estimator._v_grid_vals.min().item():.6f}, {estimator._v_grid_vals.max().item():.6f}]")
print(f"  v(t) at t=eps: {estimator._v_grid_vals[0].item():.6f}")
print(f"  v(t) at t=0.5: {estimator._v_grid_vals[n_v_grid//2].item():.6f}")
print(f"  v(t) at t=1-eps: {estimator._v_grid_vals[-1].item():.6f}")

n_base = samples_base_t.shape[0]
n_p0 = samples_p0_t.shape[0]
n_p1 = samples_p1_t.shape[0]

# Initialize the network
estimator.noiser_network._reset_parameters()
estimator.noiser_network.train()

# Setup optimizer
optimizer = torch.optim.AdamW(
    estimator.noiser_network.parameters(),
    lr=estimator.learning_rate,
    weight_decay=estimator.weight_decay
)

for iteration in range(NUM_DEBUG_ITERATIONS):
    print(f"\n{'=' * 80}")
    print(f"ITERATION {iteration + 1}/{NUM_DEBUG_ITERATIONS}")
    print(f"{'=' * 80}")

    # Sample t uniformly from [eps, 1-eps]
    t = estimator._sample_t_uniform(BATCH_SIZE)

    # Sample x0 from p0, x1 from p1
    idx_p0 = torch.randint(0, n_p0, (BATCH_SIZE,), device=estimator.device)
    idx_p1 = torch.randint(0, n_p1, (BATCH_SIZE,), device=estimator.device)
    x0 = samples_p0_t[idx_p0]  # [batch, dim]
    x1 = samples_p1_t[idx_p1]  # [batch, dim]

    print(f"\n--- Sampled Values ---")
    print(f"t (time values, uniform on [eps, 1-eps]): {t.cpu().numpy()}")
    print(f"  Shape: {t.shape}")

    # Print x0 samples (from p0)
    print(f"\nx0 (samples from p0):")
    print(f"  Shape: {x0.shape}")
    for i in range(min(BATCH_SIZE, 4)):
        print(f"  Sample {i}: {x0[i].cpu().numpy()}")

    # Print x1 samples (from p1)
    print(f"\nx1 (samples from p1):")
    print(f"  Shape: {x1.shape}")
    for i in range(min(BATCH_SIZE, 4)):
        print(f"  Sample {i}: {x1[i].cpu().numpy()}")

    # Compute g(t), g'(t), gamma(t) = g(t)
    t_exp = t.unsqueeze(-1)  # [batch, 1]
    g_t = estimator.g(t)  # [batch]
    g_prime_t = estimator.dgdt(t)  # [batch]

    print(f"\ng(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t))):")
    print(f"  Values: {g_t.cpu().numpy()}")
    print(f"\ng'(t) = k*exp(-k*t)*(1-exp(-k*(1-t))) - k*exp(-k*(1-t))*(1-exp(-k*t)):")
    print(f"  Values: {g_prime_t.cpu().numpy()}")
    print(f"\ngamma(t) = g(t) (no h(t) in direct3):")
    print(f"  Values: {g_t.cpu().numpy()}")
    print(f"  Shape: {g_t.shape}")

    # Compute mu(t, x0, x1)
    mu_t = estimator.mu(t, x0, x1)
    print(f"\nmu(t, x0, x1) = (1-t)*x0 + t*x1:")
    print(f"  Shape: {mu_t.shape}")
    for i in range(min(BATCH_SIZE, 4)):
        print(f"  Sample {i}: {mu_t[i].cpu().numpy()}")

    # ===== COMPUTE AVERAGED NOISER OVER ALL BASE SAMPLES =====
    print(f"\n{'=' * 60}")
    print("COMPUTING AVERAGED NOISER OVER BASE DISTRIBUTION")
    print(f"{'=' * 60}")
    print(f"\nFor each (t, x0, x1) tuple, we compute the noiser for ALL {n_base} base samples")
    print(f"and average them to get E_{{p_*}}[noiser(x,t;x0,x1)].\n")
    print(f"Noiser formula: d^T * mu' * gamma + ||d||^2")
    print(f"where d = x - mu(t), mu' = x1 - x0, gamma = g(t)\n")

    # Compute average noiser (matching the actual training implementation)
    avg_noisers = []
    for i in range(BATCH_SIZE):
        # Expand t, x0, x1 to match all base samples
        t_i = t[i].expand(n_base)  # [n_base]
        x0_i = x0[i].unsqueeze(0).expand(n_base, -1)  # [n_base, dim]
        x1_i = x1[i].unsqueeze(0).expand(n_base, -1)  # [n_base, dim]

        # Compute noiser for all base samples
        noisers_i = estimator.noiser(t_i, samples_base_t, x0_i, x1_i)  # [n_base]

        print(f"Batch element {i}:")
        print(f"  t={t[i].item():.4f}, x0={x0[i].cpu().numpy()}, x1={x1[i].cpu().numpy()}")
        print(f"  Individual noiser for all {n_base} base samples:")
        print(f"    Min: {noisers_i.min().item():+.6f}, Max: {noisers_i.max().item():+.6f}, "
              f"Mean: {noisers_i.mean().item():+.6f}, Std: {noisers_i.std().item():.6f}")

        # Average over all base samples
        avg_noisers.append(noisers_i.mean())

    avg_noisers = torch.stack(avg_noisers)  # [batch_size]
    print(f"\nAveraged noiser (targets before variance normalization):")
    print(f"  Values: {avg_noisers.cpu().numpy()}")
    print(f"  Shape: {avg_noisers.shape}")

    print(f"\n{'=' * 60}")

    # ===== DETAILED GROUND TRUTH CALCULATION (for first base sample, first batch element) =====
    print(f"\n{'=' * 60}")
    print("DETAILED GROUND TRUTH NOISER CALCULATION")
    print(f"{'=' * 60}")
    print(f"\nShowing detailed calculation for first batch element (i=0) and first base sample.")
    print(f"This is just one of the {n_base} samples that get averaged.\n")

    # Use first batch element
    idx = 0
    t_single = t[idx]  # scalar
    x0_single = x0[idx:idx+1]  # [1, dim]
    x1_single = x1[idx:idx+1]  # [1, dim]
    x_single = samples_base_t[0:1]  # [1, dim] - first base sample

    # Compute intermediate values for noiser calculation
    dim = x0.shape[-1]

    # g(t), g'(t), gamma(t) = g(t)
    g_t_single = estimator.g(t_single.unsqueeze(0))  # [1]
    g_prime_t_single = estimator.dgdt(t_single.unsqueeze(0))  # [1]
    gamma_t_single = g_t_single  # In direct3, gamma = g (no h)

    print(f"t = {t_single.item():.4f}")
    print(f"x (first base sample) = {x_single[0].cpu().numpy()}")
    print(f"x0 = {x0_single[0].cpu().numpy()}")
    print(f"x1 = {x1_single[0].cpu().numpy()}")

    print(f"\ng(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t))):")
    print(f"  Value: {g_t_single[0].item():.6f}")

    print(f"\ng'(t) = k*exp(-k*t)*(1-exp(-k*(1-t))) - k*exp(-k*(1-t))*(1-exp(-k*t)):")
    print(f"  Value: {g_prime_t_single[0].item():.6f}")

    print(f"\ngamma(t) = g(t) (no h(t) in direct3):")
    print(f"  Value: {gamma_t_single[0].item():.6f}")

    # mu(t) and mu'(t)
    mu_t_single = estimator.mu(t_single.unsqueeze(0), x0_single, x1_single)  # [1, dim]
    mu_prime_single = estimator.dmudt(x0_single, x1_single)  # [1, dim]

    print(f"\nmu(t) = (1-t)*x0 + t*x1:")
    print(f"  Value: {mu_t_single[0].cpu().numpy()}")

    print(f"\nmu'(t) = x1 - x0 (constant in t):")
    print(f"  Value: {mu_prime_single[0].cpu().numpy()}")

    # d = x - mu(t)
    d = x_single - mu_t_single  # [1, dim]
    print(f"\nd = x - mu(t):")
    print(f"  Value: {d[0].cpu().numpy()}")

    # ||d||^2
    d_norm_sq = (d ** 2).sum(dim=-1)  # [1]
    print(f"\n||d||^2:")
    print(f"  Value: {d_norm_sq[0].item():.6f}")

    # d^T * mu'(t)
    d_dot_mu_prime = (d * mu_prime_single).sum(dim=-1)  # [1]
    print(f"\nd^T * mu'(t):")
    print(f"  Value: {d_dot_mu_prime[0].item():.6f}")

    # gamma(t)
    print(f"\ngamma(t):")
    print(f"  Value: {gamma_t_single[0].item():.6f}")

    # === INTERMEDIATE STEP: Conditional estimand (what the network learns) ===
    print(f"\n{'=' * 60}")
    print("INTERMEDIATE STEP: Conditional Estimand (without centering term)")
    print(f"{'=' * 60}")

    # noiser = d^T * mu' * gamma + ||d||^2
    noiser_single = d_dot_mu_prime * gamma_t_single + d_norm_sq
    print(f"\nnoiser(x,t;x0,x1) = d^T * mu' * gamma + ||d||^2:")
    print(f"  d^T * mu' * gamma = {(d_dot_mu_prime * gamma_t_single)[0].item():.6f}")
    print(f"  ||d||^2 = {d_norm_sq[0].item():.6f}")
    print(f"  Total noiser = {noiser_single[0].item():.6f}")

    # Verify against method
    noiser_via_method = estimator.noiser(
        t_single.unsqueeze(0), x_single, x0_single, x1_single
    )
    print(f"\nVerification via estimator.noiser(): {noiser_via_method[0].item():.6f}")

    # === FULL SCORE (with centering term) ===
    print(f"\n{'=' * 60}")
    print("FULL SCORE: With centering term (for ground truth comparison)")
    print(f"{'=' * 60}")
    print(f"\nFull score = -dim * g'/g + v(t) * noiser / gamma^3")
    print(f"           = -dim * g'/g + (d^T * mu' * gamma + ||d||^2) / gamma^3")
    print(f"\nNote: The neural network learns E[noiser]/v(t), which is later rescaled.")
    print(f"The centering term -dim * g'/g is added during integration.\n")

    # g'/g
    g_prime_over_g = g_prime_t_single / g_t_single
    print(f"g'(t)/g(t):")
    print(f"  Value: {g_prime_over_g[0].item():.6f}")

    # gamma^3
    gamma_cubed = gamma_t_single ** 3
    print(f"\ngamma(t)^3:")
    print(f"  Value: {gamma_cubed[0].item():.6f}")

    # Centering term: -dim * g'/g
    centering_term = -dim * g_prime_over_g
    print(f"\nCentering term: -dim * g'/g where dim={dim}:")
    print(f"  Value: {centering_term[0].item():.6f}")

    # noiser / gamma^3 (this is what gets rescaled by v(t))
    noiser_over_gamma3 = noiser_single / gamma_cubed
    print(f"\nnoiser / gamma^3:")
    print(f"  Value: {noiser_over_gamma3[0].item():.6f}")

    # Full score = -dim * g'/g + noiser / gamma^3
    full_score = centering_term + noiser_over_gamma3
    print(f"\nFull score = -dim * g'/g + noiser / gamma^3:")
    print(f"  Value: {full_score[0].item():.6f}")

    print(f"\n{'=' * 60}")

    # ===== ALL TARGETS COMPARISON (for each batch element) =====
    print(f"\n{'=' * 60}")
    print("ALL TARGETS COMPARISON (for each batch element)")
    print(f"{'=' * 60}")
    print(f"\nComparing noiser targets vs score targets for all batch elements.")
    print(f"Score formula (from direct.py): s = -dim*gamma'/gamma + ||r||^2*gamma'/gamma^3 + r^T*mu'/gamma^2")
    print(f"Noiser formula (direct3.py): noiser = d^T * mu' * gamma + ||d||^2\n")

    # Also compute score targets (as in debug_direct_eldr.py) for comparison
    avg_scores = []
    for i in range(BATCH_SIZE):
        t_i = t[i].expand(n_base)
        x0_i = x0[i].unsqueeze(0).expand(n_base, -1)
        x1_i = x1[i].unsqueeze(0).expand(n_base, -1)

        # Compute score for all base samples using the score formula
        # s = -dim*gamma'/gamma + ||r||^2*gamma'/gamma^3 + r^T*mu'/gamma^2
        gamma_i = estimator.g(t_i)  # [n_base]
        gamma_prime_i = estimator.dgdt(t_i)  # [n_base]
        mu_t_i = estimator.mu(t_i, x0_i, x1_i)  # [n_base, dim]
        mu_prime_i = estimator.dmudt(x0_i, x1_i)  # [n_base, dim]

        r_i = samples_base_t - mu_t_i  # [n_base, dim]
        r_norm_sq_i = (r_i ** 2).sum(dim=-1)  # [n_base]
        r_dot_mu_prime_i = (r_i * mu_prime_i).sum(dim=-1)  # [n_base]

        term1_i = -dim * gamma_prime_i / gamma_i
        term2_i = r_norm_sq_i * gamma_prime_i / (gamma_i ** 3)
        term3_i = r_dot_mu_prime_i / (gamma_i ** 2)
        scores_i = term1_i + term2_i + term3_i  # [n_base]

        avg_scores.append(scores_i.mean())

    avg_scores = torch.stack(avg_scores)

    col1, col2, col3, col4, col5, col6 = "Batch", "t", "noiser", "noiser*v/g^3", "noiser*v/g^3-d*g'/g", "score"
    col7 = "score+d*g'/g"
    print(f"{col1:<6} {col2:<8} {col3:<14} {col4:<14} {col5:<20} {col6:<14} {col7:<14}")
    print("-" * 100)

    for i in range(BATCH_SIZE):
        g_t_i = estimator.g(t[i:i+1])[0].item()
        g_prime_i = estimator.dgdt(t[i:i+1])[0].item()
        v_t_i = estimator._interpolate_v(t[i:i+1])[0].item()
        gamma_cubed_i = g_t_i ** 3

        noiser_i = avg_noisers[i].item()
        noiser_v_gamma3_i = noiser_i * v_t_i / gamma_cubed_i
        centering_i = dim * g_prime_i / g_t_i
        noiser_v_gamma3_minus_centering_i = noiser_v_gamma3_i - centering_i

        score_i = avg_scores[i].item()
        score_plus_centering_i = score_i + centering_i

        print(f"{i:<6} {t[i].item():<8.4f} {noiser_i:<+14.6f} {noiser_v_gamma3_i:<+14.6f} {noiser_v_gamma3_minus_centering_i:<+20.6f} {score_i:<+14.6f} {score_plus_centering_i:<+14.6f}")

    print(f"\nLegend:")
    print(f"  noiser = E[d^T * mu' * gamma + ||d||^2]")
    print(f"  noiser*v/g^3 = noiser * v(t) / gamma^3")
    print(f"  noiser*v/g^3-d*g'/g = noiser * v(t) / gamma^3 - dim*gamma'/gamma")
    print(f"  score = E[-dim*gamma'/gamma + ||r||^2*gamma'/gamma^3 + r^T*mu'/gamma^2]")
    print(f"  score+d*g'/g = score + dim*gamma'/gamma (decentered score)")

    print(f"\n{'=' * 60}")

    # ===== VARIANCE NORMALIZATION =====
    print(f"\n{'=' * 60}")
    print("VARIANCE NORMALIZATION")
    print(f"{'=' * 60}")

    v_t = estimator._interpolate_v(t)
    print(f"\nv(t) (interpolated from precomputed grid):")
    print(f"  Values: {v_t.cpu().numpy()}")

    scaled_targets = avg_noisers / v_t
    print(f"\nScaled targets = avg_noiser / v(t):")
    print(f"  Values: {scaled_targets.cpu().numpy()}")

    print(f"\n{'=' * 60}")

    # Forward pass through neural network
    predictions = estimator.noiser_network(t)  # [batch]
    print(f"\nNeural network predictions (NN(t)):")
    print(f"  Values: {predictions.detach().cpu().numpy()}")
    print(f"  Shape: {predictions.shape}")

    # MSE loss per sample (comparing to scaled targets)
    mse_per_sample = (predictions - scaled_targets.detach()) ** 2
    print(f"\nMSE per sample (compared to scaled targets):")
    print(f"  Values: {mse_per_sample.detach().cpu().numpy()}")

    # Show comparison
    print(f"\nComparison:")
    for i in range(BATCH_SIZE):
        print(f"  Batch {i}: NN prediction = {predictions[i].item():+.6f}, "
              f"Scaled target = {scaled_targets[i].item():+.6f}, "
              f"MSE = {mse_per_sample[i].item():.6f}")

    # ===== LOSS COMPUTATION =====
    print(f"\n{'=' * 60}")
    print("LOSS COMPUTATION")
    print(f"{'=' * 60}")

    print(f"\ng(t) values:")
    print(f"  Values: {g_t.cpu().numpy()}")

    # Compute weights: v(t)^2 / (g(t) + eps)^6
    weights = v_t**2 / (g_t + estimator.eps)**6
    print(f"\nWeights = v(t)^2 / (g(t) + eps)^6:")
    print(f"  Values: {weights.cpu().numpy()}")

    # Weighted loss
    weighted_losses = mse_per_sample * weights
    print(f"\nWeighted loss per sample (MSE * weight):")
    print(f"  Values: {weighted_losses.detach().cpu().numpy()}")

    loss = weighted_losses.mean()
    print(f"\nTotal loss (mean of weighted losses):")
    print(f"  Value: {loss.item():.6f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Print gradient norms
    total_grad_norm = 0.0
    for name, param in estimator.noiser_network.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
    total_grad_norm = np.sqrt(total_grad_norm)
    print(f"\nGradient norm before clipping: {total_grad_norm:.6f}")

    torch.nn.utils.clip_grad_norm_(estimator.noiser_network.parameters(), max_norm=10.0)
    optimizer.step()

    print(f"\n{'=' * 80}")

print("\n" + "=" * 80)
print("Debug session complete!")
print("=" * 80)
