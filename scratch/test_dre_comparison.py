"""Compare BDRE, MDRE, and TDRE on toy Gaussian problem.

Ground truth: For two multivariate Gaussians p = N(mu_p, Sigma_p) and q = N(mu_q, Sigma_q),
the log density ratio at x is:

log p(x)/q(x) = -0.5 * [
    log|Sigma_p|/|Sigma_q|
    + (x - mu_p)^T Sigma_p^{-1} (x - mu_p)
    - (x - mu_q)^T Sigma_q^{-1} (x - mu_q)
]

Adapted from deep-preemptive-exploration/src/unit_tests/test_tre_comparison.py
Class mapping:
  - DirectDRE -> BDRE (Binary classification DRE)
  - MultinomialTRE -> MDRE (Multiclass classification DRE)
  - MultiheadTRE -> TDRE (Telescoping DRE)
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from src.density_ratio_estimation.bdre import BDRE
from src.density_ratio_estimation.mdre import MDRE
from src.density_ratio_estimation.tdre import TDRE


def compute_ground_truth_log_ratio(
    x: torch.Tensor,
    mu_p: torch.Tensor,
    cov_p: torch.Tensor,
    mu_q: torch.Tensor,
    cov_q: torch.Tensor,
) -> torch.Tensor:
    """Compute closed-form log density ratio for Gaussians.

    Args:
        x: Points to evaluate, shape (batch_size, dim)
        mu_p: Mean of numerator distribution, shape (dim,)
        cov_p: Covariance of numerator distribution, shape (dim, dim)
        mu_q: Mean of denominator distribution, shape (dim,)
        cov_q: Covariance of denominator distribution, shape (dim, dim)

    Returns:
        Log density ratios, shape (batch_size,)
    """
    dim = x.shape[1]

    # Log determinant terms
    log_det_p = torch.linalg.slogdet(cov_p)[1]
    log_det_q = torch.linalg.slogdet(cov_q)[1]

    # Precision matrices
    prec_p = torch.linalg.inv(cov_p)
    prec_q = torch.linalg.inv(cov_q)

    # Mahalanobis distances
    diff_p = x - mu_p.unsqueeze(0)  # (batch, dim)
    diff_q = x - mu_q.unsqueeze(0)  # (batch, dim)

    mahal_p = torch.sum(diff_p @ prec_p * diff_p, dim=1)  # (batch,)
    mahal_q = torch.sum(diff_q @ prec_q * diff_q, dim=1)  # (batch,)

    # Log ratio: log p(x) - log q(x)
    # = -0.5 * (log|Sigma_p| + mahal_p) + 0.5 * (log|Sigma_q| + mahal_q)
    # = 0.5 * (log|Sigma_q| - log|Sigma_p| + mahal_q - mahal_p)
    log_ratio = 0.5 * (log_det_q - log_det_p + mahal_q - mahal_p)

    return log_ratio


def generate_gaussian_data(
    n_samples: int,
    mu_p: torch.Tensor,
    cov_p: torch.Tensor,
    mu_q: torch.Tensor,
    cov_q: torch.Tensor,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate samples from two Gaussian distributions.

    Returns:
        Tuple of (samples_p, samples_q), each shape (n_samples, dim)
    """
    dim = mu_p.shape[0]

    # Create distributions
    dist_p = torch.distributions.MultivariateNormal(mu_p, cov_p)
    dist_q = torch.distributions.MultivariateNormal(mu_q, cov_q)

    # Sample
    samples_p = dist_p.sample((n_samples,)).to(device)
    samples_q = dist_q.sample((n_samples,)).to(device)

    return samples_p, samples_q


@dataclass
class ExperimentConfig:
    """Configuration for the comparison experiment."""
    var_dim: int = 20
    n_train_samples: int = 10000
    n_test_samples: int = 2000
    num_waypoints_list: Tuple[int, ...] = (2, 4, 8)  # Different m values to test
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    # Distribution parameters
    mean_shift: float = 3.0  # Shift in mean
    var_p: float = 0.3  # Variance of p (numerator)
    var_q: float = 2.0  # Variance of q (denominator)


def run_experiment(config: ExperimentConfig):
    """Run comparison experiment."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"Running on device: {config.device}")
    print(f"Variable dimension: {config.var_dim}")
    print(f"Num waypoints to test (m): {config.num_waypoints_list}")
    print()

    # Setup Gaussian distributions
    mu_p = torch.zeros(config.var_dim, device=config.device)
    mu_q = torch.ones(config.var_dim, device=config.device) * config.mean_shift

    cov_p = torch.eye(config.var_dim, device=config.device) * config.var_p
    cov_q = torch.eye(config.var_dim, device=config.device) * config.var_q

    print("Distribution setup:")
    print(f"  p ~ N(0, {config.var_p}*I)")
    print(f"  q ~ N({config.mean_shift}, {config.var_q}*I)")
    print()

    # Generate data
    print("Generating data...")
    train_p, train_q = generate_gaussian_data(
        config.n_train_samples, mu_p, cov_p, mu_q, cov_q, config.device
    )
    test_p, test_q = generate_gaussian_data(
        config.n_test_samples, mu_p, cov_p, mu_q, cov_q, config.device
    )

    # Compute ground truth on test samples from both distributions
    test_all = torch.cat([test_p, test_q], dim=0)
    ground_truth = compute_ground_truth_log_ratio(test_all, mu_p, cov_p, mu_q, cov_q)

    print(f"Ground truth log ratio stats:")
    print(f"  On p samples: mean={ground_truth[:config.n_test_samples].mean():.4f}, "
          f"std={ground_truth[:config.n_test_samples].std():.4f}")
    print(f"  On q samples: mean={ground_truth[config.n_test_samples:].mean():.4f}, "
          f"std={ground_truth[config.n_test_samples:].std():.4f}")
    print()

    def correlation(pred, target):
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        return (pred_centered * target_centered).sum() / (
            pred_centered.norm() * target_centered.norm()
        )

    all_results = {}

    # =========================================================================
    # 1. BDRE (baseline)
    # =========================================================================
    print("=" * 70)
    print("Training BDRE (baseline)...")

    bdre = BDRE(input_dim=config.var_dim)
    bdre.fit(train_p, train_q)

    with torch.no_grad():
        bdre_pred = bdre.predict_ldr(test_all)

    bdre_mse = ((bdre_pred - ground_truth) ** 2).mean().item()
    bdre_mae = (bdre_pred - ground_truth).abs().mean().item()
    bdre_bias = (bdre_pred - ground_truth).mean().item()
    bdre_corr = correlation(bdre_pred, ground_truth).item()

    all_results['BDRE'] = {
        'mse': bdre_mse,
        'mae': bdre_mae,
        'bias': bdre_bias,
        'corr': bdre_corr,
    }

    print(f"  MSE: {bdre_mse:.4f}, MAE: {bdre_mae:.4f}, Bias: {bdre_bias:.4f}, Corr: {bdre_corr:.4f}")
    print()

    # =========================================================================
    # Loop over different m values
    # =========================================================================
    for num_waypoints in config.num_waypoints_list:
        print("=" * 70)
        print(f"Testing with m = {num_waypoints} waypoints")
        print("=" * 70)

        # Reset seed for fair comparison
        torch.manual_seed(config.seed + num_waypoints)

        # ---------------------------------------------------------------------
        # TDRE (Telescoping - multiple binary classifiers)
        # ---------------------------------------------------------------------
        print(f"\nTraining TDRE (m={num_waypoints})...")

        tdre = TDRE(input_dim=config.var_dim, num_waypoints=num_waypoints)
        tdre.fit(train_p, train_q)

        with torch.no_grad():
            tdre_pred = tdre.predict_ldr(test_all)

        tdre_mse = ((tdre_pred - ground_truth) ** 2).mean().item()
        tdre_mae = (tdre_pred - ground_truth).abs().mean().item()
        tdre_bias = (tdre_pred - ground_truth).mean().item()
        tdre_corr = correlation(tdre_pred, ground_truth).item()

        all_results[f'TDRE_m{num_waypoints}'] = {
            'mse': tdre_mse,
            'mae': tdre_mae,
            'bias': tdre_bias,
            'corr': tdre_corr,
            'num_waypoints': num_waypoints,
        }

        print(f"  MSE: {tdre_mse:.4f}, MAE: {tdre_mae:.4f}, Bias: {tdre_bias:.4f}, Corr: {tdre_corr:.4f}")

        # ---------------------------------------------------------------------
        # MDRE (Multiclass classification)
        # ---------------------------------------------------------------------
        print(f"\nTraining MDRE (m={num_waypoints})...")

        mdre = MDRE(input_dim=config.var_dim, num_waypoints=num_waypoints)
        mdre.fit(train_p, train_q)

        with torch.no_grad():
            mdre_pred = mdre.predict_ldr(test_all)

        mdre_mse = ((mdre_pred - ground_truth) ** 2).mean().item()
        mdre_mae = (mdre_pred - ground_truth).abs().mean().item()
        mdre_bias = (mdre_pred - ground_truth).mean().item()
        mdre_corr = correlation(mdre_pred, ground_truth).item()

        all_results[f'MDRE_m{num_waypoints}'] = {
            'mse': mdre_mse,
            'mae': mdre_mae,
            'bias': mdre_bias,
            'corr': mdre_corr,
            'num_waypoints': num_waypoints,
        }

        print(f"  MSE: {mdre_mse:.4f}, MAE: {mdre_mae:.4f}, Bias: {mdre_bias:.4f}, Corr: {mdre_corr:.4f}")

    # =========================================================================
    # Summary Table
    # =========================================================================
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Method':<20} {'Waypoints':<12} {'MSE':<12} {'MAE':<10} {'Bias':<12} {'Corr':<8}")
    print("-" * 90)

    base_mse = all_results['BDRE']['mse']

    for name, res in all_results.items():
        num_wp = res.get('num_waypoints', '-')
        ratio = res['mse'] / base_mse
        print(f"{name:<20} {str(num_wp):<12} {res['mse']:<12.4f} {res['mae']:<10.4f} {res['bias']:<12.4f} {res['corr']:<8.4f} ({ratio:.2f}x)")

    return all_results


if __name__ == '__main__':
    config = ExperimentConfig()
    results = run_experiment(config)
