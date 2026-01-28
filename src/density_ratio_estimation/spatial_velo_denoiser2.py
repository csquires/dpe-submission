from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import integrate

from src.density_ratio_estimation.base import DensityRatioEstimator


class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron for estimating vector fields.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
            
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [batch, 1]
            x: Spatial values [batch, dim]
        Returns:
            Vector field [batch, output_dim]
        """
        tx = torch.cat([t, x], dim=-1)
        return self.net(tx)


def compute_divergence(output: torch.Tensor, x: torch.Tensor, epsilon: torch.Tensor = None) -> torch.Tensor:
    """
    Computes exact divergence (trace of Jacobian).

    Args:
        output: Vector field [batch, dim]
        x: Input points [batch, dim] (must have requires_grad=True)
        epsilon: Ignored (kept for API compatibility)

    Returns:
        Divergence estimates [batch]
    """
    batch_size, dim = x.shape
    divergence = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

    for i in range(dim):
        grad_i = torch.autograd.grad(
            outputs=output[:, i].sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
        )[0]
        divergence = divergence + grad_i[:, i]

    return divergence


class SpatialVeloDenoiser(DensityRatioEstimator):
    """
    Denoiser-based variant of the interpolant estimator.
    
    Modified to train Velocity (b) and Denoiser (eta) networks completely separately
    and sequentially.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 2e-3,
        k: float = 0.5,
        n_t: int = 50,
        eps: float = 0.01,
        device: Optional[str] = None,
        integration_steps: int = 10000,
        integration_type: Literal['1', '2', '3'] = '1',
        verbose: bool = False,
        log_every: int = 100,
        antithetic: bool = False,
    ):
        super().__init__(input_dim)
        self.integration_type = integration_type
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.k = k
        self.n_t = n_t
        self.eps = eps
        self.integration_steps = integration_steps
        self.verbose = verbose
        self.log_every = log_every
        self.antithetic = antithetic
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.net_b = None
        self.net_eta = None

    def init_model(self) -> None:
        # Initialize two completely separate networks
        self.net_b = MLP(self.input_dim, self.hidden_dim, output_dim=self.input_dim).to(self.device)
        self.net_eta = MLP(self.input_dim, self.hidden_dim, output_dim=self.input_dim).to(self.device)

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """gamma(t) = (1 - exp(-k*t)) * (1 - exp(-k*(1-t)))"""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        return (1 - torch.exp(-self.k * t)) * (1 - torch.exp(-self.k * (1 - t)))

    def dgamma_dt(self, t: torch.Tensor) -> torch.Tensor:
        """gamma'(t)"""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        exp_kt = torch.exp(-self.k * t)
        exp_k1t = torch.exp(-self.k * (1 - t))
        return self.k * exp_kt * (1 - exp_k1t) - self.k * exp_k1t * (1 - exp_kt)

    def compute_time_score(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        gamma_t: torch.Tensor,
        gamma_dot_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute partial_t log rho(t, x) using:
        dt_log_rho = -div(b) - b.score 
                   = -div(b) + b.eta/gamma
        """
        x = x.clone().requires_grad_(True)

        # Run separate forward passes
        b_pred = self.net_b(t, x)
        eta_pred = self.net_eta(t, x)

        # Exact divergence (trace of Jacobian)
        div_b = compute_divergence(b_pred, x)

        # Scalar terms
        b_dot_eta = (b_pred * eta_pred).sum(dim=-1, keepdim=True)

        return -div_b.view(-1, 1) + b_dot_eta / gamma_t

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        self.init_model()

        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()
        n_p0 = samples_p0.shape[0]
        n_p1 = samples_p1.shape[0]

        if self.verbose:
            print(f"[SpatialVeloDenoiser] Starting Sequential Training with batch-based time sampling.")
            print(f"[SpatialVeloDenoiser] gamma range: [{self.gamma(torch.tensor(self.eps)).item():.4f}, {self.gamma(torch.tensor(0.5)).item():.4f}]")

        # ==========================================
        # PHASE 1: Train Velocity Network (b)
        # ==========================================
        if self.verbose:
            print(f"\n[Phase 1/2] Training Velocity Network (b) for {self.n_epochs} epochs...")

        self.net_b.train()
        self.net_eta.eval() # Eta not trained here
        optimizer_b = optim.Adam(self.net_b.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        for epoch in range(self.n_epochs):
            # Sample batches
            p0_idx = torch.randint(0, n_p0, (self.batch_size,))
            p1_idx = torch.randint(0, n_p1, (self.batch_size,))
            x0 = samples_p0[p0_idx].to(self.device)
            x1 = samples_p1[p1_idx].to(self.device)

            # Sample time uniformly from [eps, 1-eps] per sample
            t = torch.rand(self.batch_size, device=self.device) * (1 - 2 * self.eps) + self.eps
            t_batch = t.unsqueeze(-1)  # [B, 1]

            # Sample noise (independent per sample)
            z = torch.randn_like(x0)

            # Compute gamma and gamma' for each t
            gamma_t = self.gamma(t).unsqueeze(-1)  # [B, 1]
            gamma_prime_t = self.dgamma_dt(t).unsqueeze(-1)  # [B, 1]

            # Interpolant mean
            I_t = (1 - t_batch) * x0 + t_batch * x1
            dtIt = x1 - x0  # derivative of I_t w.r.t. t

            if self.antithetic:
                # Antithetic sampling: x_t+ and x_t-
                x_t_plus = I_t + gamma_t * z
                x_t_minus = I_t - gamma_t * z

                b_plus = self.net_b(t_batch, x_t_plus)
                b_minus = self.net_b(t_batch, x_t_minus)

                # Velocity loss with antithetic sampling
                b_norm_sq_plus = (b_plus ** 2).sum(dim=-1)
                b_norm_sq_minus = (b_minus ** 2).sum(dim=-1)
                target_dot_b_plus = ((dtIt + gamma_prime_t * z) * b_plus).sum(dim=-1)
                target_dot_b_minus = ((dtIt - gamma_prime_t * z) * b_minus).sum(dim=-1)
                loss_b = (0.25 * b_norm_sq_plus - 0.5 * target_dot_b_plus
                        + 0.25 * b_norm_sq_minus - 0.5 * target_dot_b_minus).mean()
            else:
                # Standard (non-antithetic) training
                x_t = I_t + gamma_t * z
                b_pred = self.net_b(t_batch, x_t)

                # Velocity loss: 0.5*||b||² - ((x1 - x0)+gamma_t'z)·b
                b_norm_sq = (b_pred ** 2).sum(dim=-1)
                target_dot_b = ((dtIt + gamma_prime_t * z) * b_pred).sum(dim=-1)
                loss_b = (0.5 * b_norm_sq - target_dot_b).mean()

            optimizer_b.zero_grad()
            loss_b.backward()
            optimizer_b.step()

            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"  [Epoch {epoch+1}] loss_b={loss_b.item():.4f}")

        # ==========================================
        # PHASE 2: Train Denoiser Network (eta)
        # ==========================================
        if self.verbose:
            print(f"\n[Phase 2/2] Training Denoiser Network (eta) for {self.n_epochs} epochs...")

        self.net_b.eval() # B is now fixed
        self.net_eta.train()
        optimizer_eta = optim.Adam(self.net_eta.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        for epoch in range(self.n_epochs):
            p0_idx = torch.randint(0, n_p0, (self.batch_size,))
            p1_idx = torch.randint(0, n_p1, (self.batch_size,))
            x0 = samples_p0[p0_idx].to(self.device)
            x1 = samples_p1[p1_idx].to(self.device)

            # Sample time uniformly from [eps, 1-eps] per sample
            t = torch.rand(self.batch_size, device=self.device)
            t_batch = t.unsqueeze(-1)  # [B, 1]

            # Sample noise (independent per sample)
            z = torch.randn_like(x0)

            # Compute gamma for each t
            gamma_t = self.gamma(t).unsqueeze(-1)  # [B, 1]

            # Interpolant
            x_t = (1 - t_batch) * x0 + t_batch * x1 + gamma_t * z

            # Predict eta
            eta_pred = self.net_eta(t_batch, x_t)

            # Denoiser loss: 0.5*||d||² - z·d
            # (Minimizes distance to z, consistent with d = z)
            eta_norm_sq = (eta_pred ** 2).sum(dim=-1)
            z_dot_eta = (z * eta_pred).sum(dim=-1)
            loss_eta = (0.5 * eta_norm_sq - z_dot_eta).mean()

            optimizer_eta.zero_grad()
            loss_eta.backward()
            optimizer_eta.step()

            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"  [Epoch {epoch+1}] loss_eta={loss_eta.item():.4f}")

        if self.verbose:
            print(f"[SpatialVeloDenoiser] Training complete")
        
        self.net_eta.eval() # Set both to eval for inference

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        if self.net_b is None or self.net_eta is None:
            raise RuntimeError("SpatialVeloDenoiser model is not trained. Call fit() before predict_ldr().")

        self.net_b.eval()
        self.net_eta.eval()
        samples = xs.float().to(self.device)
        n_samples = samples.shape[0]

        # Integration grid
        n_points = self.integration_steps
        if n_points % 2 == 0:
            n_points += 1
        t_vals = torch.linspace(self.eps, 1 - self.eps, n_points, device=self.device)

        # Compute time scores at each t for all samples
        time_scores = []
        for t_val in t_vals:
            t_batch = torch.full((n_samples, 1), t_val.item(), device=self.device)
            gamma_t = self.gamma(t_val)
            gamma_dot_t = self.dgamma_dt(t_val)

            time_score = self.compute_time_score(t_batch, samples, gamma_t, gamma_dot_t)
            time_scores.append(time_score.detach())

        time_scores = torch.stack(time_scores, dim=0)  # [n_points, n_samples, 1]
        time_scores = time_scores.squeeze(-1)  # [n_points, n_samples]

        if self.integration_type == '3':
            # Simpson's rule integration
            t_np = t_vals.cpu().numpy()
            h = (t_np[-1] - t_np[0]) / (n_points - 1)

            integrand = time_scores.cpu().numpy()  # [n_points, n_samples]
            integral = integrand[0] + integrand[-1]
            for i in range(1, n_points - 1):
                if i % 2 == 0:
                    integral += 2 * integrand[i]
                else:
                    integral += 4 * integrand[i]
            integral *= h / 3
            out = -torch.from_numpy(integral)
        elif self.integration_type == '1':
            out = -time_scores.mean(dim=0).cpu()
        elif self.integration_type == '2':
            out = -torch.trapz(time_scores, t_vals, dim=0).cpu()

        return out


if __name__ == '__main__':
    from torch.distributions import MultivariateNormal
    from experiments.utils.prescribed_kls import create_two_gaussians_kl
    from src.density_ratio_estimation.bdre import BDRE
    from src.models.binary_classification import make_binary_classifier

    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 100
    KL_DISTANCE = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # === CREATE SYNTHETIC DATA ===
    gaussian_pair = create_two_gaussians_kl(DIM, KL_DISTANCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair['mu0'].to(DEVICE), gaussian_pair['Sigma0'].to(DEVICE)
    mu1, Sigma1 = gaussian_pair['mu1'].to(DEVICE), gaussian_pair['Sigma1'].to(DEVICE)
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)
    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))
    samples_test = p0.sample((NSAMPLES_TEST,))

    # === TRUE LDR ===
    true_ldrs = p0.log_prob(samples_test) - p1.log_prob(samples_test)
    print(f"True LDR range: [{true_ldrs.min().item():.4f}, {true_ldrs.max().item():.4f}]")
    print(f"True LDR mean: {true_ldrs.mean().item():.4f}")
    print()

    # === BDRE BASELINE ===
    print("=" * 50)
    print("BDRE (Baseline)")
    print("=" * 50)
    classifier = make_binary_classifier("default", input_dim=DIM)
    bdre = BDRE(classifier, device=DEVICE)
    bdre.fit(samples_p0.to(DEVICE), samples_p1.to(DEVICE))
    bdre_ldrs = bdre.predict_ldr(samples_test.to(DEVICE))
    bdre_mae = torch.mean(torch.abs(bdre_ldrs.cpu() - true_ldrs.cpu()))
    print(f"BDRE MAE: {bdre_mae.item():.4f}")
    print(f"BDRE LDR range: [{bdre_ldrs.min().item():.4f}, {bdre_ldrs.max().item():.4f}]")
    print()

    # === SPATIAL VELO DENOISER (Sequential Training) ===
    print("=" * 50)
    print(f"SpatialVeloDenoiser (Sequential Separate Networks)")
    print("=" * 50)
    import pandas as pd
    report = {
        'eps': [], 'epochs': [], 'antithetic': [], 'lr': [], 
        'k': [], 'steps': [],'type':[], 'mae': []
    }
    param_grid = {
        'eps': [2.2e-4, 2.2e-3, 2.2e-2], #[2.1e-3, 2.2e-3, 2.3e-3],
        'epochs': [300, 3000, 6000],
        'antithetic': [True],
        'lr': [1.4e-3, 9e-4, 5e-3, 1e-2], #[1.3e-3, 1.4e-3, 1.5e-3],
        'k': [20],
        'steps': [3000, 15000],
        'type': ['2', '3']
    }
    for r in range(1):
        for this_eps in param_grid['eps']:
            for this_epochs in param_grid['epochs']:
                for this_antithetic in param_grid['antithetic']:
                    for this_lr in param_grid['lr']:
                        for this_k in param_grid['k']:
                            for this_integration_steps in param_grid['steps']:
                                for this_integration_type in param_grid['type']:
                                
                                    estimator = SpatialVeloDenoiser(
                                        DIM,
                                        n_epochs=this_epochs,
                                        verbose=False,
                                        log_every=101,
                                        device=DEVICE,
                                        eps=this_eps,
                                        antithetic=this_antithetic,
                                        lr=this_lr,
                                        k=this_k,
                                        integration_steps=this_integration_steps,
                                        integration_type=this_integration_type,
                                    )
                                
                                    estimator.fit(samples_p0, samples_p1)

                                    est_ldrs = estimator.predict_ldr(samples_test)
                                    mae = torch.mean(torch.abs(est_ldrs.cpu() - true_ldrs.cpu())).item()
                                    
                                    report['eps'].append(this_eps)
                                    report['epochs'].append(this_epochs)
                                    report['antithetic'].append(this_antithetic)
                                    report['lr'].append(this_lr)
                                    report['k'].append(this_k)
                                    report['steps'].append(this_integration_steps)
                                    report['type'].append(this_integration_type)
                                    report['mae'].append(mae)
                                    print(f'Report so far:\n{pd.DataFrame(report)}')

    # --- summary ---
    df = pd.DataFrame(report)
    hyperparams = ['eps', 'epochs', 'antithetic', 'lr', 'k', 'steps', 'type']

    print("\n" + "="*80)
    print("FINAL STATISTICAL REPORT")
    print("="*80)

    # hyperparameter treatment
    print("\n--- Single Hyperparameter Stats ---")
    for param in hyperparams:
        print(f"\nStats by [{param}]:")
        stats = df.groupby(param)['mae'].agg(['mean', 'std', 'min', 'max', lambda x: x.quantile(0.75)-x.quantile(0.25)])
        print(stats)

    # pairwise hyperparam treatment
    print("\n" + "-"*40)
    print("--- Pairwise Hyperparameter Stats ---")
    print("-" * 40)
    import itertools
    for p1, p2 in itertools.combinations(hyperparams, 2):
        print(f"\nStats by [{p1} AND {p2}]:")
        stats = df.groupby([p1, p2])['mae'].agg(['mean', 'std', 'min', 'max', lambda x:x.quantile(0.75)-x.quantile(0.25)])
        print(stats)

    print("\n" + "="*80)
    print("GLOBAL T10:")
    print(df.sort_values(by='mae', ascending=True).head(10))
    print("="*80)
    
    # print(f"SpatialVeloDenoiser MAE:\n{report}")
    print(f"SpatialVeloDenoiser LDR range: [{est_ldrs.min().item():.4f}, {est_ldrs.max().item():.4f}]")
    print()

    # === COMPARISON ===
    print("=" * 50)
    print("Comparison")
    print("=" * 50)
    print(f"BDRE MAE:                {bdre_mae.item():.4f}")
    print(f"SpatialVeloDenoiser MAE: {mae:.4f}")