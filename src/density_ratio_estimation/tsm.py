from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
from scipy import integrate

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D


class TSM(DensityRatioEstimator):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1e-3,
        reweight: bool = False,
        eps: float = 1e-5,
        device: Optional[str] = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ):
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.reweight = reweight
        self.eps = eps
        self.rtol = rtol
        self.atol = atol
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = None
        self.optimizer = None

    def init_model(self) -> None:
        self.model = TimeScoreNetwork1D(self.input_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

    def time_score_loss(
        self,
        p0_samples: torch.Tensor,
        p1_samples: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t0 = torch.zeros((len(p1_samples), 1), device=p1_samples.device) + self.eps
        t1 = torch.ones((len(p0_samples), 1), device=p0_samples.device)

        if self.reweight:
            lambda_t = (1 - t ** 2).squeeze()
            lambda_t0 = (1 - t0.squeeze() ** 2)
            lambda_t1 = (1 - t1.squeeze() ** 2 + self.eps ** 2)
            lambda_dt = (-2 * t.squeeze())
        else:
            lambda_t = lambda_t0 = lambda_t1 = 1.0
            lambda_dt = 0.0

        # term1 = (2 * self.model(p1_samples, t0)).squeeze() * lambda_t0
        # term2 = (2 * self.model(p0_samples, t1)).squeeze() * lambda_t1

        term1 = (2 * self.model(p0_samples, t0)).squeeze() * lambda_t0
        term2 = (2 * self.model(p1_samples, t1)).squeeze() * lambda_t1

        t = t.clone().detach().requires_grad_(True)
        x_t_score = self.model(x_t, t)
        x_t_score_dt = autograd.grad(x_t_score.sum(), t, create_graph=True)[0]
        term3 = (2 * x_t_score_dt).squeeze() * lambda_t
        term4 = x_t_score.squeeze() * lambda_dt if isinstance(lambda_dt, torch.Tensor) else x_t_score.squeeze() * lambda_dt
        term5 = (x_t_score ** 2).squeeze() * lambda_t

        loss = term1 - term2 + term3 + term4 + term5
        return loss.mean()

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        self.init_model()
        self.model.train()

        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()
        n_p0 = samples_p0.shape[0]
        n_p1 = samples_p1.shape[0]

        for _ in range(self.n_epochs):
            p0_idx = torch.randint(0, n_p0, (self.batch_size,))
            p1_idx = torch.randint(0, n_p1, (self.batch_size,))
            p0_samples = samples_p0[p0_idx].to(self.device)
            p1_samples = samples_p1[p1_idx].to(self.device)

            t = torch.rand(self.batch_size, 1, device=self.device) * (1 - self.eps)
            # x_t = t * p0_samples + torch.sqrt(1 - t ** 2) * p1_samples
            x_t = torch.sqrt(1 - t ** 2) * p0_samples + t * p1_samples

            self.optimizer.zero_grad()
            loss = self.time_score_loss(p0_samples, p1_samples, x_t, t)
            loss.backward()
            self.optimizer.step()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("TSM model is not trained. Call fit() before predict_ldr().")

        self.model.eval()
        samples = xs.to(self.device)

        with torch.no_grad():
            def ode_func(t, y, samples_tensor):
                t_tensor = torch.ones(samples_tensor.size(0), 1, device=self.device) * t
                score = self.model(samples_tensor, t_tensor)
                # return score.squeeze().cpu().numpy()
                return (-score).squeeze().cpu().numpy()

            ode_fn = lambda t, y: ode_func(t, y, samples)
            solution = integrate.solve_ivp(
                ode_fn,
                (self.eps, 1.0),
                np.zeros((samples.size(0),)),
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
            )
            log_ratios = solution.y[:, -1]

        return torch.from_numpy(log_ratios)


if __name__ == '__main__':
    from torch.distributions import MultivariateNormal
    from experiments.utils.two_gaussians_kl import create_two_gaussians_kl
    
    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 10
    KL_DIVERGENCE = 5

    # === CREATE SYNTHETIC DATA ===
    gaussian_pair = create_two_gaussians_kl(DIM, KL_DIVERGENCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
    mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)
    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))
    samples_pstar1 = p0.sample((NSAMPLES_TEST,))

    # === DENSITY RATIO ESTIMATION ===
    tsm = TSM(DIM)
    tsm.fit(samples_p0, samples_p1)

    # === EVALUATION ===
    est_ldrs = tsm.predict_ldr(samples_pstar1)
    true_ldrs = p0.log_prob(samples_pstar1) - p1.log_prob(samples_pstar1)
    mae = torch.mean(torch.abs(est_ldrs - true_ldrs))
    print(f'MAE: {mae}')
