from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
from scipy import integrate

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.time_score_matching.time_score_net_2d import TimeScoreNetwork2D


class TriangularTSM(DensityRatioEstimator):
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
        self.model = TimeScoreNetwork2D(self.input_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

    def path_t_tprime(self, tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
          tau=0   -> (t, t') = (0, 0)
          tau=0.5 -> (t, t') = (0.5, 1)
          tau=1   -> (t, t') = (1, 0)
        """
        t = tau
        t_prime = 4.0 * tau * (1.0 - tau) 
        t = torch.clamp(t, min=self.eps, max=1.0)
        t_prime = torch.clamp(t_prime, min=0.0, max=1.0)
        return t, t_prime

    def sample_x_tau(
        self,
        p0_samples: torch.Tensor,
        p1_samples: torch.Tensor,
        pstar_samples: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        t, t_prime = self.path_t_tprime(tau)
        sqrt_1_minus_t2 = torch.sqrt(torch.clamp(1.0 - t ** 2, min=self.eps))
        sqrt_1_minus_tp2 = torch.sqrt(torch.clamp(1.0 - t_prime ** 2, min=self.eps))
        x_t = t * p0_samples + sqrt_1_minus_t2 * p1_samples
        x_t_tprime = sqrt_1_minus_tp2 * x_t + t_prime * pstar_samples
        return x_t_tprime

 
    def time_score_loss(
        self,
        p0_samples: torch.Tensor,
        p1_samples: torch.Tensor,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        tau0 = torch.zeros((len(p1_samples), 1), device=p1_samples.device) + self.eps
        tau1 = torch.ones((len(p0_samples), 1), device=p0_samples.device)
        t0, tp0 = self.path_t_tprime(tau0)
        t1, tp1 = self.path_t_tprime(tau1)

        if self.reweight:
            lambda_tau = (1.0 - tau ** 2).squeeze()
            lambda_tau0 = (1.0 - tau0.squeeze() ** 2)
            lambda_tau1 = (1.0 - tau1.squeeze() ** 2 + self.eps ** 2)
            lambda_dtau = (-2.0 * tau.squeeze())
        else:
            lambda_tau = lambda_tau0 = lambda_tau1 = 1.0
            lambda_dtau = 0.0

        term1 = (2.0 * self.model(p1_samples, t0, tp0)).squeeze() * lambda_tau0
        term2 = (2.0 * self.model(p0_samples, t1, tp1)).squeeze() * lambda_tau1
        
        tau = tau.clone().detach().requires_grad_(True)
        t_mid, tp_mid = self.path_t_tprime(tau)
        x_tau_score = self.model(x_tau, t_mid, tp_mid)
        x_tau_score_dtau = autograd.grad(x_tau_score.sum(), tau, create_graph=True)[0]
        term3 = (2.0 * x_tau_score_dtau).squeeze() * lambda_tau
        term4 = x_tau_score.squeeze() * lambda_dtau if isinstance(lambda_dtau, torch.Tensor) else x_tau_score.squeeze() * lambda_dtau
        term5 = (x_tau_score ** 2).squeeze() * lambda_tau

        loss = term1 - term2 + term3 + term4 + term5
        return loss.mean()

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor, samples_pstar: torch.Tensor) -> None:
        self.init_model()
        self.model.train()

        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()
        samples_pstar = samples_pstar.float()
        n_p0 = samples_p0.shape[0]
        n_p1 = samples_p1.shape[0]
        n_pstar = samples_pstar.shape[0]

        for _ in range(self.n_epochs):
            p0_idx = torch.randint(0, n_p0, (self.batch_size,))
            p1_idx = torch.randint(0, n_p1, (self.batch_size,))
            pstar_idx = torch.randint(0, n_pstar, (self.batch_size,))

            p0_batch = samples_p0[p0_idx].to(self.device)
            p1_batch = samples_p1[p1_idx].to(self.device)
            pstar_batch = samples_pstar[pstar_idx].to(self.device)

            tau = self.eps + torch.rand(self.batch_size, 1, device=self.device) * (1.0 - 2.0 * self.eps)
            x_tau = self.sample_x_tau(p0_batch, p1_batch, pstar_batch, tau).detach()

            self.optimizer.zero_grad()
            loss = self.time_score_loss(p0_batch, p1_batch, x_tau, tau)
            loss.backward()
            self.optimizer.step()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("TSM model is not trained. Call fit() before predict_ldr().")

        self.model.eval()
        samples = xs.to(self.device)

        with torch.no_grad():
            def ode_func(tau_scalar, y, samples_tensor):
                tau_tensor = torch.ones(samples_tensor.size(0), 1, device=self.device) * float(tau_scalar)
                t, t_prime = self.path_t_tprime(tau_tensor)
                score = self.model(samples_tensor, t, t_prime)  
                return score.squeeze().cpu().numpy()

            ode_fn = lambda tau_scalar, y: ode_func(tau_scalar, y, samples)
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


if __name__ == "__main__":
    from torch.distributions import MultivariateNormal
    from experiments.utils.two_gaussians_kl import create_two_gaussians_kl

    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 10
    KL_DISTANCE = 5

    gaussian_pair = create_two_gaussians_kl(DIM, KL_DISTANCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair["mu0"], gaussian_pair["Sigma0"]
    mu1, Sigma1 = gaussian_pair["mu1"], gaussian_pair["Sigma1"]

    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))

    samples_pstar_train = p0.sample((NSAMPLES_TRAIN,))
    samples_eval = p0.sample((NSAMPLES_TEST,))

    tsm = TriangularTSM(DIM)
    tsm.fit(samples_p0, samples_p1, samples_pstar_train)

    est_ldrs = tsm.predict_ldr(samples_eval)
    true_ldrs = p0.log_prob(samples_eval) - p1.log_prob(samples_eval)
    mae = torch.mean(torch.abs(est_ldrs - true_ldrs))
    print(f"MAE: {mae}")