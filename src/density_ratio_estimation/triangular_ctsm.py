"""
TriangularCTSM: Continuous-time score matching for triangular density ratio estimation.

V2 (barycentric continuous path via three anchor distributions).
"""
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from scipy import integrate

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D
from src.waypoints.path_1d import CtsmPath1D
from src.waypoints.triangular_continuous import BarycentricCtsm1D
from src.waypoints.piecewise_sb import PiecewiseSBCtsm1D


class TriangularCTSM(DensityRatioEstimator):
    """
    Triangular continuous-time score matching for density ratio estimation.

    Uses a continuous barycentric path through p0 -> p* -> p1 (flavor-A path).
    Trains a score network via MSE loss matching target scores from the path.
    Inference: ODE integration from tau=eps to 1-eps yields log(p0/p1).

    Contract: fit(samples_p0, samples_p1, samples_pstar) with three tensors [N, D].
    predict_ldr(xs) returns log density ratios as 1D CPU tensor [N].
    """

    def __init__(
        self,
        input_dim: int,
        path: Optional[CtsmPath1D] = None,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1e-3,
        eps: float = 1e-3,
        device: Optional[str] = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        n_hidden_layers: int = 3,
    ) -> None:
        """
        Initialize TriangularCTSM.

        Args:
            input_dim: Input dimension D.
            path: CtsmPath1D instance. If None, instantiate BarycentricCtsm1D(sigma=1.0, vertex=0.5, eps=eps).
            hidden_dim: Hidden layer width for TimeScoreNetwork1D.
            n_epochs: Number of training epochs.
            batch_size: Batch size for stochastic gradient descent.
            lr: Adam learning rate.
            eps: Margin for tau sampling and ODE integration bounds. tau in [eps, 1-eps].
            device: Device string ("cuda", "cpu", etc.). If None, auto-detect: cuda if available, else cpu.
            rtol: Relative tolerance for ODE solver.
            atol: Absolute tolerance for ODE solver.
            n_hidden_layers: Number of hidden layers for TimeScoreNetwork1D.
        """
        super().__init__(input_dim)

        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.rtol = rtol
        self.atol = atol
        self.n_hidden_layers = n_hidden_layers

        # device resolution
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # path resolution
        if path is None:
            self.path = BarycentricCtsm1D(sigma=1.0, vertex=0.5, eps=eps)
        else:
            self.path = path

        # model placeholders
        self.model = None
        self.optimizer = None

    def init_model(self) -> None:
        """
        Initialize or reinitialize the score network and optimizer.

        Constructs TimeScoreNetwork1D(input_dim, hidden_dim) and moves to device.
        Creates Adam optimizer with standard betas and eps.
        """
        self.model = TimeScoreNetwork1D(self.input_dim, self.hidden_dim, n_hidden_layers=self.n_hidden_layers).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """
        Train the score network on three-tensor contract.

        Args:
            samples_p0: Samples from p0, shape [N0, D].
            samples_p1: Samples from p1, shape [N1, D].
            samples_pstar: Samples from p* (anchor distribution), shape [Nstar, D].

        Training loop:
        - Randomly sample from each distribution each epoch.
        - Generate tau uniformly in [eps, 1-eps].
        - Call path.sample_and_target(x0, x1, xstar, tau, epsilon) to obtain (x_tau, target, lambda_t).
        - Compute MSE loss: (target - lambda_t * pred)^2.
        - Update model via gradient descent.
        """
        self.init_model()
        self.model.train()

        # move samples to device and cast to float
        samples_p0 = samples_p0.to(self.device).float()
        samples_p1 = samples_p1.to(self.device).float()
        samples_pstar = samples_pstar.to(self.device).float()

        # sample counts
        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]
        n_star = samples_pstar.shape[0]

        # training loop
        for _ in range(self.n_epochs):
            # sample indices
            idx0 = torch.randint(0, n0, (self.batch_size,), device=self.device)
            idx1 = torch.randint(0, n1, (self.batch_size,), device=self.device)
            idx_star = torch.randint(0, n_star, (self.batch_size,), device=self.device)

            # extract minibatches
            x0 = samples_p0[idx0]  # [B, D]
            x1 = samples_p1[idx1]  # [B, D]
            xstar = samples_pstar[idx_star]  # [B, D]

            # sample tau in [eps, 1-eps]
            tau = (
                torch.rand(self.batch_size, 1, device=self.device)
                * (1.0 - 2.0 * self.eps)
                + self.eps
            )  # [B, 1]

            # sample noise
            epsilon = torch.randn_like(x0)  # [B, D]

            # get path samples and targets (all detached)
            x_tau, target, lambda_t = self.path.sample_and_target(x0, x1, xstar, tau, epsilon)
            # x_tau: [B, D]
            # target: [B, 1]
            # lambda_t: [B, 1]

            # forward pass
            pred = self.model(x_tau, tau)  # [B, 1]

            # MSE loss
            loss = torch.mean((target - lambda_t * pred) ** 2)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.model.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict log density ratios log(p0(x) / p1(x)) via ODE integration.

        Args:
            xs: Test samples, shape [N, D], on CPU or device (will be moved to self.device).

        Returns:
            Log density ratios, shape [N], on CPU.

        Procedure:
        - Set model to eval mode.
        - Integrate ODE d(log_ratio)/d(tau) = -score(x, tau) from tau=eps to 1-eps.
        - ODE initial condition: log_ratio(tau=eps) = 0.
        - Use scipy.integrate.solve_ivp with RK45 method.
        """
        if self.model is None:
            raise RuntimeError(
                "TriangularCTSM is not trained. Call fit() before predict_ldr()."
            )

        self.model.eval()

        # move samples to device
        samples = xs.to(self.device).float()
        n = samples.shape[0]

        with torch.no_grad():

            def ode_func(tau_scalar, y):
                """ODE function for scipy.integrate.solve_ivp."""
                # tau_scalar: float in [eps, 1-eps]
                # y: array [n] (current log_ratio values)

                # create tau tensor
                tau_tensor = torch.full(
                    (n, 1),
                    float(tau_scalar),
                    device=self.device,
                    dtype=torch.float32,
                )

                # evaluate score network
                score = self.model(samples, tau_tensor)  # [n, 1]

                # return dy/dtau = -score
                dydt = (-score).squeeze(-1).cpu().numpy()  # [n]
                return dydt

            # solve ODE from eps to 1-eps
            solution = integrate.solve_ivp(
                ode_func,
                (self.eps, 1.0 - self.eps),
                np.zeros(n),  # initial condition
                method="RK45",
                rtol=self.rtol,
                atol=self.atol,
            )

            # extract final values (at tau = 1 - eps)
            log_ratios = solution.y[:, -1]  # [n]

        return torch.from_numpy(log_ratios)


if __name__ == "__main__":
    from torch.distributions import MultivariateNormal

    from experiments.utils.prescribed_kls import create_two_gaussians_kl

    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 10
    KL_DIVERGENCE = 5.0

    # create Gaussian pair with controlled KL divergence
    gaussian_pair = create_two_gaussians_kl(
        dim=DIM,
        k=KL_DIVERGENCE,
        beta=0.5,
    )
    mu0 = gaussian_pair["mu0"]
    Sigma0 = gaussian_pair["Sigma0"]
    mu1 = gaussian_pair["mu1"]
    Sigma1 = gaussian_pair["Sigma1"]

    # instantiate p0 and p1
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

    # instantiate p* (anchor): midpoint in mean and covariance
    mu_star = (mu0 + mu1) / 2.0
    Sigma_star = (Sigma0 + Sigma1) / 2.0
    pstar = MultivariateNormal(mu_star, covariance_matrix=Sigma_star)

    # sample from all three distributions
    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))
    samples_pstar = pstar.sample((NSAMPLES_TRAIN,))
    samples_test = p0.sample((NSAMPLES_TEST,))

    # instantiate and train estimator
    estimator = TriangularCTSM(input_dim=DIM)
    estimator.fit(samples_p0, samples_p1, samples_pstar)

    # predict and evaluate
    est_ldrs = estimator.predict_ldr(samples_test)
    true_ldrs = p0.log_prob(samples_test) - p1.log_prob(samples_test)
    mae = torch.mean(torch.abs(est_ldrs - true_ldrs))

    print(f"MAE: {mae}")

    # V1 CTSM: piecewise-SB path, vertex sweep over {0.2, 0.5, 0.8}
    for vertex in [0.2, 0.5, 0.8]:
        # construct V1 path with current vertex
        path_v1 = PiecewiseSBCtsm1D(sigma=1.0, vertex=vertex, eps=1e-3)

        # construct estimator with V1 path
        estimator_v1 = TriangularCTSM(input_dim=DIM, path=path_v1)

        # fit on training samples (reuse from V2 block)
        estimator_v1.fit(samples_p0, samples_p1, samples_pstar)

        # predict on test samples
        est_ldrs_v1 = estimator_v1.predict_ldr(samples_test)

        # compute MAE against true log density ratios (reuse from V2 block)
        mae_v1 = torch.mean(torch.abs(est_ldrs_v1 - true_ldrs))

        # print result
        print(f"[V1 CTSM, vertex={vertex}] MAE: {mae_v1}")

        # smoke-test bound: catch "totally broken" without policing toy noise
        # (NSAMPLES_TEST=10 yields high MAE variance; V2 itself reports ~3.3 here)
        assert torch.isfinite(mae_v1) and mae_v1 < 10.0, (
            f"V1 CTSM vertex={vertex} regression: MAE {mae_v1} >= 10.0"
        )
