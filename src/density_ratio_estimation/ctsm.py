from typing import Optional

import torch
import torch.optim as optim

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.density_ratio_estimation._ema import EMA, maybe_clip_grad, sample_time_and_iw
from src.models.time_score_matching.time_score_net_1d import TimeScoreNetwork1D


class CTSM(DensityRatioEstimator):
    """
    Conditional Time Score Matching for density ratio estimation.

    Uses Schrodinger Bridge path with closed-form conditional time score target.
    Replaces TSM's Hyvarinen loss with direct MSE regression to avoid autograd
    w.r.t. t. Training: sample t, epsilon, construct SB path x_t, compute
    closed-form target and weight via _epsilon_target, then MSE(target - weight*pred).
    Inference: ODE integration from t=eps to t=1-eps.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 1e-3,
        sigma: float = 1.0,
        eps: float = 1e-3,
        device: Optional[str] = None,
        integration_steps: int = 200,
        n_hidden_layers: int = 3,
        ema_decay: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        time_dist: str = "uniform",
        activation: str = "elu",
    ) -> None:
        """
        Initialize CTSM estimator.

        Args mostly inherited from base. ema_decay (default None) enables EMA
        of model parameters for inference; if set, must be in (0, 1).
        grad_clip_norm (default None) clips gradient norm at the given value
        before each optimizer step; None disables clipping.
        time_dist: importance sampling time distribution. in {"uniform", "beta_2_2",
        "beta_5_5"}; default "uniform" preserves current behavior.
        activation: score network activation function {"elu", "gelu", "silu"};
        default "elu" preserves byte-identical behavior.
        """
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.sigma = sigma
        self.eps = eps
        self.integration_steps = integration_steps
        self.n_hidden_layers = n_hidden_layers
        self.ema_decay = ema_decay
        self.grad_clip_norm = grad_clip_norm
        if time_dist not in {"uniform", "beta_2_2", "beta_5_5"}:
            raise ValueError(
                f"time_dist must be in {{'uniform', 'beta_2_2', 'beta_5_5'}}; "
                f"got {time_dist!r}"
            )
        self.time_dist = time_dist
        if activation not in ("elu", "gelu", "silu"):
            raise ValueError(
                f"activation must be in {{'elu', 'gelu', 'silu'}}; got {activation!r}"
            )
        self.activation = activation
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        self.ema: Optional[EMA] = None

    def init_model(self) -> None:
        """
        Initialize neural network and optimizer.

        Instantiate TimeScoreNetwork1D on device, create Adam optimizer.
        """
        self.model = TimeScoreNetwork1D(self.input_dim, self.hidden_dim, n_hidden_layers=self.n_hidden_layers, activation=self.activation).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )
        self.ema = EMA(self.model, self.ema_decay) if self.ema_decay is not None else None

    def _epsilon_target(
        self,
        epsilon: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute closed-form CTSM target and weight for Schrodinger Bridge path.

        SB path: x_t = (1-t)*x0 + t*x1 + \sigma*\sqrt{t(1-t)}*\epsilon

        All operations element-wise, outputs shape [B, 1]:
        - var = \sigma^2 * t * (1-t)
        - std = \sigma * \sqrt{t * (1-t)}
        - d_var = \sigma^2 * (1 - 2*t)
        - delta = x1 - x0  (shape [B, dim])
        - delta_sq = sum(delta^2, dim=-1, keepdim=True)
        - eps_sq = sum(\epsilon^2, dim=-1, keepdim=True)
        - delta_dot_eps = sum(delta * \epsilon, dim=-1, keepdim=True)
        - temp = \sqrt{2 * delta_sq + 1e-8}  (numerical stability)
        - lambda_t = var / temp
        - target = (d_var * (eps_sq - dim) / 2 + std * delta_dot_eps) / temp

        Returns: tuple (lambda_t, target) both of shape [B, 1].
        """
        # [B, 1]
        var = self.sigma**2 * t * (1 - t)
        std = self.sigma * torch.sqrt(t * (1 - t))
        d_var = self.sigma**2 * (1 - 2 * t)

        # [B, dim]
        delta = x1 - x0
        # [B, 1]
        delta_sq = torch.sum(delta**2, dim=-1, keepdim=True)
        eps_sq = torch.sum(epsilon**2, dim=-1, keepdim=True)
        delta_dot_eps = torch.sum(delta * epsilon, dim=-1, keepdim=True)

        dim = epsilon.shape[-1]
        # [B, 1]
        temp = torch.sqrt(2 * delta_sq + 1e-8)
        lambda_t = var / temp
        target = (d_var * (eps_sq - dim) / 2 + std * delta_dot_eps) / temp

        return lambda_t, target

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """
        Train CTSM on paired samples from p0 and p1.

        Initialize model, loop epochs: sample batch indices, extract x0 and x1,
        sample t and epsilon, construct SB path, compute target via _epsilon_target,
        forward pass, MSE loss, backprop.
        """
        self.init_model()
        self.model.train()

        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()
        n0 = samples_p0.shape[0]
        n1 = samples_p1.shape[0]

        samples_p0 = samples_p0.to(self.device)
        samples_p1 = samples_p1.to(self.device)

        for _ in range(self.n_epochs):
            # sample batch indices with replacement
            idx0 = torch.randint(0, n0, (self.batch_size,))
            idx1 = torch.randint(0, n1, (self.batch_size,))

            x0 = samples_p0[idx0]  # [B, dim]
            x1 = samples_p1[idx1]  # [B, dim]

            # sample t with importance weighting
            t, iw = sample_time_and_iw(self.time_dist, self.batch_size, self.eps, self.device)  # [B, 1], [B, 1]
            epsilon = torch.randn_like(x0)  # [B, dim]

            # construct SB path
            x_t = (1 - t) * x0 + t * x1 + self.sigma * torch.sqrt(t * (1 - t)) * epsilon  # [B, dim]

            # compute target
            lambda_t, target = self._epsilon_target(epsilon, x0, x1, t)  # [B, 1], [B, 1]

            # forward pass
            pred = self.model(x_t, t)  # [B, 1]

            # loss with importance weighting
            err = target - lambda_t * pred  # [B, 1]
            mse_loss = torch.mean(iw * (err ** 2))

            # backprop
            self.optimizer.zero_grad()
            mse_loss.backward()
            maybe_clip_grad(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model)

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """Estimate log density ratio via trapezoidal quadrature.

        Procedure:
            - verify model is fitted, set eval mode, move xs to device.
            - build uniform tau grid of self.integration_steps points in [eps, 1-eps].
            - evaluate -score(xs, tau) at each grid point (batched over xs).
            - return torch.trapezoid along the tau axis.

        Returns:
            log density ratios as a 1D CPU tensor [N].
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        xs = xs.to(self.device)
        n = xs.shape[0]

        # if EMA is active, evaluate with the shadow weights and restore on exit
        if self.ema is not None:
            self.ema.apply_to(self.model)
        try:
            ts = torch.linspace(self.eps, 1.0 - self.eps, self.integration_steps, device=self.device)
            with torch.no_grad():
                vals = torch.stack([
                    -self.model(xs, torch.full((n, 1), float(t.item()), device=self.device)).squeeze(-1)
                    for t in ts
                ])
            dt = (1.0 - 2.0 * self.eps) / (self.integration_steps - 1)
            return torch.trapezoid(vals, dx=dt, dim=0).cpu()
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)


if __name__ == "__main__":
    from torch.distributions import MultivariateNormal

    from experiments.utils.prescribed_kls import create_two_gaussians_kl

    DIM = 2
    NSAMPLES_TRAIN = 10000
    NSAMPLES_TEST = 10
    KL_DIVERGENCE = 5

    # === CREATE SYNTHETIC DATA ===
    gaussian_pair = create_two_gaussians_kl(DIM, KL_DIVERGENCE, beta=0.5)
    mu0, Sigma0 = gaussian_pair["mu0"], gaussian_pair["Sigma0"]
    mu1, Sigma1 = gaussian_pair["mu1"], gaussian_pair["Sigma1"]
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))
    samples_test = p0.sample((NSAMPLES_TEST,))

    # === TRAIN AND EVALUATE ===
    ctsm = CTSM(DIM)
    ctsm.fit(samples_p0, samples_p1)

    est_ldrs = ctsm.predict_ldr(samples_test)
    true_ldrs = p0.log_prob(samples_test) - p1.log_prob(samples_test)
    mae = torch.mean(torch.abs(est_ldrs - true_ldrs))
    print(f"MAE: {mae}")
