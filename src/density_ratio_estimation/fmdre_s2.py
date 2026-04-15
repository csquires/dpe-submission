"""flow matching density ratio estimator with classifier-free guidance (S2 setting).

estimates log p0(x) / p1(x) via conditional flow matching with CFG dropout.
the estimator trains a velocity field on samples from p0 and p1 using conditional
flow matching with dropout of condition labels (CFG). at inference, integrates a
ratio ode using the unconditional (neutral) trajectory to estimate log density ratios.

reference: arXiv:2602.24201 (S2 setting with neutral trajectory via unconditional field).
"""

from typing import Optional
import torch

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.flow.cond_vel_score_mlp import CondVelScoreMLP
from src.models.flow.train_conditional import train_conditional_flow
from src.models.flow.ratio_ode import ratio_ode_s2


class FMDRE_S2(DensityRatioEstimator):
    """flow matching density ratio estimator with classifier-free guidance (S2 setting).

    unlike FMDRE (S1) which follows the numerator's velocity field u_t(x|c0), FMDRE_S2
    uses a neutral trajectory via the unconditional velocity field u_t(x|uncond) to
    compute the ratio. this is achieved by training with CFG dropout (conditionally
    setting c to a sentinel value) and integrating the ratio ODE with the unconditional
    trajectory.

    hyperparameters control training dynamics (n_epochs, batch_size, lr), CFG behavior
    (p_uncond, uncond_cond), and inference precision (integration_steps, eps, chunk_size).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_epochs: int = 1000,
        batch_size: int = 512,
        lr: float = 2e-3,
        score_weight: float = 1.0,
        eps: float = 0.01,
        device: Optional[str] = None,
        integration_steps: int = 10000,
        div_method: str = "exact",
        chunk_size: int = 500,
        verbose: bool = False,
        log_every: int = 100,
        p_uncond: float = 0.5,
        uncond_cond: float = -1.0,
    ) -> None:
        """initialize fmdre_s2 estimator with hyperparameters including cfg dropout.

        procedure:
          1. call super().__init__(input_dim) to store input dimension
          2. store all hyperparameters as instance attributes (including p_uncond, uncond_cond)
          3. auto-detect device (cuda if available, else cpu)
          4. initialize model placeholder to None

        args:
            input_dim: dimensionality of data samples
            hidden_dim: width of hidden layers in velocity mlp (default 256)
            n_epochs: number of training epochs (default 1000)
            batch_size: training batch size (default 512)
            lr: adam learning rate (default 2e-3)
            score_weight: weight on score matching loss (default 1.0)
            eps: time clamping to [eps, 1-eps] (default 0.01)
            device: target device ('cuda' or 'cpu'); auto-detects if None
            integration_steps: number of ode integration steps (default 10000)
            div_method: divergence estimation method (default 'exact')
            chunk_size: batch chunking for memory efficiency (default 500)
            verbose: print training progress (default False)
            log_every: epoch interval for verbose logging (default 100)
            p_uncond: CFG dropout probability during training (default 0.5)
            uncond_cond: sentinel value for unconditional condition (default -1.0)
        """
        super().__init__(input_dim)
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.score_weight = score_weight
        self.eps = eps
        self.integration_steps = integration_steps
        self.div_method = div_method
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.log_every = log_every
        self.p_uncond = p_uncond
        self.uncond_cond = uncond_cond

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None

    def init_model(self) -> None:
        """instantiate velocity mlp on the device."""
        self.model = CondVelScoreMLP(self.input_dim, self.hidden_dim).to(self.device)

    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor) -> None:
        """train the conditional velocity field on samples from p0 and p1 with CFG dropout.

        procedure:
          1. call init_model() to instantiate the model
          2. convert inputs to float32
          3. call train_conditional_flow with cfg hyperparameters (p_uncond, sentinel_cond)
          4. print completion message if verbose (prefix "[FMDRE_S2]")
          5. set model to eval mode

        note: train_conditional_flow signature includes optional cfg parameters
              that are passed through. if the function does not support them,
              they are silently ignored.
        """
        self.init_model()

        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()

        self.model = train_conditional_flow(
            self.model,
            samples_p0,
            samples_p1,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            score_weight=self.score_weight,
            eps=self.eps,
            device=str(self.device),
            verbose=self.verbose,
            log_every=self.log_every,
            p_uncond=self.p_uncond,
            sentinel_cond=self.uncond_cond,
        )

        if self.verbose:
            print("[FMDRE_S2] Training complete")

        self.model.eval()

    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """estimate log p0(x) / p1(x) via ratio ODE integration with S2 (unconditional) trajectory.

        procedure:
          1. validate model is fitted (raise RuntimeError if not)
          2. set model to eval mode
          3. move samples to device and convert to float32
          4. call ratio_ode_s2 with cfg hyperparameters
          5. return result on CPU, detached

        args:
            xs: [B, D] query points

        returns:
            [B] log density ratio tensor on CPU
        """
        if self.model is None:
            raise RuntimeError("FMDRE_S2 model is not trained. Call fit() before predict_ldr().")

        self.model.eval()

        samples = xs.float().to(self.device)

        ldr = ratio_ode_s2(
            self.model,
            samples,
            steps=self.integration_steps,
            eps=self.eps,
            device=str(self.device),
            div_method=self.div_method,
            chunk_size=self.chunk_size,
            uncond_cond=self.uncond_cond,
            warn_uncond=False,
        )

        return ldr.detach().cpu()


if __name__ == '__main__':
    from torch.distributions import MultivariateNormal
    from experiments.utils.prescribed_kls import create_two_gaussians_kl

    DIM = 2
    NSAMPLES_TRAIN = 1000
    NSAMPLES_TEST = 1000
    KL_DIVERGENCE = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # === CREATE SYNTHETIC DATA ===
    gaussian_pair = create_two_gaussians_kl(DIM, KL_DIVERGENCE, beta=0.5)
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

    # === FMDRE_S2 TRAINING AND EVALUATION ===
    print("=" * 50)
    print("FMDRE_S2 (Flow Matching with CFG Dropout - S2 Setting)")
    print("=" * 50)
    estimator = FMDRE_S2(DIM, verbose=True, p_uncond=0.5)
    estimator.fit(samples_p0.to(DEVICE), samples_p1.to(DEVICE))

    est_ldrs = estimator.predict_ldr(samples_test.to(DEVICE))
    true_ldrs_cpu = true_ldrs.to(DEVICE)
    mae = torch.mean(torch.abs(est_ldrs.cpu() - true_ldrs_cpu.cpu()))
    print(f"FMDRE_S2 MAE: {mae.item():.4f}")
    print(f"FMDRE_S2 LDR range: [{est_ldrs.min().item():.4f}, {est_ldrs.max().item():.4f}]")
    print()
