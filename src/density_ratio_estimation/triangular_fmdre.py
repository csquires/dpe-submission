"""flow matching density ratio estimator with triangular p_* trajectory.

estimates log p0(x) / p1(x) via conditional flow matching with three samples
(p0, p1, p*). the estimator trains a velocity field on all three sample sets,
using p* as an intermediate reference point (e.g., geometric-mean Gaussian).
at inference, integrates a ratio ode to estimate log density ratios.

reference: arXiv:2602.24201 (triangular setting with p* trajectory).
"""

from typing import Optional
import torch
from torch import Tensor

from src.density_ratio_estimation.base import DensityRatioEstimator
from src.models.flow.multiclass_vel_score_mlp import MultiClassVelScoreMLP
from src.models.flow.train_triangular import train_triangular_flow
from src.models.flow.ratio_ode import ratio_ode_triangular


class TriangularFMDRE(DensityRatioEstimator):
    """flow matching density ratio estimator with triangular p_* trajectory.

    unlike FMDRE (S1) which trains on two samples (p0, p1) with a binary condition,
    TriangularFMDRE trains on three samples (p0, p1, p*) with a ternary condition,
    where p* is an intermediate reference point. this enables more flexible density
    ratio estimation via a three-class velocity field.

    hyperparameters control training dynamics (n_epochs, batch_size, lr), model
    capacity (hidden_dim), and inference precision (integration_steps, eps).
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
        div_method: str = "hutch_rademacher",
        verbose: bool = False,
        log_every: int = 100,
        n_hidden_layers: int = 3,
        adam_betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.0,
        cosine_min_factor: float = 1.0,
        triangular_p_uncond: float = 0.0,
        layernorm: str = "off",
    ) -> None:
        """initialize triangular fmdre estimator with hyperparameters.

        procedure:
          1. call super().__init__(input_dim) to store input dimension
          2. store all hyperparameters as instance attributes
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
            div_method: divergence estimation method (default 'hutch_rademacher'); one of {'exact', 'hutch_gaussian', 'hutch_rademacher'}
            verbose: print training progress (default False)
            log_every: epoch interval for verbose logging (default 100)
            n_hidden_layers: number of hidden layers in velocity mlp (default 3)
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
        self.verbose = verbose
        self.log_every = log_every
        self.n_hidden_layers = n_hidden_layers
        self.adam_betas = tuple(adam_betas)
        self.weight_decay = float(weight_decay)
        if not (0.0 <= cosine_min_factor <= 1.0):
            raise ValueError(f"cosine_min_factor must be in [0, 1], got {cosine_min_factor}")
        self.cosine_min_factor = float(cosine_min_factor)
        if not (0.0 <= triangular_p_uncond <= 1.0):
            raise ValueError(f"triangular_p_uncond must be in [0, 1], got {triangular_p_uncond}")
        self.triangular_p_uncond = float(triangular_p_uncond)
        if layernorm not in ("off", "pre", "post"):
            raise ValueError(f"layernorm must be in {{'off', 'pre', 'post'}}; got {layernorm!r}")
        self.layernorm = layernorm

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None

    def init_model(self) -> None:
        """instantiate MultiClassVelScoreMLP with K=3.

        procedure:
          1. create MultiClassVelScoreMLP(input_dim, num_classes=3, hidden_dim, n_hidden_layers)
          2. move model to device
          3. store in self.model
        """
        self.model = MultiClassVelScoreMLP(
            self.input_dim,
            num_classes=3,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=self.n_hidden_layers,
            layernorm=self.layernorm,
        ).to(self.device)

    def fit(
        self,
        samples_p0: Tensor,
        samples_p1: Tensor,
        samples_pstar: Tensor
    ) -> None:
        """train the velocity field on samples from p0, p1, and p*.

        procedure:
          1. call init_model() to instantiate the model
          2. convert all inputs to float32
          3. call train_triangular_flow with all hyperparameters
          4. print completion message if verbose (prefix "[TriangularFMDRE]")
          5. set model to eval mode

        args:
            samples_p0: [N, D] samples from p0
            samples_p1: [N, D] samples from p1
            samples_pstar: [N, D] samples from p* (intermediate reference)

        note: this method extends the base class fit(samples_p0, samples_p1)
              with an additional samples_pstar argument. this is project convention
              for triangular estimators (TriangularCTSM, TriangularVFM).
        """
        self.init_model()

        samples_p0 = samples_p0.float()
        samples_p1 = samples_p1.float()
        samples_pstar = samples_pstar.float()

        self.model = train_triangular_flow(
            self.model,
            samples_p0,
            samples_p1,
            samples_pstar,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            score_weight=self.score_weight,
            eps=self.eps,
            device=str(self.device),
            verbose=self.verbose,
            log_every=self.log_every,
            adam_betas=self.adam_betas,
            weight_decay=self.weight_decay,
            cosine_min_factor=self.cosine_min_factor,
            triangular_p_uncond=self.triangular_p_uncond,
        )

        if self.verbose:
            print("[TriangularFMDRE] Training complete")

        self.model.eval()

    def predict_ldr(self, xs: Tensor) -> Tensor:
        """estimate log p0(x) / p1(x) via ratio ODE with triangular trajectory.

        procedure:
          1. validate model is fitted (raise RuntimeError if not)
          2. set model to eval mode
          3. move samples to device and convert to float32
          4. call ratio_ode_triangular with hyperparameters
          5. return result on CPU, detached

        args:
            xs: [B, D] query points

        returns:
            [B] log density ratio tensor on CPU

        raises:
            RuntimeError: if model is not trained (self.model is None)
        """
        if self.model is None:
            raise RuntimeError(
                "TriangularFMDRE model is not trained. Call fit() before predict_ldr()."
            )

        self.model.eval()

        samples = xs.float().to(self.device)

        ldr = ratio_ode_triangular(
            self.model,
            samples,
            steps=self.integration_steps,
            eps=self.eps,
            device=str(self.device),
            div_method=self.div_method,
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

    # === GEOMETRIC-MEAN REFERENCE ===
    mu_star = 0.5 * (mu0 + mu1)
    prec_star = 0.5 * (torch.linalg.inv(Sigma0) + torch.linalg.inv(Sigma1))
    Sigma_star = torch.linalg.inv(prec_star)
    pstar = MultivariateNormal(mu_star, covariance_matrix=Sigma_star)

    # === SAMPLE ===
    samples_p0 = p0.sample((NSAMPLES_TRAIN,))
    samples_p1 = p1.sample((NSAMPLES_TRAIN,))
    samples_pstar = pstar.sample((NSAMPLES_TRAIN,))
    samples_test = p0.sample((NSAMPLES_TEST,))

    # === TRUE LDR ===
    true_ldrs = p0.log_prob(samples_test) - p1.log_prob(samples_test)
    print(f"True LDR range: [{true_ldrs.min().item():.4f}, {true_ldrs.max().item():.4f}]")
    print(f"True LDR mean: {true_ldrs.mean().item():.4f}")
    print()

    # === TRIANGULARFMDRE TRAINING AND EVALUATION ===
    print("=" * 50)
    print("TriangularFMDRE (Flow Matching with Triangular p_* Trajectory)")
    print("=" * 50)
    estimator = TriangularFMDRE(DIM, verbose=True)
    estimator.fit(
        samples_p0.to(DEVICE),
        samples_p1.to(DEVICE),
        samples_pstar.to(DEVICE)
    )

    est_ldrs = estimator.predict_ldr(samples_test.to(DEVICE))
    true_ldrs_cpu = true_ldrs.to(DEVICE)
    mae = torch.mean(torch.abs(est_ldrs.cpu() - true_ldrs_cpu.cpu()))
    print(f"TriangularFMDRE MAE: {mae.item():.4f}")
    print(f"TriangularFMDRE LDR range: [{est_ldrs.min().item():.4f}, {est_ldrs.max().item():.4f}]")
    print()
