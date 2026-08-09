"""prior construction and conjugate updates for sequential eig_boed."""
import numpy as np
import torch
import scipy.stats
from dataclasses import dataclass
from ex.utils.prescribed_eigs import compute_gaussian_eig


@dataclass
class Prior:
    """generative gaussian prior for conjugate sequential design.

    fields (except eig_star, eigengap) are float32 torch tensors of shape (d,) or (d,d).
    eig_star and eigengap are python float.
    """
    mu: torch.Tensor              # (d,) float32, zeros
    Sigma: torch.Tensor           # (d,d) float32, symmetric
    eig_star: float               # top eigenvalue's EIG under optimal design
    xi_opt: torch.Tensor          # (d,) float32, top eigenvector of Sigma
    lam: torch.Tensor             # (d,) float32, eigenvalues of Sigma (ascending order)
    eigengap: float               # lam[-1] / lam[-2]


def eigengap_ladder(n_priors: int, cfg: dict) -> list:
    """stratified eigengap targets spanning log-space.

    inputs:
    - n_priors: number of priors to generate
    - cfg: dict with keys eigengap_min (float), eigengap_max (float)

    outputs:
    - list of n_priors float eigengap targets in ascending order

    procedure:
    log-space grid between eigengap_min and eigengap_max across n_priors.
    """
    targets = np.exp(np.linspace(
        np.log(cfg["eigengap_min"]),
        np.log(cfg["eigengap_max"]),
        n_priors
    ))
    return targets.tolist()


def build_prior(config_seed: int, geometry: str, prior_idx: int, cfg: dict) -> Prior:
    """construct a single prior with stratified spectrum and (optionally) random rotation.

    inputs:
    - config_seed: random seed for rotation matrix Q (if geometry=='rot')
    - geometry: 'diag' or 'rot'
    - prior_idx: index in [0, n_priors); seeds spectrum alone
    - cfg: dict with data_dim (int), sigma2, eig_min, eig_max, num_priors, eigengap_min, eigengap_max

    outputs:
    - Prior instance with all fields populated

    procedure:
    1. spectrum seeding (prior_idx alone):
       - seed_seq = np.random.SeedSequence(prior_idx)
       - derive target eigengap from ladder
       - compute rho_min, rho_max as in eigengap_ladder()
       - lambda_max = rho_max (pinned)
       - lambda_2 = clamp(lambda_max / target_eigengap, min=rho_min)
       - lam = [rho_min, lambda_2, lambda_max]

    2. build covariance (float64 working precision):
       - Sigma_diag = diag(lam)
       - if geometry == 'rot': rotate via Q seeded from (config_seed, prior_idx)
       - Sigma = Q @ Sigma_diag @ Q.T

    3. symmetrize:
       - Sigma = 0.5 * (Sigma + Sigma.T)

    4. extract eigenvectors and eigenvalues:
       - evals, evecs = eigh(Sigma)
       - xi_opt = evecs[:, -1] (top eigenvector)
       - lam_sorted = evals (ascending)

    5. cast to float32 and populate Prior:
       - eig_star = 0.5 * log1p(lam_sorted[-1] / cfg['sigma2'])
       - assert |eig_star - cfg['eig_max']| < 1e-4
    """
    d = cfg["data_dim"]
    sigma2 = cfg["sigma2"]

    # note: d must be 3 for the spectrum layout [rho_min, lambda_2, rho_max]
    assert d == 3, f"d must be 3; got {d}"

    # spectrum seeding (prior_idx alone)
    eig_min = cfg["eig_min"]
    eig_max = cfg["eig_max"]
    rho_min = sigma2 * (np.exp(2 * eig_min) - 1)
    rho_max = sigma2 * (np.exp(2 * eig_max) - 1)

    # get target eigengap for this prior_idx
    targets = eigengap_ladder(cfg["num_priors"], cfg)
    target_eigengap = targets[prior_idx]

    # construct spectrum [rho_min, lambda_2, rho_max]
    lambda_max = rho_max
    lambda_2 = np.clip(lambda_max / target_eigengap, rho_min, rho_max)
    lam = np.array([rho_min, lambda_2, lambda_max], dtype=np.float64)

    # build covariance (float64)
    lam_f64 = torch.tensor(lam, dtype=torch.float64)
    Sigma_diag = torch.diag(lam_f64)

    if geometry == "diag":
        Sigma_f64 = Sigma_diag
    elif geometry == "rot":
        seed_rng = np.random.default_rng(
            np.random.SeedSequence([config_seed, prior_idx])
        )
        Q_np = scipy.stats.ortho_group.rvs(d, random_state=seed_rng)
        Q = torch.tensor(Q_np, dtype=torch.float64)
        Sigma_f64 = Q @ Sigma_diag @ Q.T
    else:
        raise ValueError(f"geometry must be 'diag' or 'rot'; got {geometry}")

    # symmetrize
    Sigma_f64 = 0.5 * (Sigma_f64 + Sigma_f64.T)

    # extract eigenvectors and eigenvalues
    evals, evecs = torch.linalg.eigh(Sigma_f64)
    evals_np = evals.detach().numpy()
    xi_opt_f64 = evecs[:, -1]

    # cast to float32
    mu = torch.zeros(d, dtype=torch.float32)
    Sigma_f32 = Sigma_f64.to(torch.float32)
    xi_opt_f32 = xi_opt_f64.to(torch.float32)
    lam_f32 = torch.tensor(evals_np, dtype=torch.float32)

    # compute eig_star
    eig_star = 0.5 * np.log1p(evals_np[-1] / sigma2)
    assert abs(eig_star - cfg["eig_max"]) < 1e-4, (
        f"eig_star {eig_star:.6f} does not match cfg eig_max {cfg['eig_max']:.6f}"
    )

    # compute eigengap
    eigengap = float(evals_np[-1] / evals_np[-2])

    return Prior(
        mu=mu,
        Sigma=Sigma_f32,
        eig_star=eig_star,
        xi_opt=xi_opt_f32,
        lam=lam_f32,
        eigengap=eigengap,
    )


def posterior_update(
    mu: torch.Tensor,
    Sigma: torch.Tensor,
    xi: torch.Tensor,
    y_obs: float,
    sigma2: float,
) -> tuple:
    """conjugate update for a linear gaussian model with scalar observation.

    inputs:
    - mu: (d,) prior mean (float32)
    - Sigma: (d,d) prior covariance (float32, symmetric)
    - xi: (d,) design vector (float32)
    - y_obs: observed scalar outcome (python float)
    - sigma2: likelihood variance (python float)

    outputs:
    - (mu_new, Sigma_new): posterior mean and covariance (both float32)

    procedure:
    1. cast to float64 for numerical stability
    2. compute posterior covariance via precision form:
       - Sigma_inv = inv(Sigma)
       - P = Sigma_inv + outer(xi,xi)/sigma2
       - Sigma_new = inv(P)
    3. symmetrize
    4. compute posterior mean:
       - mu_new = Sigma_new @ (Sigma_inv @ mu + xi * y_obs / sigma2)
    5. cast to float32

    Sigma_new depends only on Sigma and xi (via precision) and does NOT
    depend on y_obs. this independence drives why the design sequence
    alone determines the whole EIG trajectory.
    """
    # cast to float64
    Sigma_f64 = Sigma.double()
    mu_f64 = mu.double()
    xi_f64 = xi.double()

    # compute posterior covariance
    Sigma_inv = torch.linalg.inv(Sigma_f64)
    update_term = torch.outer(xi_f64, xi_f64) / sigma2
    P = Sigma_inv + update_term
    Sigma_new_f64 = torch.linalg.inv(P)

    # symmetrize
    Sigma_new_f64 = 0.5 * (Sigma_new_f64 + Sigma_new_f64.T)

    # compute posterior mean
    mu_new_f64 = Sigma_new_f64 @ (Sigma_inv @ mu_f64 + xi_f64 * y_obs / sigma2)

    # cast to float32
    mu_new = mu_new_f64.float()
    Sigma_new = Sigma_new_f64.float()

    return (mu_new, Sigma_new)


def eig_true(Sigma: torch.Tensor, xi: torch.Tensor, sigma2: float) -> float:
    """expected information gain of design xi under prior Sigma.

    inputs:
    - Sigma: (d,d) covariance (float32)
    - xi: (d,) design (float32)
    - sigma2: likelihood variance (float)

    outputs:
    - python float

    formula: 0.5 * log1p(xi^T Sigma xi / sigma2)
    """
    rho = xi @ Sigma @ xi
    eig = 0.5 * torch.log1p(rho / sigma2)
    return float(eig)
