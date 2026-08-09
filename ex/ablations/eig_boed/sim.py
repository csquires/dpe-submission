"""gaussian-linear simulator: theta = mu + z @ L^T, y = theta @ xi + sqrt(sigma2) * e.

randomness (z, e) drawn on CPU via seeded torch.Generator, then transferred to
device. per-trial randomness is FRESH (annealed); the caller may reuse z across
a round if needed.

seed derivation from (config_seed, geometry, prior_idx, method, seed_rep, rnd,
trial) is a pure function: deterministic and independent of global
torch.random state.

determinism: NO global torch.random mutation. all randomness flows through
make_generator -> simulate's local g.
"""

import hashlib
import math
import numpy as np
import torch


def derive_seed(config_seed: int,
                geometry: str,
                prior_idx: int,
                method: str,
                seed_rep: int,
                rnd: int,
                trial: int) -> int:
    """derive a seed from identifiers; pure function of inputs.

    maps (config_seed, geometry, prior_idx, method, seed_rep, rnd, trial) to
    a 64-bit seed via SeedSequence. geometry -> geom_id (diag=0, rot=1);
    method -> hash via blake2b (not builtin hash, which is salted).

    args:
        config_seed: configuration seed (globally fixed per run).
        geometry: "diag" or "rot".
        prior_idx: index into prior ladder (0..n_priors-1).
        method: method name (e.g., "VFM", "FMDRE").
        seed_rep: seed-repeat index (0..n_seeds-1).
        rnd: round index (0..n_rounds-1).
        trial: trial index (0..n_trials-1).

    returns:
        int, a 64-bit seed from SeedSequence.
    """
    geom_id = 0 if geometry == "diag" else 1
    method_bytes = method.encode('utf-8')
    method_hash = int(
        hashlib.blake2b(method_bytes, digest_size=8).hexdigest(), 16
    ) & 0xFFFFFFFFFFFFFFFF

    # coerce all entropy elements to uint32 range [0, 2^32) via modulo.
    # handles negative sentinels (e.g., trial=-1 -> 2^32-1) while keeping
    # non-negative values in [0, 2^32) unchanged. deterministic.
    entropy = [
        int(config_seed) % (2**32),
        int(geom_id) % (2**32),
        int(prior_idx) % (2**32),
        int(method_hash) % (2**32),
        int(seed_rep) % (2**32),
        int(rnd) % (2**32),
        int(trial) % (2**32)
    ]
    ss = np.random.SeedSequence(entropy)
    return int(ss.generate_state(1)[0])


def make_generator(seed: int) -> torch.Generator:
    """create a CPU torch.Generator seeded with seed.

    args:
        seed: integer seed.

    returns:
        torch.Generator on device='cpu', with manual_seed(seed).
    """
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed))
    return g


def draw_params(mu: torch.Tensor,
                Sigma: torch.Tensor,
                seed: int,
                device: str) -> torch.Tensor:
    """draw single N(mu, Sigma) sample for theta_star.

    symmetrize Sigma, cholesky factorize (float64), draw z ~ N(0,I) on CPU via
    seeded generator, then compute theta_star = mu + z @ L^T and transfer to device.

    args:
        mu: (d,) prior mean, float32 or float64.
        Sigma: (d,d) prior covariance, float32 or float64.
        seed: seed (int), passed to make_generator.
        device: target device ("cpu" or "cuda:*").

    returns:
        theta_star: (d,) float32, on device.
    """
    d = mu.shape[0]

    # symmetrize and cholesky on CPU in float64
    Sigma_sym = 0.5 * (Sigma + Sigma.T)
    Sigma_cpu = Sigma_sym.to(dtype=torch.float64, device='cpu')

    try:
        L = torch.linalg.cholesky(Sigma_cpu)
    except RuntimeError:
        # fallback: L = V @ diag(sqrt(w)) where V, w from eigh
        w, V = torch.linalg.eigh(Sigma_cpu)
        w = torch.clamp(w, min=1e-12)
        L = V @ torch.diag(torch.sqrt(w))

    # generate z on CPU
    g = make_generator(seed)
    z = torch.randn(d, generator=g, device='cpu', dtype=torch.float64)

    # compute theta_star
    mu_cpu = mu.to(dtype=torch.float64, device='cpu')
    theta_star_cpu = mu_cpu + z @ L.T

    # cast to float32 and transfer to device; returns torch.Tensor, not numpy array
    theta_star = theta_star_cpu.to(dtype=torch.float32, device=device)

    return theta_star


def simulate(mu: torch.Tensor,
             Sigma: torch.Tensor,
             xi: torch.Tensor,
             sigma2: float,
             n: int,
             seed: int,
             device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """simulate (theta, y) from gaussian-linear model.

    theta = mu + z @ L^T, where z ~ N(0, I) and L @ L^T = Sigma;
    y = theta @ xi + sqrt(sigma2) * e, where e ~ N(0, 1).

    randomness is drawn on CPU via seeded generator, then transferred to device.
    float64 cholesky for stability, then cast to float32.

    invariant: xi used here MUST be the SAME tensor object whose eig_true
    is computed and logged by the caller. do NOT re-normalize xi.

    args:
        mu: (d,) prior mean, float32 or float64.
        Sigma: (d, d) prior covariance, float32 or float64.
        xi: (d,) or (d, 1) design vector, float64 (externally computed).
        sigma2: noise variance (float).
        n: number of samples (int).
        seed: seed (int), passed to make_generator.
        device: target device ("cpu" or "cuda:*").

    returns:
        (theta, y):
            theta: (n, d), float32, on device.
            y: (n, 1), float32, on device.

    notes:
        - Sigma is symmetrized before cholesky: (Sigma + Sigma.T) / 2.
        - cholesky condition number ~32 (spectrum ~1.7..53.6), benign.
        - eigh fallback optional if cholesky fails.
        - device mismatch guard: mu, Sigma cast to float64 on CPU, then result
          transferred to device.
        - n large (e.g., 10000): generate z, e on CPU then .to(device) once.
        - NO global torch.random mutation.
    """
    # ensure Sigma is float64 (stability) and symmetrized
    d = mu.shape[0]
    assert xi.shape[0] == d, f"xi dimension {xi.shape[0]} != mu dimension {d}"

    Sigma_sym = 0.5 * (Sigma + Sigma.T)  # symmetrize
    Sigma_cpu = Sigma_sym.to(dtype=torch.float64, device='cpu')

    # cholesky factorization; fallback to eigh if needed
    try:
        L = torch.linalg.cholesky(Sigma_cpu)
    except RuntimeError:
        # fallback: L = V @ diag(sqrt(w)) where V, w from eigh
        w, V = torch.linalg.eigh(Sigma_cpu)
        w = torch.clamp(w, min=1e-12)  # guard against near-zero eigenvalues
        L = V @ torch.diag(torch.sqrt(w))

    # generate randomness on CPU
    g = make_generator(seed)
    z = torch.randn((n, d), generator=g, device='cpu', dtype=torch.float64)
    e = torch.randn((n, 1), generator=g, device='cpu', dtype=torch.float64)

    # compute theta and y
    mu_cpu = mu.to(dtype=torch.float64, device='cpu')
    theta_cpu = mu_cpu + z @ L.T
    xi_col = xi.reshape(-1, 1).to(dtype=torch.float64, device='cpu')
    y_cpu = theta_cpu @ xi_col + math.sqrt(sigma2) * e

    # cast to float32 and transfer to device
    theta = theta_cpu.to(dtype=torch.float32, device=device)
    y = y_cpu.to(dtype=torch.float32, device=device)

    return theta, y
