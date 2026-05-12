import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from scipy.special import xlogy, polygamma, psi as digamma


def sample_dirichlet_weights(alpha: float, n_draws: int, seed: int | None = None, K: int = 10) -> np.ndarray:
    """
    draw weight vectors from Dirichlet distribution for 10 MNIST classes.

    args:
        alpha: concentration parameter for Dirichlet(alpha * ones(K))
        n_draws: number of weight vectors to draw
        seed: random seed for reproducibility (default: None)
        K: number of classes / dimensionality of Dirichlet (default: 10)

    returns:
        [n_draws, K] numpy array where each row sums to 1.0

    procedure:
        create local rng (seeded if seed is not None).
        call rng.dirichlet(alpha * ones(K), size=n_draws).
        return result.
    """
    rng = np.random.RandomState(seed)
    return rng.dirichlet(alpha * np.ones(K), size=n_draws)


def subsample_mnist(dataset, weights: np.ndarray, min_per_class: int = 10) -> list[int]:
    """
    return indices of subset where class frequencies match target weights.

    args:
        dataset: torchvision.datasets.MNIST instance
        weights: [10] numpy array of target class weights (should sum to 1.0)
        min_per_class: minimum samples required per class (default: 10)

    returns:
        list[int] of indices into dataset

    procedure:
        extract labels and build class_indices dict mapping class to list of indices.
        compute total_budget as floor of min_c(count_c / max(weights[c], 1e-10)).
        for each class, set target[c] = max(round(weights[c] * total_budget), min_per_class).
        iterate, reducing total_budget until all target[c] <= count_c.
        select first target[c] indices from class_indices[c] and concatenate.
    """
    # extract labels and build class index map
    labels = dataset.targets.numpy() if isinstance(dataset.targets, torch.Tensor) else dataset.targets
    class_indices = {c: [] for c in range(10)}
    for idx in range(len(labels)):
        c = labels[idx]
        class_indices[c].append(idx)

    # compute class counts
    count_c = np.array([len(class_indices[c]) for c in range(10)])

    # check if min_per_class is feasible
    if min_per_class * 10 > count_c.sum():
        raise ValueError(f"min_per_class={min_per_class} * 10 classes exceeds dataset size")

    # compute total budget, reducing iteratively until feasible
    total_budget = int(np.floor((count_c / np.maximum(weights, 1e-10)).min()))

    while True:
        target = np.array([max(int(np.round(weights[c] * total_budget)), min_per_class) for c in range(10)])

        # check if all targets are feasible
        if np.all(target <= count_c):
            break

        # reduce budget and retry
        total_budget -= 1

    # select and concatenate indices
    selected_indices = []
    for c in range(10):
        selected_indices.extend(class_indices[c][:target[c]])

    return selected_indices


def get_mnist_dataset(root: str = './data', train: bool = True, download: bool = True) -> torchvision.datasets.MNIST:
    """
    simple wrapper for loading MNIST with ToTensor transform.

    args:
        root: directory to cache MNIST data (default: './data')
        train: load training (True) or test (False) set (default: True)
        download: download if not already present (default: True)

    returns:
        torchvision.datasets.MNIST instance with ToTensor transform applied

    procedure:
        create transform = ToTensor().
        return MNIST(root, train=train, download=download, transform=transform).
    """
    transform = transforms.ToTensor()
    return torchvision.datasets.MNIST(root, train=train, download=download, transform=transform)


def invert_weights(w: np.ndarray) -> np.ndarray:
    """
    compute reciprocal-normalized weights w'_i = w_i^{-1} / sum_j w_j^{-1}.

    args:
        w: [K] or [n, K] numpy array of class weights

    returns:
        same shape as input, each row sums to 1.0

    procedure:
        record if 1D. reshape to [1, K] if needed.
        floor w at epsilon=1e-15.
        compute reciprocals and normalize row-wise.
        squeeze back to original shape if 1D.
    """
    is_1d = (w.ndim == 1)
    if is_1d:
        w = w.reshape(1, -1)

    w_safe = np.maximum(w, 1e-15)
    recip = 1.0 / w_safe
    result = recip / recip.sum(axis=-1, keepdims=True)

    if is_1d:
        result = result.squeeze(axis=0)

    return result


def bound_moments(alpha: float, K: int) -> dict:
    """closed-form moments of the pointwise sandwich on KL(w || invert(w)).

    pointwise bracket from semisynth appendix:
        ell(w) = 2 T1 - T2/K + log K  <=  KL(w || invert(w))  <=  2 T1 - T2 + log K = u(w)
    where T1 = sum_j w_j log w_j, T2 = sum_j log w_j, w ~ Dir(alpha 1_K).

    args:
        alpha: dirichlet concentration (alpha > 0).
        K: simplex dimension (number of classes).

    returns:
        dict with keys {E_ell, E_u, Var_ell, Var_u, E_T1, E_T2, Var_T1, Var_T2, Cov_T1_T2}.

    procedure:
        1. evaluate the digamma table at alpha + s, K alpha + s for s in {0,1,2}
           and the trigamma analogues.
        2. assemble mu1, mu2 (means of T1, T2), Var(T2), E[T1^2], Var(T1),
           E[T1 T2], Cov(T1, T2) per the appendix derivation.
        3. apply linear combinations to get E[ell], E[u], Var(ell), Var(u).
    """
    psi1_a1 = digamma(alpha + 1.0)
    psi1_Ka1 = digamma(K * alpha + 1.0)
    psi1_a2 = digamma(alpha + 2.0)
    psi1_Ka2 = digamma(K * alpha + 2.0)
    psi1_a = digamma(alpha)
    psi1_Ka = digamma(K * alpha)

    tri_a = polygamma(1, alpha)
    tri_Ka = polygamma(1, K * alpha)
    tri_a1 = polygamma(1, alpha + 1.0)
    tri_Ka1 = polygamma(1, K * alpha + 1.0)
    tri_a2 = polygamma(1, alpha + 2.0)
    tri_Ka2 = polygamma(1, K * alpha + 2.0)

    phi1 = psi1_a1 - psi1_Ka1
    phi2 = psi1_a2 - psi1_Ka2
    phi3 = psi1_a1 - psi1_Ka2
    phi4 = psi1_a - psi1_Ka1
    Delta1 = tri_a1 - tri_Ka1
    Delta2 = tri_a2 - tri_Ka2

    mu1 = phi1
    mu2 = K * (psi1_a - psi1_Ka)

    var_T2 = K * tri_a - (K ** 2) * tri_Ka

    a1 = (alpha + 1.0) / (K * alpha + 1.0)
    a2 = ((K - 1) * alpha) / (K * alpha + 1.0)
    E_T1_sq = a1 * (phi2 ** 2 + Delta2) + a2 * (phi3 ** 2 - tri_Ka2)
    var_T1 = E_T1_sq - mu1 ** 2

    E_T1_T2 = phi1 ** 2 + Delta1 + (K - 1) * (phi1 * phi4 - tri_Ka1)
    cov_T1_T2 = E_T1_T2 - mu1 * mu2

    log_K = float(np.log(K))
    E_ell = 2 * mu1 - mu2 / K + log_K
    E_u = 2 * mu1 - mu2 + log_K
    var_ell = 4 * var_T1 - (4 / K) * cov_T1_T2 + var_T2 / (K ** 2)
    var_u = 4 * var_T1 - 4 * cov_T1_T2 + var_T2

    return {
        "E_ell": float(E_ell),
        "E_u": float(E_u),
        "Var_ell": float(max(var_ell, 0.0)),
        "Var_u": float(max(var_u, 0.0)),
        "E_T1": float(mu1),
        "E_T2": float(mu2),
        "Var_T1": float(max(var_T1, 0.0)),
        "Var_T2": float(max(var_T2, 0.0)),
        "Cov_T1_T2": float(cov_T1_T2),
    }


def expected_kl_jensen_ub(alpha: float, K: int) -> float:
    """outer-Jensen upper bound on E[KL(w || invert(w))] for alpha > 1.

    E[KL] <= -2 [psi(K alpha + 1) - psi(alpha + 1)] + log(K (K alpha - 1)/(alpha - 1)),
    valid only for alpha > 1; returns +inf otherwise.
    """
    if alpha <= 1.0:
        return float('inf')
    entropy_term = digamma(K * alpha + 1.0) - digamma(alpha + 1.0)
    log_ES = float(np.log(K * (K * alpha - 1.0) / (alpha - 1.0)))
    return float(-2.0 * entropy_term + log_ES)


def weight_kl(w: np.ndarray, w_prime: np.ndarray) -> float | np.ndarray:
    """
    compute KL divergence between categorical distributions.

    KL(w || w') = sum_i w_i * log(w_i / w'_i)

    args:
        w: [K] or [n, K] numpy array of target weights
        w_prime: [K] or [n, K] numpy array of model weights

    returns:
        float if inputs are 1D, [n] array if 2D

    procedure:
        record if 1D. reshape to [1, K] if needed.
        compute ratio with clipped denominator.
        use xlogy for 0*log(0) = 0 handling.
        squeeze to scalar if 1D.
    """
    is_1d = (w.ndim == 1)
    if is_1d:
        w = w.reshape(1, -1)
        w_prime = w_prime.reshape(1, -1)

    ratio = w / np.maximum(w_prime, 1e-15)
    kl = np.sum(xlogy(w, ratio), axis=-1)

    if is_1d:
        return float(kl[0])

    return kl
