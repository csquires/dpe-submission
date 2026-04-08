import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms


def sample_dirichlet_weights(alpha: float, n_draws: int, seed: int | None = None) -> np.ndarray:
    """
    draw weight vectors from Dirichlet distribution for 10 MNIST classes.

    args:
        alpha: concentration parameter for Dirichlet(alpha * ones(10))
        n_draws: number of weight vectors to draw
        seed: random seed for reproducibility (default: None)

    returns:
        [n_draws, 10] numpy array where each row sums to 1.0

    procedure:
        create local rng (seeded if seed is not None).
        call rng.dirichlet(alpha * ones(10), size=n_draws).
        return result.
    """
    rng = np.random.RandomState(seed)
    return rng.dirichlet(alpha * np.ones(10), size=n_draws)


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
