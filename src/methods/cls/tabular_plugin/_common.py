"""shared counting + Laplace-smoothing helper for the tabular plugin estimators."""
import torch


def count_and_smooth(s: torch.Tensor, a: torch.Tensor, n_states: int, n_actions: int, alpha: float) -> torch.Tensor:
    """count (s, a) pairs and apply laplace smoothing.

    flattens (s, a) to joint index k = s * n_actions + a, histograms via bincount,
    adds laplace smoothing constant alpha, normalizes to density, and reshapes to
    tabular form [n_states, n_actions].

    args:
        s: state indices, shape [N], int64
        a: action indices, shape [N], int64
        n_states: number of states
        n_actions: number of actions
        alpha: laplace smoothing parameter (> 0)

    returns:
        d_hat: empirical density [n_states, n_actions], sums to 1.0
    """
    K = n_states * n_actions
    flat = s * n_actions + a
    count = torch.bincount(flat, minlength=K).float() + alpha
    d = count / count.sum()
    return d.reshape(n_states, n_actions)
