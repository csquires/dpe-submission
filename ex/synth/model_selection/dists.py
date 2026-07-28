"""shared gaussian recipe + closed-form ELDR for the model_selection experiment.

the 4 test distributions p* are deterministic functions of the pair
(p0, p1): step1 samples from them, step3 needs their (mu, Sigma) to evaluate the
population ELDR E_{p*}[log p0 - log p1] in closed form. defining the recipe once
keeps the two steps in sync (step3's analytics can't drift from step1's sampling).

for gaussians the pointwise log-ratio is quadratic in x, so its expectation under
p* = N(mu_s, S_s) is exact:
  E_{p*}[log N(.;mu0,S0) - log N(.;mu1,S1)]
    = 0.5 (logdet S1 - logdet S0)
      - 0.5 [ tr(S0^{-1} S_s) + (mu_s-mu0)^T S0^{-1} (mu_s-mu0) ]
      + 0.5 [ tr(S1^{-1} S_s) + (mu_s-mu1)^T S1^{-1} (mu_s-mu1) ]
using E_{N(mu_s,S_s)}[(x-a)^T A (x-a)] = tr(A S_s) + (mu_s-a)^T A (mu_s-a).
"""
import numpy as np


def test_dist_params(mu0, S0, mu1, S1, sqrt):
    """the 4 test dists (mu*, Sigma*) as functions of p0=(mu0,S0), p1=(mu1,S1).

    `sqrt` is the elementwise sqrt of the caller's backend (torch.sqrt for step1
    sampling, np.sqrt for step3 analytics) so one recipe serves both. order is
    fixed [p0, p1, midpoint, distant] to match the stored test-set axis.
    """
    return [
        (mu0, S0),                        # pstar1 = p0
        (mu1, S1),                        # pstar2 = p1
        ((mu0 + mu1) * 0.5, sqrt(S0)),    # pstar3 = midpoint mean, sqrt(S0) cov
        (-2 * mu1, 2 * S0),               # pstar4 = distant gaussian
    ]


def analytic_eldr(mu0, S0, mu1, S1, mu_s, S_s):
    """closed-form E_{N(mu_s,S_s)}[log N(.;mu0,S0) - log N(.;mu1,S1)] (numpy, scalar)."""
    S0i = np.linalg.inv(S0)
    S1i = np.linalg.inv(S1)
    _, ld0 = np.linalg.slogdet(S0)
    _, ld1 = np.linalg.slogdet(S1)
    q0 = np.trace(S0i @ S_s) + (mu_s - mu0) @ S0i @ (mu_s - mu0)
    q1 = np.trace(S1i @ S_s) + (mu_s - mu1) @ S1i @ (mu_s - mu1)
    return 0.5 * (ld1 - ld0) - 0.5 * q0 + 0.5 * q1


def true_eldr_arr(mu0_arr, S0_arr, mu1_arr, S1_arr):
    """population ELDR for every (row, test_set): float64 array (nrows, 4)."""
    mu0_arr = np.asarray(mu0_arr, dtype=np.float64)
    S0_arr = np.asarray(S0_arr, dtype=np.float64)
    mu1_arr = np.asarray(mu1_arr, dtype=np.float64)
    S1_arr = np.asarray(S1_arr, dtype=np.float64)
    n = mu0_arr.shape[0]
    out = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        dists = test_dist_params(mu0_arr[i], S0_arr[i], mu1_arr[i], S1_arr[i], np.sqrt)
        for j, (mu_s, S_s) in enumerate(dists):
            out[i, j] = analytic_eldr(mu0_arr[i], S0_arr[i], mu1_arr[i], S1_arr[i], mu_s, S_s)
    return out
