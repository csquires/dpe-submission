"""ELBO scoring for alpha-tempered posterior via DRE.

two functions:
  - elbo_samples: draw (p0, p1, pstar) samples from the gaussian-linear model.
    q_\\alpha = get_fractional_posterior(..., alpha, sigma=sqrt(sigma2)).
    p0, p1 are used for DRE training; pstar is used for the elbo reduction.
  - fit_elbo: safe fit+predict with error handling (never raises).
    mirrors frozen.fit_eldr but scores via mean(predict_ldr(pstar)).
    returns FitResult(ok, value, error).

imports from:
  ex.ablations.eig_boed.sim (simulate)
  ex.utils.fractional_posterior (get_fractional_posterior)
  ex.utils.hpo.frozen (FitResult, METHOD_ALIAS, METHOD_SPECS, normalize_hp)
"""

import math

import numpy as np
import torch

from ex.ablations.eig_boed import sim
from ex.utils.fractional_posterior import get_fractional_posterior
from ex.utils.hpo.frozen import FitResult, METHOD_ALIAS, METHOD_SPECS, normalize_hp


def elbo_samples(
    belief_mu: torch.Tensor,
    belief_Sigma: torch.Tensor,
    xi: torch.Tensor,
    y_obs: torch.Tensor,
    alpha: float,
    sigma2: float,
    n: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """draw (p0, p1, pstar) samples for ELBO estimation via likelihood ratio.

    procedure:
      xi --> xi_col (d,1) --> q_\\alpha = get_fractional_posterior(belief, xi_col,
        y_obs, alpha, sigma=\\sqrt{sigma2})
      seed --> 4 independent sub-seeds (s0..s3) via np.random.SeedSequence
      p0     = [theta0 ~ belief, y0 = theta0 @ xi + noise]        (paired draw, s0)
      p1     = [theta1 ~ q_\\alpha, y1 ~ belief predictive]       (independent, s1/s2)
      pstar  = [theta_star ~ q_\\alpha, y = y_obs broadcast]      (s3)

    args:
        belief_mu: (d,), prior mean.
        belief_Sigma: (d, d), prior covariance.
        xi: (d,) or (d, 1), design vector.
        y_obs: scalar or (1,), observed outcome.
        alpha: tempering parameter in [0, 1].
        sigma2: noise variance (NOT std).
        n: number of samples.
        seed: rng seed for sub-seed derivation.
        device: torch device ("cpu" or "cuda:*").

    returns:
        (p0, p1, pstar), each (n, d+1), float32, on device.

    gotchas:
      - get_fractional_posterior's sigma arg is a STD; pass math.sqrt(sigma2).
      - alpha=0 aliases (belief_mu, belief_Sigma) as (mu_q, Sigma_q); never
        mutated in place here, so this is safe.
      - sim.simulate carries its own cholesky -> eigh+clamp fallback for
        near-singular Sigma_q; do not bypass it with a raw MultivariateNormal.
    """
    d = xi.shape[0]
    xi_col = xi.reshape(d, 1)

    y_obs_val = y_obs.item() if torch.is_tensor(y_obs) else float(y_obs)
    mu_q, Sigma_q = get_fractional_posterior(
        belief_mu, belief_Sigma, xi_col, y_obs_val, alpha, sigma=math.sqrt(sigma2)
    )

    # derive 4 independent sub-seeds from seed
    sub_seeds = [
        int(np.random.SeedSequence([seed, k]).generate_state(1)[0]) for k in range(4)
    ]
    s0, s1, s2, s3 = sub_seeds

    # p0: belief-joint, paired draw
    theta0, y0 = sim.simulate(belief_mu, belief_Sigma, xi_col, sigma2, n, s0, device)  # (n,d),(n,1)
    p0 = torch.cat([theta0, y0], dim=-1)  # (n, d+1)

    # p1: q_alpha theta paired with an independent belief-predictive y
    theta1, _ = sim.simulate(mu_q, Sigma_q, xi_col, sigma2, n, s1, device)  # (n,d)
    _, y1 = sim.simulate(belief_mu, belief_Sigma, xi_col, sigma2, n, s2, device)  # (n,1)
    p1 = torch.cat([theta1, y1], dim=-1)  # (n, d+1)

    # pstar: q_alpha theta paired with the fixed observation
    theta_star, _ = sim.simulate(mu_q, Sigma_q, xi_col, sigma2, n, s3, device)  # (n,d)
    y_star = torch.full((n, 1), y_obs_val, dtype=torch.float32, device=device)  # (n,1)
    pstar = torch.cat([theta_star, y_star], dim=-1)  # (n, d+1)

    return p0, p1, pstar


def fit_elbo(
    method: str,
    hp: dict,
    p0: torch.Tensor,
    p1: torch.Tensor,
    pstar: torch.Tensor,
    *,
    input_dim: int,
    device: str,
    seed: int = None,
) -> FitResult:
    """build, fit, and score a DRE model for ELBO estimation.

    procedure (mirrors frozen.fit_eldr's build/fit dispatch and error
    convention exactly; differs only in the final reduction):
      method --> canonical (METHOD_ALIAS) --> spec = METHOD_SPECS[canonical]
      spec --> builder(**input_dim, device, num_waypoints, **normalize_hp(hp))
      model.fit(p0, p1[, pstar] if requires_pstar)
      val = mean(model.predict_ldr(pstar))   [NOT predict_eldr]
      non-finite guard --> FitResult

    args:
        method: method name (may be alias or canonical).
        hp: hyperparameters.
        p0: (n0, d+1), belief-joint samples.
        p1: (n1, d+1), q_\\alpha x belief-predictive samples.
        pstar: (n_pstar, d+1), q_\\alpha at fixed y_obs.
        input_dim: d+1 (latent dimension), keyword-only.
        device: torch device, keyword-only.
        seed: optional rng seed, keyword-only.

    returns:
        FitResult(ok, value, error):
          ok=True, value=float, error=None  -> value = mean(predict_ldr(pstar))
          ok=False, value=None, error=str   -> error in {cuda_oom,
            exception:<T>, non_finite}

    raises:
        KeyError if method/canonical is not in METHOD_SPECS (setup error,
        propagates; never caught).
    """
    canonical = METHOD_ALIAS.get(method, method)
    if canonical not in METHOD_SPECS:
        raise KeyError(f"method {method} (canonical: {canonical}) not in METHOD_SPECS")

    spec = METHOD_SPECS[canonical]
    builder = spec["builder"]
    requires_pstar = spec.get("requires_pstar", False)
    num_waypoints = spec.get("num_waypoints", None)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    builder_kwargs = {
        "input_dim": input_dim,
        "device": device,
        "num_waypoints": num_waypoints if num_waypoints is not None else 0,
        **normalize_hp(canonical, hp),
    }
    try:
        model = builder(**builder_kwargs)
    except Exception as e:
        if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return FitResult(False, None, "cuda_oom")
        return FitResult(False, None, f"exception:{type(e).__name__}")

    try:
        if requires_pstar:
            model.fit(p0, p1, pstar)
        else:
            model.fit(p0, p1)
    except Exception as e:
        if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return FitResult(False, None, "cuda_oom")
        return FitResult(False, None, f"exception:{type(e).__name__}")

    try:
        with torch.no_grad():
            val = float(torch.mean(model.predict_ldr(pstar)).item())
    except Exception as e:
        if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return FitResult(False, None, "cuda_oom")
        return FitResult(False, None, f"exception:{type(e).__name__}")

    if not math.isfinite(val):
        return FitResult(False, None, "non_finite")

    return FitResult(True, val, None)
