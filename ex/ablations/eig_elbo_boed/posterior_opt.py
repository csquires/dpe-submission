"""1-D tempered-posterior search and posterior-approximation error.

optimize_alpha runs a GPSampler-driven optuna study over a scalar $\\alpha \\in
[\\alpha_{lo}, \\alpha_{hi}]$, scoring each candidate by the DRE-ELBO of the
fractional posterior $q_\\alpha$, and selects $\\hat\\alpha$ via the honest
protocol (argmax over finite-scoring trials, fallback to $\\alpha=1$). the
study's journal, sampler seed, and per-trial seeds all live on the ALPHA_TAG
offset stream (rnd=round_idx, trial>=ALPHA_TAG) so they never collide with the
design-search trials [0, n_trials) run in the same round.

post_kl computes the closed-form KL divergence between the tempered posterior
and the exact (alpha=1) conjugate update of the same pre-round belief.
"""
import math
import os
import time

import numpy as np
import optuna
from optuna.samplers import GPSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

import torch

from ex.ablations.eig_boed import sim
from ex.ablations.eig_boed.study import reseed_sampler
from ex.ablations.eig_elbo_boed import elbo_dre
from ex.utils.fractional_posterior import get_fractional_posterior

ALPHA_TAG = 100_000


def optimize_alpha(belief_mu, belief_Sigma, xi_r, y_obs, method, hp, cfg, device, *,
                    cell, round_idx, journal_path):
    """select $\\hat\\alpha$ maximizing DRE-ELBO of $q_\\alpha$, return the updated belief.

    pseudocode:
      cell -> (geometry, prior_idx, seed_rep); cs = cfg["config_seed"]
      gp_seed = sim.derive_seed(cs, geometry, prior_idx, method, seed_rep,
        rnd=round_idx, trial=ALPHA_TAG)
      study = GPSampler(gp_seed)-driven optuna study on journal_path, load_if_exists,
        enqueue linspace(alpha_lo, alpha_hi, n_startup_alpha) startup grid with the
        nearest point snapped to 1.0 (guarantees the exact update is always tried;
        alpha=1 is interior when alpha_hi > 1)
      objective(trial):
        reseed_sampler(sampler, trial_seed) FIRST, trial_seed at
          trial=ALPHA_TAG+1+trial.number
        alpha = trial.suggest_float("alpha", alpha_lo, alpha_hi)
        sample_seed at trial=ALPHA_TAG+50_000+trial.number
        (p0, p1, pstar) = elbo_dre.elbo_samples(..., alpha, sigma2, nsamples,
          sample_seed, device)
        res = elbo_dre.fit_elbo(method, hp, p0, p1, pstar, input_dim=d+1, seed=
          sample_seed); record {trial_idx, alpha_val, elbo_est, state, walltime_s}
        return res.value (nan/raise on failure, caught by study.optimize)
      alpha_hat = argmax elbo_est over finite trials; else alpha_hat=1.0, fell_back=True
      (mu_q, Sigma_q) = get_fractional_posterior(belief_mu, belief_Sigma, xi_r, y_obs,
        alpha_hat, sigma=sqrt(cfg["sigma2"])), computed in float64, cast to float32

    args:
      belief_mu: (d,) torch.Tensor float32, pre-round belief mean.
      belief_Sigma: (d,d) torch.Tensor float32, pre-round belief covariance.
      xi_r: (d,) torch.Tensor, design selected this round.
      y_obs: float or 0-d torch.Tensor, observed outcome at xi_r.
      method: str, DRE method name.
      hp: dict, method hyperparameters (resolve_hp output).
      cfg: dict with keys alpha_lo, alpha_hi (>= 1.0; > 1 permits over-tempering), n_startup_alpha,
        n_trials_alpha, sigma2, nsamples, config_seed.
      device: str, torch device.
      cell: tuple (arm, geometry, prior_idx, method, seed_rep).
      round_idx: int, current round r.
      journal_path: str, per-round alpha journal file (caller-owned; not derived here).

    returns:
      alpha_hat: float in [alpha_lo, alpha_hi], selected tempering exponent.
      mu_q: (d,) torch.Tensor float32, belief mean at alpha_hat.
      Sigma_q: (d,d) torch.Tensor float32, belief covariance at alpha_hat.
      alpha_records: list[dict], keys {trial_idx, alpha_val, elbo_est, state, walltime_s}.
      fell_back: bool, True iff no trial produced a finite elbo_est.
    """
    _, geometry, prior_idx, _, seed_rep = cell
    cs = cfg["config_seed"]

    alpha_lo = cfg["alpha_lo"]
    alpha_hi = cfg["alpha_hi"]
    # alpha_hi >= 1 so alpha=1 (the exact update) is reachable. alpha_hi > 1 permits
    # over-tempering (over-confident, still a proper PD posterior for finite alpha),
    # making alpha=1 an INTERIOR optimum rather than a boundary one.
    assert alpha_hi >= 1.0, f"alpha_hi must be >= 1.0 (alpha=1 is the exact update), got {alpha_hi}"
    n_startup_alpha = cfg["n_startup_alpha"]
    T_alpha = cfg["n_trials_alpha"]
    sigma2 = cfg["sigma2"]
    nsamples = cfg["nsamples"]
    d = belief_mu.shape[0]

    gp_seed = sim.derive_seed(cs, geometry, prior_idx, method, seed_rep,
                               rnd=round_idx, trial=ALPHA_TAG)
    sampler = GPSampler(seed=gp_seed, n_startup_trials=n_startup_alpha)
    # per-CELL resume (shard skip-if-done): start each round's alpha study fresh so
    # a requeued same-job attempt never inherits stale/failed trials.
    if os.path.exists(journal_path):
        os.remove(journal_path)
    storage = JournalStorage(JournalFileBackend(journal_path))
    study = optuna.create_study(
        study_name=f"alpha_r{round_idx}",
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    # startup grid spans [alpha_lo, alpha_hi]; snap the nearest point to exactly
    # alpha=1.0 so the exact update is always tried (alpha=1 is interior).
    startup_grid = np.linspace(alpha_lo, alpha_hi, n_startup_alpha)
    startup_grid[np.argmin(np.abs(startup_grid - 1.0))] = 1.0
    existing = {t.number for t in study.get_trials(deepcopy=False, states=None)}
    for i in range(n_startup_alpha):
        if i not in existing:
            study.enqueue_trial({"alpha": float(startup_grid[i])})

    records = {}  # trial.number -> alpha_records entry

    def objective(trial):
        # reseed-before-suggest is load-bearing for resume determinism (see docstring)
        trial_seed = sim.derive_seed(cs, geometry, prior_idx, method, seed_rep,
                                      rnd=round_idx, trial=ALPHA_TAG + 1 + trial.number)
        reseed_sampler(sampler, trial_seed)

        alpha = trial.suggest_float("alpha", alpha_lo, alpha_hi)
        sample_seed = sim.derive_seed(cs, geometry, prior_idx, method, seed_rep,
                                       rnd=round_idx, trial=ALPHA_TAG + 50_000 + trial.number)

        t0 = time.perf_counter()
        try:
            p0, p1, pstar = elbo_dre.elbo_samples(
                belief_mu, belief_Sigma, xi_r, y_obs, alpha, sigma2, nsamples,
                sample_seed, device)
            res = elbo_dre.fit_elbo(method, hp, p0, p1, pstar, input_dim=d + 1,
                                     device=device, seed=sample_seed)
            elapsed = time.perf_counter() - t0

            if not res.ok:
                records[trial.number] = {
                    "trial_idx": trial.number, "alpha_val": float(alpha),
                    "elbo_est": float("nan"), "state": res.error,
                    "walltime_s": elapsed,
                }
                raise RuntimeError(res.error or "fit_elbo not ok")

            elbo_val = float(res.value)

        except Exception as e:
            elapsed = time.perf_counter() - t0
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            if trial.number not in records:
                records[trial.number] = {
                    "trial_idx": trial.number, "alpha_val": float(alpha),
                    "elbo_est": float("nan"), "state": f"exception:{type(e).__name__}",
                    "walltime_s": elapsed,
                }
            raise

        records[trial.number] = {
            "trial_idx": trial.number, "alpha_val": float(alpha),
            "elbo_est": elbo_val, "state": "COMPLETE", "walltime_s": elapsed,
        }
        return elbo_val

    n_have = len([t for t in study.get_trials(deepcopy=False, states=None)
                  if t.state != TrialState.WAITING])
    study.optimize(objective, n_trials=max(0, T_alpha - n_have), gc_after_trial=True,
                    catch=(RuntimeError, ValueError))

    # rebuild records from study.trials so every trial (incl. those that failed
    # before recording) contributes a row, mirroring eig_boed/study.py's pattern
    alpha_records = []
    for t in study.trials:
        rec = records.get(t.number)
        if rec is None:
            rec = {"trial_idx": t.number, "alpha_val": float("nan"),
                    "elbo_est": float("nan"), "state": str(t.state), "walltime_s": 0.0}
        alpha_records.append(rec)

    # honest protocol: argmax over finite elbo_est only; fallback to alpha=1.0
    finite = [r for r in alpha_records if np.isfinite(r["elbo_est"])]
    if finite:
        best = max(finite, key=lambda r: r["elbo_est"])
        alpha_hat = float(best["alpha_val"])
        fell_back = False
    else:
        alpha_hat = 1.0
        fell_back = True

    mu_f64 = belief_mu.to(dtype=torch.float64)
    Sigma_f64 = belief_Sigma.to(dtype=torch.float64)
    xi_col_f64 = xi_r.to(dtype=torch.float64).reshape(-1, 1)
    y_val = y_obs.item() if torch.is_tensor(y_obs) else float(y_obs)
    y_f64 = torch.tensor(y_val, dtype=torch.float64, device=device)

    mu_q_f64, Sigma_q_f64 = get_fractional_posterior(
        mu_f64, Sigma_f64, xi_col_f64, y_f64, alpha_hat, sigma=math.sqrt(sigma2))

    mu_q = mu_q_f64.to(dtype=torch.float32)
    Sigma_q = Sigma_q_f64.to(dtype=torch.float32)

    return alpha_hat, mu_q, Sigma_q, alpha_records, fell_back


def post_kl(mu_a, Sigma_a, mu_1, Sigma_1):
    """closed-form $D_{KL}(\\mathcal N(\\mu_a,\\Sigma_a) \\Vert \\mathcal N(\\mu_1,\\Sigma_1))$.

    fixed $d=3$; cholesky-based logdet and inverse for numerical stability, with
    an eigh+clamp fallback if either covariance is near-singular.

    pseudocode:
      cast mu_a, Sigma_a, mu_1, Sigma_1 to float64; symmetrize both covariances
      L_1 = cholesky(Sigma_1) [fallback: V,w = eigh(Sigma_1), L_1 = V @ diag(sqrt(w))]
      logdet_1 = 2*sum(log(diag(L_1))); Sigma_1_inv = cholesky_inverse(L_1)
      L_a = cholesky(Sigma_a) [same fallback]; logdet_a = 2*sum(log(diag(L_a)))
      KL = 0.5*[tr(Sigma_1_inv Sigma_a) + (mu_1-mu_a)^T Sigma_1_inv (mu_1-mu_a)
                - d + logdet_1 - logdet_a]

    args:
      mu_a: (3,) torch.Tensor, mean of distribution a (the tempered posterior).
      Sigma_a: (3,3) torch.Tensor, covariance of distribution a.
      mu_1: (3,) torch.Tensor, mean of the reference (exact, alpha=1) posterior.
      Sigma_1: (3,3) torch.Tensor, covariance of the reference posterior.

    returns:
      float, the KL divergence.
    """
    d = 3
    dtype = torch.float64
    mu_a = mu_a.to(dtype=dtype)
    Sigma_a = Sigma_a.to(dtype=dtype)
    mu_1 = mu_1.to(dtype=dtype)
    Sigma_1 = Sigma_1.to(dtype=dtype)

    Sigma_a = 0.5 * (Sigma_a + Sigma_a.T)
    Sigma_1 = 0.5 * (Sigma_1 + Sigma_1.T)

    try:
        L_1 = torch.linalg.cholesky(Sigma_1)
    except RuntimeError:
        w_1, V_1 = torch.linalg.eigh(Sigma_1)
        w_1 = torch.clamp(w_1, min=1e-12)
        L_1 = V_1 @ torch.diag(torch.sqrt(w_1))

    logdet_1 = 2.0 * torch.sum(torch.log(torch.clamp(torch.diag(L_1), min=1e-12)))
    Sigma_1_inv = torch.cholesky_inverse(L_1)

    try:
        L_a = torch.linalg.cholesky(Sigma_a)
    except RuntimeError:
        w_a, V_a = torch.linalg.eigh(Sigma_a)
        w_a = torch.clamp(w_a, min=1e-12)
        L_a = V_a @ torch.diag(torch.sqrt(w_a))

    logdet_a = 2.0 * torch.sum(torch.log(torch.clamp(torch.diag(L_a), min=1e-12)))

    term_tr = torch.trace(Sigma_1_inv @ Sigma_a)
    delta_mu = mu_1 - mu_a
    term_quad = delta_mu @ Sigma_1_inv @ delta_mu

    kl = 0.5 * (term_tr + term_quad - float(d) + logdet_1 - logdet_a)
    return float(kl)
