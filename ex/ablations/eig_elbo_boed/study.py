"""eig_elbo_boed sequential BOED study orchestrator.

channel-agnostic orchestrator for R rounds of design selection (design_select,
"dre_eig" or "analytic") followed by alpha-tempered posterior optimization
(posterior_opt, DRE-ELBO guided BO over alpha). unifies both design channels
under one flat h5 shard schema; writes atomically via _atomic_h5_write.

study_seed/reseed_sampler are reused (imported, not redefined) from
ex.ablations.eig_boed.study; select_design and
optimize_alpha derive their own trial-level seeds internally from
(cell, round_idx, cfg["config_seed"]) and are not passed a seed here.
"""
import os
from pathlib import Path

import numpy as np
import torch

from ex.ablations.eig_boed import priors, sim
from ex.ablations.eig_boed.study import study_seed, reseed_sampler  # noqa: F401 (re-export)
from ex.ablations.eig_elbo_boed import design_select, posterior_opt
from ex.utils.step2_runner.worker import _atomic_h5_write
from ex.utils.step2_runner.load_winners import resolve_hp
from ex.utils.hpo.frozen import METHOD_ALIAS


def _resolve_hp_flex(winners, method):
    """resolve_hp tolerant to the method-name convention differing across winners
    files: the EIG gold file uses short names (e.g. MDRE) while the ELBO file uses
    canonical names (e.g. MDRE_15). try the given name, else the short alias that
    METHOD_ALIAS maps onto it."""
    try:
        return resolve_hp(winners, method, bucket_id=None)
    except KeyError:
        for short, canon in METHOD_ALIAS.items():
            if canon == method:
                return resolve_hp(winners, short, bucket_id=None)
        raise


def run_study(cell, cfg, out_dir, device, *, winners_elbo, winners_eig=None,
              config_hash, force=False):
    """execute one sequential BOED study with alpha-tempered posterior updates.

    pseudocode:
      1. load prior, draw theta_star (seeded, device-agnostic).
      2. initialize belief = prior; extract prior data for the shard.
      3. for r in range(R):
         a. snapshot mu_pre, Sigma_pre; compute ceil_r from Sigma_pre (cross-check only).
         b. xi_r, design_records, eig_star_r <- design_select.select_design
            (channel-agnostic; eig_star_r is the single source of truth for the shard).
         c. observe y_obs at xi_r (seeded via sim.derive_seed, replay-safe).
         d. alpha_hat, mu, Sigma, alpha_records, fell_back <- posterior_opt.optimize_alpha
            (1-D BO over alpha maximizing DRE-ELBO of the tempered posterior; belief update).
         e. mu1, Sigma1 <- priors.posterior_update(mu_pre, Sigma_pre, xi_r, y_obs, sigma2)
            (exact conjugate reference at alpha=1, same pre-round belief);
            post_kl_r <- posterior_opt.post_kl(mu, Sigma, mu1, Sigma1).
         f. flatten this round's design_records / alpha_records into flat per-trial arrays.
      4. assemble flat h5 payload (prior + per-round + per-trial + per-alpha-trial arrays).
      5. _atomic_h5_write the shard; return status dict.

    args:
      cell: tuple (arm, geometry, prior_idx, method, seed_rep).
      cfg: dict, see required keys below.
      out_dir: str, directory for shard (created if needed).
      device: str, torch device ("cuda", "cpu").

    kwargs:
      winners_elbo: dict[method -> dict], ELBO winners loaded once by step1
        via load_winners(); resolves the posterior-channel hyperparameters.
      winners_eig: optional dict[method -> dict], EIG winners; resolves the
        design-channel hyperparameters when design_channel == "dre_eig".
      config_hash: str, hash of cfg.
      force: bool, if True ignore existing shard with matching hash.

    returns:
      dict with keys: path, n_trials, n_rounds, complete, status.
    """
    # step 1: unpack and config validation (data_dim/nsamples, not d/n)
    arm, geometry, prior_idx, method, seed_rep = cell
    d = cfg["data_dim"]
    n_samples = cfg["nsamples"]
    num_priors = cfg["num_priors"]
    R = cfg["n_rounds"]
    T = cfg["n_trials"]
    n_startup = cfg["n_startup"]
    T_alpha = cfg["n_trials_alpha"]
    n_startup_alpha = cfg["n_startup_alpha"]
    sigma2 = cfg["sigma2"]
    config_seed = cfg["config_seed"]
    design_channel = cfg.get("design_channel", "dre_eig")

    required = {
        "data_dim", "nsamples", "num_priors", "n_rounds", "n_trials", "n_startup",
        "n_trials_alpha", "n_startup_alpha", "sigma2", "config_seed",
        "journal_scratch", "design_channel",
    }
    assert required.issubset(cfg.keys()), f"missing: {required - cfg.keys()}"

    # step 2: shard path and skip-if-done check
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    shard_path = os.path.join(
        out_dir, f"{arm}_{geometry}_{prior_idx}_{method}_{seed_rep}.h5"
    )

    if os.path.exists(shard_path) and not force:
        try:
            import h5py
            with h5py.File(shard_path, "r") as f:
                attrs = dict(f.attrs)
                ch_stored = attrs.get("config_hash")
                if isinstance(ch_stored, bytes):
                    ch_stored = ch_stored.decode()
                if (attrs.get("complete") and
                        attrs.get("n_trials") == R * T and
                        ch_stored == config_hash):
                    return {
                        "path": shard_path,
                        "n_trials": R * T,
                        "n_rounds": R,
                        "complete": True,
                        "status": "skip_existing",
                    }
        except Exception:
            pass  # truncated or unreadable; proceed to run

    # step 3: build prior, draw theta_star (seeded)
    prior = priors.build_prior(config_seed, geometry, prior_idx, cfg)
    seed_theta = sim.derive_seed(
        config_seed, geometry, prior_idx, method, seed_rep, rnd=-1, trial=-1
    )
    _ts = sim.draw_params(prior.mu, prior.Sigma, seed_theta, device="cpu")
    theta_star = _ts if torch.is_tensor(_ts) else torch.as_tensor(_ts, dtype=torch.float32)
    theta_star = theta_star.to(device)

    # step 4: initialize belief
    mu = prior.mu.clone().to(device)
    Sigma = prior.Sigma.clone().to(device)

    # step 5: prior data for the shard (y-independent)
    mu0 = prior.mu.cpu().numpy()
    Sigma0 = prior.Sigma.cpu().numpy()
    eig_star_val = float(prior.eig_star)
    xi_opt = prior.xi_opt.cpu().numpy()

    # step 6: resolve per-CHANNEL hyperparameters (winners-per-channel). the DRE-EIG
    # design channel uses EIG winners; the DRE-ELBO posterior channel uses ELBO winners.
    # analytic design uses no design hp -> hp_eig stays None.
    hp_elbo = resolve_hp(winners_elbo, method, bucket_id=None)
    hp_eig = (_resolve_hp_flex(winners_eig, method)
              if (design_channel == "dre_eig" and winners_eig is not None) else None)

    # step 7: per-trial and per-round accumulators (flat shard schema)
    round_idx_flat, trial_idx_flat = [], []
    a_flat, b_flat, xi_flat = [], [], []
    est_eig_flat, true_eig_flat = [], []
    state_flat, walltime_flat = [], []

    alpha_round_idx_flat, alpha_trial_idx_flat = [], []
    alpha_val_flat, elbo_est_flat = [], []
    alpha_state_flat, alpha_walltime_flat = [], []

    Sigma_r_list, Sigma_exact_r_list = [], []
    xi_r_list, y_obs_r_list = [], []
    alpha_r_list, fell_back_r_list = [], []
    eig_star_r_list, post_kl_r_list = [], []

    # step 8: journal directory; one root per job, sub-paths per round
    journal_root = os.path.join(
        os.path.expandvars(cfg["journal_scratch"]),
        os.environ.get("SLURM_JOB_ID", "local"),
    )
    Path(journal_root).mkdir(parents=True, exist_ok=True)

    # step 9: R-round loop
    for r in range(R):
        # 9.1: pre-update ceiling, for cross-check against select_design's own
        # eig_star_r (the single source of truth); do not store ceil_r directly
        # in the shard.
        mu_pre = mu.clone()
        Sigma_pre = Sigma.clone()
        lam_max_r = float(torch.linalg.eigvalsh(0.5 * (Sigma_pre + Sigma_pre.T)).max())
        ceil_r = float(0.5 * np.log1p(lam_max_r / sigma2))

        # 9.2: design selection
        journal_path_design = f"{journal_root}/design_r{r}.log"
        xi_r, design_records, eig_star_r = design_select.select_design(
            mu, Sigma, method, hp_eig, cfg, device,
            channel=design_channel, cell=cell, round_idx=r,
            journal_path=journal_path_design,
        )
        assert abs(ceil_r - eig_star_r) < 1e-4, (
            f"round {r}: ceil_r={ceil_r} != eig_star_r={eig_star_r} from select_design"
        )
        eig_star_r_list.append(eig_star_r)
        xi_r_tensor = torch.as_tensor(xi_r, dtype=Sigma.dtype, device=device)

        # 9.3: observe y_obs (seeded, replay-safe; Sigma is y-independent)
        eps_gen = sim.make_generator(sim.derive_seed(
            config_seed, geometry, prior_idx, method, seed_rep, rnd=r, trial=-1))
        eps = float(torch.randn(1, generator=eps_gen).item())
        y_obs = float((theta_star @ xi_r_tensor).item() + np.sqrt(sigma2) * eps)
        y_obs_r_list.append(y_obs)

        # 9.4: alpha-tempered posterior optimization
        journal_path_alpha = f"{journal_root}/alpha_r{r}.log"
        alpha_hat, mu, Sigma, alpha_records, fell_back = posterior_opt.optimize_alpha(
            mu_pre, Sigma_pre, xi_r_tensor, y_obs, method, hp_elbo, cfg, device,
            cell=cell, round_idx=r, journal_path=journal_path_alpha,
        )
        alpha_r_list.append(float(alpha_hat))
        fell_back_r_list.append(int(fell_back))

        # 9.5: exact reference posterior for the KL metric
        mu1, Sigma1 = priors.posterior_update(mu_pre, Sigma_pre, xi_r_tensor, y_obs, sigma2)
        Sigma_exact_r_list.append(Sigma1.cpu().numpy())
        post_kl_r = posterior_opt.post_kl(mu, Sigma, mu1, Sigma1)
        post_kl_r_list.append(post_kl_r)

        # 9.6: record round data (Sigma_r = alpha-tempered post-update)
        Sigma_r_list.append(Sigma.cpu().numpy())
        xi_r_list.append(xi_r)

        # 9.7: flatten design records (per-trial, append-only; [] for analytic)
        for rec in design_records:
            round_idx_flat.append(r)
            trial_idx_flat.append(rec["trial_idx"])
            a_flat.append(rec["a"])
            b_flat.append(rec["b"])
            xi_flat.append(rec["xi"])
            est_eig_flat.append(rec["est_eig"])
            true_eig_flat.append(rec["true_eig"])
            state_flat.append(rec["state"])
            walltime_flat.append(rec["walltime_s"])

        # 9.8: flatten alpha records (per-trial, append-only)
        for rec in alpha_records:
            alpha_round_idx_flat.append(r)
            alpha_trial_idx_flat.append(rec["trial_idx"])
            alpha_val_flat.append(rec["alpha_val"])
            elbo_est_flat.append(rec["elbo_est"])
            alpha_state_flat.append(rec["state"])
            alpha_walltime_flat.append(rec["walltime_s"])

    # step 10: assemble shard payload (flat, no groups/compound dtypes)
    payload = {
        # prior, before round 0
        "mu0": mu0,                                             # (d,)
        "Sigma0": Sigma0,                                       # (d, d)
        "eig_star": np.array(eig_star_val),                     # scalar
        "xi_opt": xi_opt,                                       # (d,)

        # per-round (length R)
        "Sigma_r": np.stack(Sigma_r_list, axis=0),               # (R, d, d)
        "Sigma_exact_r": np.stack(Sigma_exact_r_list, axis=0),   # (R, d, d)
        "xi_r": np.array(xi_r_list),                             # (R, d)
        "y_obs_r": np.array(y_obs_r_list),                       # (R,)
        "eig_star_r": np.array(eig_star_r_list),                 # (R,)
        "alpha_r": np.array(alpha_r_list, dtype=float),          # (R,)
        "fell_back_r": np.array(fell_back_r_list, dtype=np.int8),  # (R,)
        "post_kl_r": np.array(post_kl_r_list, dtype=float),      # (R,)

        # per-design-trial flat (length R*T for dre_eig; 0 for analytic)
        "round_idx": np.array(round_idx_flat, dtype=int),
        "trial_idx": np.array(trial_idx_flat, dtype=int),
        "a": np.array(a_flat, dtype=float),
        "b": np.array(b_flat, dtype=float),
        "xi": np.array(xi_flat),                                 # (n_trials, d)
        "est_eig": np.array(est_eig_flat, dtype=float),
        "true_eig": np.array(true_eig_flat, dtype=float),
        "state": np.array(state_flat, dtype='S24'),
        "walltime_s": np.array(walltime_flat, dtype=float),

        # per-alpha-trial flat (length R*T_alpha)
        "alpha_round_idx": np.array(alpha_round_idx_flat, dtype=int),
        "alpha_trial_idx": np.array(alpha_trial_idx_flat, dtype=int),
        "alpha_val": np.array(alpha_val_flat, dtype=float),
        "elbo_est": np.array(elbo_est_flat, dtype=float),
        "alpha_state": np.array(alpha_state_flat, dtype='S24'),
        "alpha_walltime_s": np.array(alpha_walltime_flat, dtype=float),
    }

    # step 11: attributes
    attrs = {
        "complete": True,
        "n_trials": R * T,
        "n_rounds": R,
        "n_trials_alpha": R * T_alpha,
        "config_hash": config_hash,
        "arm": arm,
        "geometry": geometry,
        "prior_idx": prior_idx,
        "method": method,
        "seed_rep": seed_rep,
        "design_channel": design_channel,
    }

    # step 12: atomic write
    _atomic_h5_write(Path(shard_path), payload, attrs)

    # step 13: return
    return {
        "path": shard_path,
        "n_trials": R * T,
        "n_rounds": R,
        "complete": True,
        "status": "completed",
    }
