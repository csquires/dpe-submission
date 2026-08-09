"""eig_boed sequential BOED study orchestrator.

orchestrates R rounds x T trials of GP-guided design optimization with conjugate
prior updates. writes results atomically via _atomic_h5_write with flat h5 schema.
"""
import os
import time
from hashlib import blake2b
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import GPSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

import torch

from ex.ablations.eig_boed import priors, design, sim
from ex.utils.eig_ldr import joint_and_shuffled
from ex.utils.hpo.frozen import fit_eldr
from ex.utils.step2_runner.worker import _atomic_h5_write
from ex.utils.step2_runner.load_winners import resolve_hp


def study_seed(cell, r, config_seed):
    """deterministic seed from (arm, geometry, prior_idx, method, seed_rep, r).

    args:
      cell: tuple (arm, geometry, prior_idx, method, seed_rep)
      r: int, round index (0 <= r < n_rounds)
      config_seed: int, from cfg["config_seed"]

    returns:
      int seed in [0, 2^63), consistent with sim.derive_seed entropy composition.

    method_hash = blake2b(method, digest_size=8) & 0xFFFFFFFFFFFFFFFF.
    SeedSequence([config_seed, geom_id, prior_idx, method_hash, seed_rep, r])
    then int(seq.generate_state(1)[0]), NOT seq.entropy[0].
    """
    arm, geometry, prior_idx, method, seed_rep = cell
    geom_id = 0 if geometry == "diag" else 1
    method_hash = int.from_bytes(
        blake2b(method.encode(), digest_size=8).digest(), byteorder="big"
    ) & 0xFFFFFFFFFFFFFFFF
    seq = np.random.SeedSequence(
        [int(config_seed), geom_id, prior_idx, method_hash, seed_rep, r]
    )
    return int(seq.generate_state(1)[0])


def startup_designs(study_seed_r, n_startup):
    """sobol startup designs for a round.

    wrapper over design.sobol_startup; used in gather to validate shard
    startup points match recomputed sobol.

    args:
      study_seed_r: int seed for this round (study_seed(cell, r, config_seed))
      n_startup: int, number of startup trials

    returns:
      np.ndarray (n_startup, 2) float, scrambled sobol in [-1, 1]^2
    """
    return design.sobol_startup(n_startup, seed=study_seed_r)


def reseed_sampler(sampler, trial_seed):
    """reseed optuna GPSampler for trial-level replay.

    THREE operations, all required for resume invariance:
    1. reseed main RNG stream
    2. reseed independent sampler's stream
    3. clear GP model cache (warm-start drift breaks resume if not cleared)

    args:
      sampler: optuna.samplers.GPSampler instance
      trial_seed: int seed for this trial

    returns:
      None (modifies sampler in place)
    """
    sampler._rng.rng.seed(trial_seed)
    sampler._independent_sampler._rng.rng.seed(trial_seed)
    sampler._gprs_cache_list = None


def run_study(cell, cfg, out_dir, device, *, winners, config_hash, force=False):
    """execute one sequential BOED study.

    orchestrate R rounds x T trials of GP-guided design optimization with
    conjugate prior updates. write atomic shard via _atomic_h5_write.

    args:
      cell: tuple (arm, geometry, prior_idx, method, seed_rep)
      cfg: dict with keys: data_dim, nsamples, num_priors, n_rounds, n_trials,
        n_startup, sigma2, config_seed, journal_scratch, [+ other keys]
      out_dir: str, directory for shard (created if needed)
      device: str, torch device (cuda, cpu)

    kwargs (required):
      winners: dict[method -> dict], loaded once by step1 via load_winners()
      config_hash: str, sha256 hash of cfg
      force: bool, if True ignore existing shard with matching hash

    returns:
      dict with keys: path, n_trials, n_rounds, complete, status
    """
    # unpack
    arm, geometry, prior_idx, method, seed_rep = cell
    d = cfg["data_dim"]         # not cfg["d"]
    n_samples = cfg["nsamples"] # not cfg["n"]
    num_priors = cfg["num_priors"]
    R = cfg["n_rounds"]
    T = cfg["n_trials"]
    n_startup = cfg["n_startup"]
    sigma2 = cfg["sigma2"]
    config_seed = cfg["config_seed"]

    # config validation
    required = {
        "data_dim", "nsamples", "num_priors", "n_rounds", "n_trials", "n_startup",
        "sigma2", "config_seed", "journal_scratch"
    }
    assert required.issubset(cfg.keys()), f"missing: {required - cfg.keys()}"

    # shard path and skip-if-done check
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
                        "status": "skip_existing"
                    }
        except Exception:
            pass  # truncated or unreadable; proceed to run

    # build prior once (CPU-seeded, on device)
    prior = priors.build_prior(config_seed, geometry, prior_idx, cfg)

    # draw theta_star once via sim.draw_params (NOT inline cholesky)
    seed_theta = sim.derive_seed(
        config_seed, geometry, prior_idx, method, seed_rep, rnd=-1, trial=-1
    )
    _ts = sim.draw_params(prior.mu, prior.Sigma, seed_theta, device="cpu")
    theta_star = _ts if torch.is_tensor(_ts) else torch.as_tensor(_ts, dtype=torch.float32)
    theta_star = theta_star.to(device)

    # initialize belief
    mu = prior.mu.clone().to(device)
    Sigma = prior.Sigma.clone().to(device)

    # prior data for shard (before any rounds)
    mu0 = prior.mu.cpu().numpy()
    Sigma0 = prior.Sigma.cpu().numpy()
    eig_star_val = float(prior.eig_star)
    xi_opt = prior.xi_opt.cpu().numpy()

    # oracle-family tilt (arm == "oracle"): fixed per-prior unit vector in design
    # space (R^d). corrupts the exact EIG by a known, design-dependent amount to
    # form a positive control. seeded per prior only (geometry/seed-independent),
    # so 12 priors give 12 independent tilt-vs-optimum alignments.
    tilt_frac = cfg.get("oracle_methods", {}).get(method, 0.0) if arm == "oracle" else 0.0
    if arm == "oracle":
        _wg = sim.make_generator(
            sim.derive_seed(config_seed, "tilt", prior_idx, "tilt", 0, rnd=-2, trial=-2))
        _w = torch.randn(d, generator=_wg)
        tilt_w = (_w / torch.linalg.norm(_w)).to(device)
    else:
        tilt_w = None

    # resolve hyperparameters once; oracle arms carry no hp (no neural fit)
    hp = None if arm == "oracle" else resolve_hp(winners, method, bucket_id=None)

    # accumulators for flat per-trial and per-round arrays (flat h5 schema)
    round_idx_flat = []
    trial_idx_flat = []
    a_flat = []
    b_flat = []
    xi_flat = []
    est_eig_flat = []
    true_eig_flat = []
    state_flat = []
    walltime_flat = []

    Sigma_r_list = []
    xi_r_list = []
    y_obs_r_list = []
    eig_star_r_list = []

    # journal directory. isolate per job so a study never loads a STALE journal
    # from a prior (cancelled/preempted) run on a reused node; a stale journal
    # makes optimize(n_trials=T) over-run to T + n_stale. resume is therefore
    # per-cell (shard skip-if-done), not per-round-trial.
    journal_root = os.path.join(
        os.path.expandvars(cfg["journal_scratch"]),
        os.environ.get("SLURM_JOB_ID", "local"),
    )
    Path(journal_root).mkdir(parents=True, exist_ok=True)

    # R-round loop
    for r in range(R):
        # per-round study with fresh sampler
        study_seed_r = study_seed(cell, r, config_seed)
        sampler = GPSampler(seed=study_seed_r, n_startup_trials=n_startup)

        # round ceiling on the PRE-update belief: max achievable true EIG this
        # round = 0.5*log1p(lam_max(Sigma)/sigma2). used for the anytime regret
        # (round end) AND to scale the oracle tilt proportional to the signal.
        lam_max_r = float(torch.linalg.eigvalsh(0.5 * (Sigma + Sigma.T)).max())
        ceil_r = float(0.5 * np.log1p(lam_max_r / sigma2))

        journal_path = os.path.join(
            journal_root,
            f"{arm}_{geometry}_{prior_idx}_{method}_{seed_rep}_r{r}.log"
        )
        storage = JournalStorage(JournalFileBackend(journal_path))
        study = optuna.create_study(
            study_name=f"r{r}",
            storage=storage,
            direction="maximize",
            sampler=sampler,
            load_if_exists=True,
        )

        # enqueue sobol startup if absent (include WAITING in existing check)
        sobol_pts = startup_designs(study_seed_r, n_startup)
        existing = {t.number for t in study.get_trials(deepcopy=False, states=None)}
        for i in range(n_startup):
            if i not in existing:
                study.enqueue_trial({
                    "a": float(sobol_pts[i, 0]),
                    "b": float(sobol_pts[i, 1])
                })

        # per-round records dict (trial.number -> record)
        records = {}

        # per-trial objective (sampler and hp bind as closure vars)
        def objective(trial):
            # reseed before suggest_float
            trial_seed = sim.derive_seed(
                config_seed, geometry, prior_idx, method, seed_rep, rnd=r, trial=trial.number
            )
            reseed_sampler(sampler, trial_seed)

            # design proposal via design.suggest (which calls suggest_float internally)
            xi_np = design.suggest(trial)  # (3,) ndarray

            # read a, b from trial.params (design.suggest already set them)
            a = trial.params["a"]
            b = trial.params["b"]

            # convert xi ONCE to torch, use same for all downstream
            xi = torch.as_tensor(xi_np, dtype=Sigma.dtype, device=device)

            # oracle-family arm: synthetic reference, NO neural fit. est = exact
            # EIG + known design-dependent tilt (positive control). never fails,
            # so honest-protocol selection and failure_rate stay well-defined.
            if arm == "oracle":
                true_eig_val = float(priors.eig_true(Sigma, xi, sigma2))
                est_eig_val = true_eig_val + tilt_frac * ceil_r * float((tilt_w @ xi).item())
                records[trial.number] = {
                    "a": float(a), "b": float(b), "xi": xi_np.copy(),
                    "est": est_eig_val, "true": true_eig_val, "walltime": 0.0,
                }
                return est_eig_val

            est_eig_val = np.nan
            elapsed = 0.0

            try:
                # simulate under current belief
                seed_sim = sim.derive_seed(
                    config_seed, geometry, prior_idx, method, seed_rep, rnd=r, trial=trial.number
                )
                theta, y = sim.simulate(mu, Sigma, xi, sigma2, n_samples, seed_sim, device)

                # draw random generator for joint/shuffled
                gen = sim.make_generator(seed_sim)
                joint, shuffled = joint_and_shuffled(theta, y, generator=gen)

                # fit estimator with wall time
                t0 = time.perf_counter()
                res = fit_eldr(method, hp, joint, shuffled, input_dim=d+1, device=device)
                elapsed = time.perf_counter() - t0

                # check fit success
                if not res.ok:
                    raise RuntimeError(res.error or "fit returned False")

                est_eig_val = float(res.value)

            except Exception as e:
                # catch all, check for OOM
                if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()

                # true EIG even on failure
                true_eig_val = float(priors.eig_true(Sigma, xi, sigma2))

                # store record with est=nan before raising
                records[trial.number] = {
                    "a": float(a),
                    "b": float(b),
                    "xi": xi_np.copy(),
                    "est": np.nan,
                    "true": true_eig_val,
                    "walltime": elapsed
                }

                raise

            # true EIG under current belief
            true_eig_val = float(priors.eig_true(Sigma, xi, sigma2))

            # store record with est=float(res.value)
            records[trial.number] = {
                "a": float(a),
                "b": float(b),
                "xi": xi_np.copy(),
                "est": est_eig_val,
                "true": true_eig_val,
                "walltime": elapsed
            }

            return est_eig_val

        # optimize the round. subtract trials already EXECUTED (any state
        # except WAITING) so a resumed study reaches T TOTAL, not T + existing
        # (optuna's n_trials is "run this many MORE"). WAITING trials are the
        # startup designs we just enqueued; they have NOT run yet, and optimize
        # consumes the queue first, so counting them would under-run a fresh
        # study by n_startup. redundant with the per-job journal.
        n_have = len([t for t in study.get_trials(deepcopy=False, states=None)
                      if t.state != TrialState.WAITING])
        study.optimize(objective, n_trials=max(0, T - n_have), gc_after_trial=True,
                      catch=(RuntimeError, ValueError))

        # build flat arrays from study.trials (FrozenTrials with .state) and records
        for t in study.trials:
            tnum = t.number
            rec = records.get(tnum)

            if rec is None:
                # trial failed before recording
                rec = {
                    "a": np.nan,
                    "b": np.nan,
                    "xi": np.zeros(d),
                    "est": np.nan,
                    "true": np.nan,
                    "walltime": 0.0
                }

            round_idx_flat.append(r)
            trial_idx_flat.append(tnum)
            a_flat.append(rec["a"])
            b_flat.append(rec["b"])
            xi_flat.append(rec["xi"])
            est_eig_flat.append(rec["est"])
            true_eig_flat.append(rec["true"])
            walltime_flat.append(rec["walltime"])
            state_flat.append(str(t.state))

        # round-end: select xi_r via honest protocol
        trials_complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not trials_complete:
            raise RuntimeError(f"round {r} has no completed trials")

        # argmax est_eig on finite values only (honest protocol)
        est_eig_by_tnum = {}
        for j, tnum in enumerate(trial_idx_flat):
            if round_idx_flat[j] == r:
                est_eig_by_tnum[tnum] = est_eig_flat[j]

        est_eig_vals = np.array([
            (est_eig_by_tnum[t.number] if np.isfinite(est_eig_by_tnum.get(t.number, np.nan))
             else -np.inf)
            for t in study.trials
        ])
        trial_max_idx = int(np.argmax(est_eig_vals))

        # gather xi_r and related data from selected trial
        xi_r = None
        for j, tnum in enumerate(trial_idx_flat):
            if round_idx_flat[j] == r and tnum == trial_max_idx:
                xi_r = xi_flat[j]
                break

        if xi_r is None:
            raise RuntimeError(f"round {r}: could not find selected trial {trial_max_idx}")

        # round ceiling already computed at round top (ceil_r) on this same
        # pre-update Sigma; reuse it (pre-update required, else regret < 0).
        eig_star_r = ceil_r

        # observe: seed eps for replay (only moves mu; Sigma is y-indep)
        xi_r_tensor = torch.as_tensor(xi_r, dtype=Sigma.dtype, device=Sigma.device)
        eps_gen = sim.make_generator(sim.derive_seed(
            config_seed, geometry, prior_idx, method, seed_rep, rnd=r, trial=-1))
        eps = float(torch.randn(1, generator=eps_gen).item())
        y_obs = float((theta_star @ xi_r_tensor).item() + np.sqrt(sigma2) * eps)

        # conjugate posterior update (independent of y_obs for Sigma)
        mu, Sigma = priors.posterior_update(mu, Sigma, xi_r_tensor, y_obs, sigma2)

        # accumulate round data (Sigma_r = POST-update)
        Sigma_r_list.append(Sigma.cpu().numpy())
        xi_r_list.append(xi_r)
        y_obs_r_list.append(y_obs)
        eig_star_r_list.append(eig_star_r)

    # assemble shard payload (FLAT, no groups, no compound dtypes)
    payload = {
        # prior (before round 0)
        "mu0": mu0,
        "Sigma0": Sigma0,
        "eig_star": np.array(eig_star_val),
        "xi_opt": xi_opt,

        # per-round (length R)
        "Sigma_r": np.stack(Sigma_r_list, axis=0),  # (R, d, d)
        "xi_r": np.array(xi_r_list),                # (R, d)
        "y_obs_r": np.array(y_obs_r_list),          # (R,)
        "eig_star_r": np.array(eig_star_r_list),    # (R,)

        # per-trial flat (length R*T)
        "round_idx": np.array(round_idx_flat, dtype=int),
        "trial_idx": np.array(trial_idx_flat, dtype=int),
        "a": np.array(a_flat, dtype=float),
        "b": np.array(b_flat, dtype=float),
        "xi": np.array(xi_flat),
        "est_eig": np.array(est_eig_flat, dtype=float),
        "true_eig": np.array(true_eig_flat, dtype=float),
        "state": np.array(state_flat, dtype='S24'),
        "walltime_s": np.array(walltime_flat, dtype=float),
    }

    attrs = {
        "complete": True,
        "n_trials": R * T,
        "n_rounds": R,
        "config_hash": config_hash,
        "arm": arm,
        "geometry": geometry,
        "prior_idx": prior_idx,
        "method": method,
        "seed_rep": seed_rep,
    }

    # atomic write
    _atomic_h5_write(Path(shard_path), payload, attrs)

    return {
        "path": shard_path,
        "n_trials": R * T,
        "n_rounds": R,
        "complete": True,
        "status": "completed",
    }
