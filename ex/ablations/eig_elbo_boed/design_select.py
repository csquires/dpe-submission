"""select the next BOED design under a channel-specific strategy.

sequential design selection for eig_elbo_boed's per-round loop. two channels:
"dre_eig" (GP-guided optuna search maximizing a DRE-estimated EIG, mirroring
eig_boed/study.py's per-round loop) and "analytic"
(closed-form top eigenvector of the belief covariance, for elbo_boed parity).

seeds are derived internally from (cell, round_idx, cfg["config_seed"]) via
eig_boed.study's study_seed/reseed_sampler and eig_boed.sim's derive_seed, so
the caller need not thread per-trial seeds through this module.
"""
import os
import time

import numpy as np
import optuna
from optuna.samplers import GPSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
import torch

from ex.ablations.eig_boed import design, sim, priors
from ex.ablations.eig_boed.study import study_seed, reseed_sampler
from ex.utils.eig_ldr import joint_and_shuffled
from ex.utils.hpo.frozen import fit_eldr


def _round_ceiling(belief_Sigma: torch.Tensor, sigma2: float) -> float:
    """pre-update belief ceiling: 0.5*log1p(lam_max(Sigma)/sigma2).

    args:
        belief_Sigma (d,d): current belief covariance.
        sigma2 (float): likelihood variance.

    returns:
        float, round ceiling eig_star_r (used for regret and oracle-tilt scaling).
    """
    Sigma_sym = 0.5 * (belief_Sigma + belief_Sigma.T)
    lam_max = float(torch.linalg.eigvalsh(Sigma_sym).max())
    return float(0.5 * np.log1p(lam_max / sigma2))


def _analytic_select(belief_Sigma: torch.Tensor) -> np.ndarray:
    """closed-form design: top eigenvector of the symmetrized belief covariance.

    args:
        belief_Sigma (d,d): current belief covariance.

    returns:
        xi_r (d,) np.ndarray float64.
    """
    Sigma_sym = 0.5 * (belief_Sigma + belief_Sigma.T)
    _, evecs = torch.linalg.eigh(Sigma_sym)
    xi_r_torch = evecs[:, -1]  # top eigenvector, (d,)
    return xi_r_torch.detach().cpu().numpy().astype(np.float64)


def _dre_eig_select(
    belief_mu: torch.Tensor,
    belief_Sigma: torch.Tensor,
    method: str,
    hp: dict,
    cfg: dict,
    device: str,
    cell: tuple,
    round_idx: int,
    journal_path: str | None,
) -> tuple[np.ndarray, list[dict]]:
    """GP-guided per-round design search maximizing a DRE-estimated EIG.

    input --> optuna GPSampler over (a,b) in [-1,1]^2, chart-mapped to a unit
    design xi via design.suggest --> simulate (theta,y) under the belief -->
    fit_eldr for an estimated EIG --> honest-protocol argmax over finite
    per-trial estimates selects xi_r. self-contained optuna study for this
    round, journaled at journal_path; mirrors eig_boed/study.py's per-round
    objective and round-end selection.

    args:
        belief_mu (d,): current belief mean.
        belief_Sigma (d,d): current belief covariance.
        method (str): DRE method name for fit_eldr.
        hp (dict): hyperparameters for fit_eldr.
        cfg (dict): data_dim, nsamples, n_trials, n_startup, sigma2, config_seed.
        device (str): torch device.
        cell (tuple): (arm, geometry, prior_idx, method, seed_rep).
        round_idx (int): round index r.
        journal_path (str | None): optuna journal file path; journal_root is
            created by the caller (study.py); this function does not mkdir.

    returns:
        (xi_r, design_records):
            xi_r (d,) np.ndarray float64: the selected design.
            design_records: list[dict] with keys trial_idx, a, b, xi, est_eig,
                true_eig, state, walltime_s.
    """
    if journal_path is None:
        raise ValueError("journal_path is required for channel='dre_eig'")

    arm, geometry, prior_idx, _, seed_rep = cell
    cs = cfg["config_seed"]
    d = cfg["data_dim"]
    sigma2 = cfg["sigma2"]

    study_seed_r = study_seed(cell, round_idx, cs)
    sampler = GPSampler(seed=study_seed_r, n_startup_trials=cfg["n_startup"])
    # resume is per-CELL (shard skip-if-done), so start each round's study fresh:
    # a requeued same-job attempt must not inherit stale/failed trials via
    # load_if_exists (that reads n_have>=T -> runs 0 new trials).
    if os.path.exists(journal_path):
        os.remove(journal_path)
    storage = JournalStorage(JournalFileBackend(journal_path))
    study = optuna.create_study(
        study_name="design_search",
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    # enqueue sobol startup designs if absent (include WAITING in existing check)
    sobol_pts = design.sobol_startup(cfg["n_startup"], seed=study_seed_r)
    existing = {t.number for t in study.get_trials(deepcopy=False, states=None)}
    for i in range(cfg["n_startup"]):
        if i not in existing:
            study.enqueue_trial({"a": float(sobol_pts[i, 0]), "b": float(sobol_pts[i, 1])})

    records = {}  # trial.number -> {a, b, xi, est, true, walltime}

    def objective(trial):
        # mandatory first action: reseed before suggest (resume invariance)
        trial_seed = sim.derive_seed(
            cs, geometry, prior_idx, method, seed_rep, rnd=round_idx, trial=trial.number
        )
        reseed_sampler(sampler, trial_seed)

        xi_np = design.suggest(trial)  # (d,) ndarray
        a = trial.params["a"]
        b = trial.params["b"]
        xi = torch.as_tensor(xi_np, dtype=belief_Sigma.dtype, device=device)

        est_eig_val = np.nan
        elapsed = 0.0
        try:
            theta, y = sim.simulate(
                belief_mu, belief_Sigma, xi, sigma2, cfg["nsamples"], trial_seed, device
            )
            gen = sim.make_generator(trial_seed)
            joint, shuffled = joint_and_shuffled(theta, y, generator=gen)

            t0 = time.perf_counter()
            res = fit_eldr(method, hp, joint, shuffled, input_dim=d + 1, device=device)
            elapsed = time.perf_counter() - t0

            if not res.ok:
                raise RuntimeError(res.error or "fit returned False")
            est_eig_val = float(res.value)

        except Exception as e:
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            true_eig_val = float(priors.eig_true(belief_Sigma, xi, sigma2))
            records[trial.number] = {
                "a": float(a), "b": float(b), "xi": xi_np.copy(),
                "est": np.nan, "true": true_eig_val, "walltime": elapsed,
            }
            raise

        true_eig_val = float(priors.eig_true(belief_Sigma, xi, sigma2))
        records[trial.number] = {
            "a": float(a), "b": float(b), "xi": xi_np.copy(),
            "est": est_eig_val, "true": true_eig_val, "walltime": elapsed,
        }
        return est_eig_val

    n_have = len([t for t in study.get_trials(deepcopy=False, states=None)
                  if t.state != TrialState.WAITING])
    study.optimize(objective, n_trials=max(0, cfg["n_trials"] - n_have),
                    gc_after_trial=True, catch=(RuntimeError, ValueError))

    # honest-protocol argmax over finite estimates only
    est_eig_vals = np.array([
        (records[t.number]["est"]
         if t.number in records and np.isfinite(records[t.number]["est"])
         else -np.inf)
        for t in study.trials
    ])
    if len(est_eig_vals) == 0 or np.all(np.isinf(est_eig_vals)):
        raise RuntimeError("no finite EIG estimates in round")
    trial_max_idx = int(study.trials[int(np.argmax(est_eig_vals))].number)
    xi_r = np.asarray(records[trial_max_idx]["xi"], dtype=np.float64)

    design_records = [
        {
            "trial_idx": t.number,
            "a": records[t.number]["a"],
            "b": records[t.number]["b"],
            "xi": records[t.number]["xi"],
            "est_eig": records[t.number]["est"],
            "true_eig": records[t.number]["true"],
            "state": str(t.state),
            "walltime_s": records[t.number]["walltime"],
        }
        for t in study.trials if t.number in records
    ]
    return xi_r, design_records


def select_design(
    belief_mu: torch.Tensor,
    belief_Sigma: torch.Tensor,
    method: str,
    hp: dict,
    cfg: dict,
    device: str,
    *,
    channel: str,
    cell: tuple,
    round_idx: int,
    journal_path: str | None = None,
) -> tuple[np.ndarray, list[dict], float]:
    """select the next design under a channel-specific strategy.

    input (belief_mu, belief_Sigma) --> route by channel:
      "dre_eig"  --> _dre_eig_select: GP-guided optuna search over per-trial
        DRE-estimated EIG; honest-protocol argmax picks xi_r.
      "analytic" --> _analytic_select: closed-form top eigenvector of Sigma.
    --> (xi_r, design_records, eig_star_r).

    NO seed argument: the GPSampler seed is study_seed(cell, round_idx,
    cfg["config_seed"]); per-trial reseed and simulation seeds are
    sim.derive_seed(cs, geometry, prior_idx, method, seed_rep, rnd=round_idx,
    trial=trial.number), where cell unpacks as (arm, geometry, prior_idx,
    method, seed_rep) and cs = cfg["config_seed"].

    args:
        belief_mu (d,) float32: current belief mean.
        belief_Sigma (d,d) float32: current belief covariance (symmetric).
        method (str): DRE method name (e.g. 'DensityRatioEstimator').
        hp (dict): hyperparameters for fit_eldr; ignored for channel='analytic'.
        cfg (dict): data_dim, nsamples, n_trials, n_startup, sigma2, config_seed.
        device (str): 'cuda' or 'cpu'.
        channel (str): 'dre_eig' or 'analytic'.
        cell (tuple): (arm, geometry, prior_idx, method, seed_rep).
        round_idx (int): round index r.
        journal_path (str | None): optuna journal path; required for 'dre_eig',
            ignored for 'analytic'.

    returns:
        (xi_r, design_records, eig_star_r):
            xi_r (d,) np.ndarray float64: selected design.
            design_records: list[dict] (dre_eig) or [] (analytic).
            eig_star_r (float): round ceiling, 0.5*log1p(lam_max/sigma2).
    """
    eig_star_r = _round_ceiling(belief_Sigma, cfg["sigma2"])

    if channel == "dre_eig":
        xi_r, design_records = _dre_eig_select(
            belief_mu, belief_Sigma, method, hp, cfg, device, cell, round_idx, journal_path
        )
        return xi_r, design_records, eig_star_r
    elif channel == "analytic":
        xi_r = _analytic_select(belief_Sigma)
        return xi_r, [], eig_star_r
    else:
        raise ValueError(f"channel must be 'dre_eig' or 'analytic'; got {channel!r}")
