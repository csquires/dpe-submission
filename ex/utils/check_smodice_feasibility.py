"""structural lower bound on K1(alpha=1) for the smodice gridworld family.

avoids the full G_alpha x G_beta KL grid build. uses
  K1 >= E_{s ~ d_E^S} [ KL(pi_E(.|s) || pi_O,1(.|s)) ]
which needs only one Bellman occupancy solve (for d_E) plus per-state
softmax KL.

usage:
  python -m ex.utils.check_smodice_feasibility \
      [--config ex/smodice_eldr_estimation/config.yaml] \
      [--tau 0.05 0.1] [--p-slip 0.05 0.1 0.2]

prints a table of structural lower bounds across the cartesian product of
the (tau, p_slip) sweep, holding all other gridworld params fixed at
config values.
"""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import yaml
from scipy.special import rel_entr

from src.utils.gridworld import (
    build_gridworld,
    reward_to_goal,
    softmax_policy,
    value_iteration,
)
from src.utils.occupancy import bellman_occupancy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="ex/smodice_eldr_estimation/config.yaml")
    p.add_argument("--tau", type=float, nargs="+", default=None,
                   help="override tau sweep (default: just config tau)")
    p.add_argument("--p-slip", type=float, nargs="+", default=None,
                   help="override p_slip sweep (default: just config p_slip)")
    return p.parse_args()


def lower_bound_k1(L, p_slip, gamma, terminals, expert_goal, anti_goal, tau,
                   mu0_kind, mu0_centers):
    """compute structural lower bound on K1(alpha=1).

    procedure:
      1. build mdp + reward arrays.
      2. value iteration -> Q_E, Q_{O,1}.
      3. softmax -> pi_E, pi_O.
      4. bellman occupancy -> d_E [|S|, |A|]; marginalize -> d_E^S.
      5. per-state action KL via scipy.special.rel_entr.
      6. inner product with d_E^S.

    returns:
        dict with keys:
          lower_bound: float, in nats.
          d_E_state: [|S|], expert state marginal.
          per_state_kl: [|S|], action KL at each state.
    """
    mdp = build_gridworld(
        L=L, p_slip=p_slip, terminals=terminals, gamma=gamma,
        mu0_kind=mu0_kind, mu0_centers=mu0_centers,
    )
    r_E = reward_to_goal(L, expert_goal, terminals)
    r_anti = reward_to_goal(L, anti_goal, terminals)

    Q_E = value_iteration(mdp.P, r_E, gamma)
    Q_O = value_iteration(mdp.P, r_anti, gamma)

    pi_E = softmax_policy(Q_E, tau)
    pi_O = softmax_policy(Q_O, tau)

    d_E = bellman_occupancy(mdp.P, mdp.mu0, pi_E, gamma)  # [|S|, |A|]
    d_O = bellman_occupancy(mdp.P, mdp.mu0, pi_O, gamma)  # [|S|, |A|]
    d_E_state = d_E.sum(axis=1)  # [|S|]

    # per-state action KL: rel_entr handles log/0 carefully.
    # rel_entr(p, q) = p * log(p/q), elementwise, with rel_entr(0, 0) = 0.
    per_state_kl = rel_entr(pi_E, pi_O).sum(axis=1)  # [|S|]
    structural_lower = float((d_E_state * per_state_kl).sum())

    # also compute exact K1(alpha=1) = KL(d_E || d_O) joint
    eps = 1e-12
    k1_exact = float(np.sum(rel_entr(d_E, d_O + eps)))

    return {
        "lower_bound": structural_lower,
        "k1_exact": k1_exact,
        "d_E_state_max": float(d_E_state.max()),
        "per_state_kl_max": float(per_state_kl.max()),
    }


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    gw = config["gridworld"]

    taus = args.tau or [gw["tau"]]
    p_slips = args.p_slip or [gw["p_slip"]]

    L = gw["L"]
    gamma = gw["gamma"]
    terminals = [tuple(t) for t in gw["terminals"]]
    expert_goal = tuple(gw["expert_goal"])
    anti_goal = tuple(gw["anti_goal"])
    mu0_kind = gw.get("mu0_kind", "uniform")
    mu0_centers = [tuple(c) for c in gw.get("mu0_centers", [])]

    print(f"L={L}, gamma={gamma}, terminals={terminals}, "
          f"expert={expert_goal}, anti={anti_goal}, "
          f"mu0={mu0_kind}({mu0_centers})")
    print(f"\n{'tau':>6} {'p_slip':>8} {'K1_lower':>10} {'K1_exact':>10} {'max_d_E^S':>12} {'max_per_state_KL':>18}")
    print("-" * 80)

    for tau, p_slip in product(taus, p_slips):
        out = lower_bound_k1(L, p_slip, gamma, terminals, expert_goal, anti_goal,
                             tau, mu0_kind, mu0_centers)
        print(f"{tau:>6.3g} {p_slip:>8.3g} {out['lower_bound']:>10.3f} "
              f"{out['k1_exact']:>10.3f} "
              f"{out['d_E_state_max']:>12.4g} {out['per_state_kl_max']:>18.3f}")


if __name__ == "__main__":
    main()
