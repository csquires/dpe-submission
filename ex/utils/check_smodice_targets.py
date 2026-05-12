"""check feasibility of (K1, K2) target list for smodice gridworld.

builds a small alpha-beta KL grid (default G=15) and runs prescribe() on
each target. reports per-target feasibility, (alpha*, beta*), and realized
(K1, K2). useful as a fast sanity check before committing to a full
G=50 grid build.

usage:
  python -m ex.utils.check_smodice_targets \
      [--config ex/smodice_eldr_estimation/config.yaml] \
      [--tau 0.005] [--p-slip 0.05] [--G 15] \
      [--k1 0.1 0.3 0.9 2.7 8.1] [--k2 0.5 1.0 2.0]

with no overrides, uses config values.
"""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import yaml

from src.utils.gridworld import build_gridworld, shaped_reward_to_goal
from ex.utils.prescribed_kls import build_kl_grid, prescribe


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default="ex/smodice_eldr_estimation/config.yaml")
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--p-slip", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--G", type=int, default=15,
                   help="alpha/beta grid resolution (default 15)")
    p.add_argument("--k1", type=float, nargs="+", default=None)
    p.add_argument("--k2", type=float, nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    gw = config["gridworld"]

    L = gw["L"]
    p_slip = args.p_slip if args.p_slip is not None else gw["p_slip"]
    gamma = args.gamma if args.gamma is not None else gw["gamma"]
    tau = args.tau if args.tau is not None else gw["tau"]
    terminals = [tuple(t) for t in gw["terminals"]]
    expert_goal = tuple(gw["expert_goal"])
    anti_goal = tuple(gw["anti_goal"])
    mu0_kind = gw.get("mu0_kind", "uniform")
    mu0_centers = [tuple(c) for c in gw.get("mu0_centers", [])]
    reward_kind = gw.get("reward_kind", "sparse")
    reward_sigma = float(gw.get("reward_sigma", 1.0))

    k1_targets = args.k1 if args.k1 is not None else config["kl_targets"]["k1_values"]
    k2_targets = args.k2 if args.k2 is not None else config["kl_targets"]["k2_values"]

    print(f"L={L}, gamma={gamma}, tau={tau}, p_slip={p_slip}, "
          f"mu0={mu0_kind}({mu0_centers}), "
          f"reward={reward_kind}(sigma={reward_sigma}), G={args.G}")
    print(f"K1 targets: {k1_targets}")
    print(f"K2 targets: {k2_targets}")

    print("\nbuilding KL grid...")
    mdp = build_gridworld(L=L, p_slip=p_slip, terminals=terminals, gamma=gamma,
                          mu0_kind=mu0_kind, mu0_centers=mu0_centers)
    r_E = shaped_reward_to_goal(L, expert_goal, terminals,
                                kind=reward_kind, sigma=reward_sigma)
    r_anti = shaped_reward_to_goal(L, anti_goal, terminals,
                                   kind=reward_kind, sigma=reward_sigma)

    alphas = np.linspace(0, 1, args.G)
    betas = np.linspace(0, 1, args.G)

    grid = build_kl_grid(mdp, r_E, r_anti, alphas, betas, tau)
    KL1 = grid["KL1"]
    KL2 = grid["KL2"]

    print(f"\nKL1 range: [{KL1.min():.3f}, {KL1.max():.3f}]")
    print(f"KL2 range (whole grid): [{KL2.min():.3f}, {KL2.max():.3f}]")

    print(f"\n{'K1':>6} {'K2':>6} {'feas':>5} {'alpha*':>8} {'beta*':>8} "
          f"{'K1_real':>9} {'K2_real':>9}  reason")
    print("-" * 90)

    n_total = 0
    n_feasible = 0
    for K1, K2 in product(k1_targets, k2_targets):
        n_total += 1
        res = prescribe(KL1, KL2, alphas, betas, K1, K2)
        if res["feasible"]:
            n_feasible += 1
            print(f"{K1:>6.3g} {K2:>6.3g}  yes  {res['alpha_star']:>8.4f} "
                  f"{res['beta_star']:>8.4f} {res['realized_K1']:>9.3f} "
                  f"{res['realized_K2']:>9.3f}")
        else:
            print(f"{K1:>6.3g} {K2:>6.3g}   no  {'-':>8} {'-':>8} "
                  f"{'-':>9} {'-':>9}  {res['reason']}")

    print(f"\n{n_feasible}/{n_total} targets feasible")


if __name__ == "__main__":
    main()
