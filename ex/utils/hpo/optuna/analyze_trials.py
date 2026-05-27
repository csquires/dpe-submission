"""analyze trial trajectories: best-intermediate vs final, convergence step,
early-peaker candidates that the top-K-by-final ranking would miss.

uses optuna.Trial.intermediate_values -- the per-step `trial.report(score, step)`
recordings written by objective.py's step_cb -- to characterize each trial's
trajectory:

  best        = argmin over reported intermediate values (the trial's actual peak)
  best_step   = step at which `best` occurred
  conv_step   = earliest step within (1 + conv_tol) * best (the "effective n_steps"
                a follow-up training could stop at without losing quality)
  final       = trial.value (the value at the last reported step)
  Delta       = final - best  (positive Delta => the trial peaked earlier than the
                end and then overfit / drifted away from its own best)

usage:
    python -m ex.utils.hpo.optuna.analyze_trials --config <module> --method <name>
"""
import argparse
import logging
import sys

from optuna.trial import TrialState

from ex.utils.hpo.optuna.storage import create_or_load
from ex.utils.hpo.optuna.study_config import load_config


def _summarize(trial, conv_tol: float) -> dict | None:
    """compute trajectory stats for one trial; None if no intermediate values.

    returns dict with trial number, state, final value, best intermediate value,
    step at best, convergence step within conv_tol of best, and the last step
    reached (the trial's effective n_steps).
    """
    iv = trial.intermediate_values
    if not iv:
        return None
    items = sorted(iv.items())  # [(step, value), ...]
    steps, vals = zip(*items)
    best_idx = min(range(len(vals)), key=lambda i: vals[i])
    best_val = vals[best_idx]
    best_step = steps[best_idx]
    target = best_val + abs(best_val) * conv_tol
    conv_step = next((s for s, v in zip(steps, vals) if v <= target), best_step)
    return {
        "trial": trial.number,
        "state": str(trial.state).rsplit(".", 1)[-1],
        "final": trial.value,
        "best": best_val,
        "best_step": best_step,
        "conv_step": conv_step,
        "n_steps_used": steps[-1],
    }


def main() -> int:
    """rank trials by best-intermediate; flag overlooked early peakers vs by-final."""
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="dotted StudyConfig module")
    p.add_argument("--method", required=True, help="method name in config.methods")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--conv-tol", type=float, default=0.01,
                   help="rel. tolerance for convergence step (default 0.01)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.method not in cfg.methods:
        print(f"error: method '{args.method}' not in {cfg.methods}", file=sys.stderr)
        return 1
    study = create_or_load(cfg.experiment, args.method)

    rows = []
    for t in study.trials:
        if t.state not in (TrialState.COMPLETE, TrialState.PRUNED):
            continue
        s = _summarize(t, conv_tol=args.conv_tol)
        if s is not None:
            rows.append(s)
    if not rows:
        print(f"{cfg.experiment}/{args.method}: no trials with intermediate_values")
        return 0

    rows.sort(key=lambda r: r["best"])
    by_best = rows[:args.top_n]
    finals = [r for r in rows if r["state"] == "COMPLETE" and r["final"] is not None]
    finals.sort(key=lambda r: r["final"])
    by_final = finals[:args.top_n]

    print(f"\n=== {cfg.experiment}/{args.method}: top-{args.top_n} trials by BEST intermediate value ===")
    print(f"{'trial':>6} {'state':<8} {'final':>10} {'best':>10} {'best_step':>9} {'conv_step':>9} {'n_ep_used':>9}  Delta(final-best)")
    for r in by_best:
        final_s = f"{r['final']:.5f}" if r["final"] is not None else "(pruned)"
        if r["final"] is not None:
            delta = r["final"] - r["best"]
            delta_s = f"{delta:+.5f}"
        else:
            delta_s = "  n/a   "
        print(f"  {r['trial']:>4d} {r['state']:<8} {final_s:>10} {r['best']:10.5f} {r['best_step']:9d} {r['conv_step']:9d} {r['n_steps_used']:9d}  {delta_s}")

    final_set = {r["trial"] for r in by_final}
    overlooked = [r for r in by_best if r["trial"] not in final_set]
    if overlooked:
        print(f"\n=== {len(overlooked)} trial(s) in top-{args.top_n}-by-best but NOT in top-{args.top_n}-by-final ===")
        print("    (early peakers a final-only shortlist would miss; candidates for holdout)")
        for r in overlooked:
            final_s = f"{r['final']:.5f}" if r["final"] is not None else "(pruned)"
            delta = (r["final"] - r["best"]) if r["final"] is not None else None
            delta_s = f"{delta:+.5f}" if delta is not None else "n/a"
            print(f"  trial {r['trial']}: final={final_s} best={r['best']:.5f}@step {r['best_step']}  Delta={delta_s}")
    else:
        print(f"\n(top-{args.top_n}-by-best is a subset of top-{args.top_n}-by-final; no overlooked early peakers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
