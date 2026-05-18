"""progress monitor for optuna hpo studies.

reads the JournalStorage journal of every (experiment, method) study in a
StudyConfig and prints a per-study summary: trial counts by state, terminal
progress vs target_trials, best value/trial, and hyperband rung occupancy
(how many trials reported an intermediate value at each budget step).

read-only with respect to optimization -- it never dispatches or mutates
trials. use it alongside a running keeper to watch a campaign.

cli:
  python -m ex.utils.hpo.optuna.progress --config <dotted.module> [--watch N]
"""
import argparse
import collections
import sys
import time

from optuna.trial import TrialState

from ex.utils.hpo.optuna.study_config import load_config
from ex.utils.hpo.optuna.storage import create_or_load


def study_summary(experiment: str, method: str, target_trials: int) -> str:
    """format a multi-line progress summary for one (experiment, method) study."""
    study = create_or_load(experiment, method)
    trials = study.trials
    by_state = collections.Counter(t.state.name for t in trials)
    terminal = by_state.get("COMPLETE", 0) + by_state.get("PRUNED", 0)
    pct = 100.0 * terminal / target_trials if target_trials else 0.0

    lines = [
        f"  {experiment}/{method}: {terminal}/{target_trials} terminal "
        f"({pct:.0f}%)  [COMPLETE={by_state.get('COMPLETE', 0)} "
        f"PRUNED={by_state.get('PRUNED', 0)} RUNNING={by_state.get('RUNNING', 0)} "
        f"FAIL={by_state.get('FAIL', 0)}]"
    ]

    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    if complete:
        best = min(complete, key=lambda t: t.value)
        lines.append(f"    best: value={best.value:.5f}  (trial #{best.number})")

    # rung occupancy: how many trials reported an intermediate value at each step.
    rungs = collections.Counter()
    for t in trials:
        for step in t.intermediate_values:
            rungs[step] += 1
    if rungs:
        rung_str = "  ".join(f"{s}:{rungs[s]}" for s in sorted(rungs))
        lines.append(f"    rung reports: {rung_str}")
    return "\n".join(lines)


def report(config) -> None:
    """print the progress summary for every study in the config."""
    print(f"=== hpo progress: {config.experiment} "
          f"(target_trials={config.target_trials}) ===")
    for method in config.methods:
        try:
            print(study_summary(config.experiment, method, config.target_trials))
        except Exception as e:
            print(f"  {config.experiment}/{method}: <unreadable: {e}>")


def main() -> int:
    """one-shot report, or refresh every --watch seconds until interrupted."""
    p = argparse.ArgumentParser(description="optuna hpo progress monitor")
    p.add_argument("--config", required=True,
                   help="dotted StudyConfig module path")
    p.add_argument("--watch", type=int, default=0,
                   help="if >0, refresh every N seconds (Ctrl-C to stop)")
    args = p.parse_args()

    config = load_config(args.config)
    if args.watch <= 0:
        report(config)
        return 0

    try:
        while True:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
            report(config)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
