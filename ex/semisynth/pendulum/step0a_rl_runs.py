"""step0a: collect RL behavior replays (expert + chosen alphas, all seeds).

thin CLI over ex/semisynth/pendulum/step0_common.py (run from repo root
via python -m ex.semisynth.pendulum.step0a_rl_runs); run stages in order
step0a -> step0b -> step0c -> step0d. each stage is idempotent; --force
recomputes. see step0_common for the shared machinery.
"""
from ex.semisynth.pendulum.step0_common import cli, mode_rl

if __name__ == "__main__":
    cli(mode_rl, "collect RL behavior replays (expert + chosen alphas, all seeds)")
