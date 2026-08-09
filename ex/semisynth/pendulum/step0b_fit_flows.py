"""step0b: train 1-D spline flows on the replay streams.

thin CLI over ex/semisynth/pendulum/step0_common.py (run from repo root
via python -m ex.semisynth.pendulum.step0b_fit_flows); run stages in order
step0a -> step0b -> step0c -> step0d. each stage is idempotent; --force
recomputes. see step0_common for the shared machinery.
"""
from ex.semisynth.pendulum.step0_common import cli, mode_flows

if __name__ == "__main__":
    cli(mode_flows, "train 1-D spline flows on the replay streams")
