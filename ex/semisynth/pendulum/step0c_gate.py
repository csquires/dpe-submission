"""step0c: run the G1-G4 oracle-free acceptance gate.

thin CLI over ex/semisynth/pendulum/step0_common.py (run from repo root
via python -m ex.semisynth.pendulum.step0c_gate); run stages in order
step0a -> step0b -> step0c -> step0d. each stage is idempotent; --force
recomputes. see step0_common for the shared machinery.
"""
from ex.semisynth.pendulum.step0_common import cli, mode_gate

if __name__ == "__main__":
    cli(mode_gate, "run the G1-G4 oracle-free acceptance gate")
