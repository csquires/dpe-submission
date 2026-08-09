"""step0d: measure realized K1 for chosen alphas and stamp alphas_chosen.yaml.

thin CLI over ex/semisynth/pendulum/step0_common.py (run from repo root
via python -m ex.semisynth.pendulum.step0d_stamp_strata); run stages in order
step0a -> step0b -> step0c -> step0d. each stage is idempotent; --force
recomputes. see step0_common for the shared machinery.
"""
from ex.semisynth.pendulum.step0_common import cli, mode_stamp_strata

if __name__ == "__main__":
    cli(mode_stamp_strata, "measure realized K1 for chosen alphas and stamp alphas_chosen.yaml")
