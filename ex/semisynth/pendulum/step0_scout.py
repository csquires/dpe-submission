"""step0 scout: gate-free realized-K1 sweep over candidate alphas.

thin CLI over ex/semisynth/pendulum/step0_common.py (run from repo root
via python -m ex.semisynth.pendulum.step0_scout); the alpha-iteration
tool. run after step0a/step0b (typically with --config config_scout.yaml);
prints the K1 table + separation advisories and writes scout_k1.yaml.
"""
from ex.semisynth.pendulum.step0_common import cli, mode_scout

if __name__ == "__main__":
    cli(mode_scout, "gate-free realized-K1 sweep over candidate alphas")
