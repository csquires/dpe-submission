"""assemble ex/semisynth/mnist_uncond/winners.yaml from HPO holdout best_hp.json.

schema A (what the step2_runner's load_winners.resolve_hp reads):
  methods:
    <step2_method_name>:
      hyperparams: { ...best_hp verbatim... }
      score: { ...provenance... }

one GLOBAL winner per method (slices=None) -> a single `hyperparams`, NO
per_bucket: the runner's bucket_for_cell returns "alpha_idx_<a>", and schema-A
resolve_hp falls through the (absent) per_bucket to the global hyperparams for
every alpha. best_step is -1 (train to full n_steps), no splicing.

covers BOTH mnist_uncond configs -> all 18 methods.

usage (env fac, DPE_DATA_ROOT set, cwd=repo, AFTER the holdouts):
  python -m ex.semisynth.mnist_uncond.assemble_winners
"""
import glob
import json
import os

import yaml

from ex.utils.hpo.optuna.study_config import load_config

_CONFIG_MODULES = [
    "ex.utils.hpo.optuna.configs.mnist_uncond_avi",      # cls + TSM + CTSM (cpu array)
    "ex.utils.hpo.optuna.configs.mnist_uncond_gpu_avi",  # VFM + FMDRE (array_gpu)
]
_OUT = "ex/semisynth/mnist_uncond/winners.yaml"


def main() -> None:
    root = os.environ["DPE_DATA_ROOT"]
    methods_block: dict = {}
    missing: list[str] = []
    for mod in _CONFIG_MODULES:
        cfg = load_config(mod)
        for m in cfg.methods:
            hits = sorted(glob.glob(
                f"{root}/holdout/mnist_uncond/{m}/**/best_hp.json", recursive=True
            ))
            if not hits:
                missing.append(m)
                continue
            best = json.load(open(hits[0]))
            hp = dict(best["best_hp"])
            hp.update(cfg.fixed_hp or {})
            methods_block[m] = {
                "hyperparams": hp,
                "score": {
                    "best_value_median": best.get("best_value_median"),
                    "winner_trial_number": best.get("winner_trial_number"),
                    "source": "hpo_holdout_global",
                },
            }
            print(f"{m:26s} median={best.get('best_value_median')}")

    if missing:
        print(f"\nWARNING: no best_hp.json for {len(missing)}: {missing}")

    with open(_OUT, "w") as f:
        yaml.safe_dump({"methods": methods_block}, f,
                       sort_keys=False, default_flow_style=False)
    print(f"\nwrote {_OUT} with {len(methods_block)} methods (schema A)")


if __name__ == "__main__":
    main()
