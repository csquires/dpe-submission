"""assemble ex/synth/elbo/winners.yaml from HPO holdout best_hp.json.

schema A (what the step2_runner's load_winners.resolve_hp reads):
  methods:
    <step2_method_name>:
      hyperparams: { ...best_hp verbatim... }
      score: { ...provenance... }

one GLOBAL winner per method (the elbo campaign runs slices=None), so a single
`hyperparams` block and NO per_bucket: the elbo step2 adapter's bucket_for_cell
returns None, and schema-A resolve_hp falls through to the global hyperparams for
every row. best_step is -1 (train to full n_steps), no splicing.

MDRE -> MDRE_15 is the only rename: the HPO campaign inherits eig.py's method
list, which calls it "MDRE", but step2 resolves builders through METHOD_SPECS
where the key is "MDRE_15". Without this map step2 cannot resolve that method.

usage (env fac, DPE_DATA_ROOT set, cwd=repo, AFTER the holdouts finish):
  python -m ex.synth.elbo.assemble_winners [--allow-missing]
"""
import argparse
import glob
import json
import os
import sys

import yaml

from ex.utils.hpo.optuna.study_config import load_config

_CONFIG_MODULE = "ex.utils.hpo.optuna.configs.elbo_avi"
_OUT = "ex/synth/elbo/winners.yaml"
_NAME_MAP = {"MDRE": "MDRE_15"}


def main() -> int:
    """read best_hp.json per method, emit schema-A winners.yaml.

    hard-fails when any method lacks best_hp.json unless --allow-missing, so a
    half-finished holdout cannot silently produce a partial winners file that
    step2 would then run on.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allow-missing", action="store_true",
                   help="emit winners.yaml even if some methods lack best_hp.json")
    args = p.parse_args()

    root = os.environ["DPE_DATA_ROOT"]
    cfg = load_config(_CONFIG_MODULE)

    methods_block: dict = {}
    missing: list[str] = []
    for m in cfg.methods:
        hits = sorted(glob.glob(
            f"{root}/holdout/{cfg.experiment}/{m}/**/best_hp.json", recursive=True
        ))
        if not hits:
            missing.append(m)
            continue
        best = json.load(open(hits[0]))
        hp = dict(best["best_hp"])
        hp.update(cfg.fixed_hp or {})  # ensure pinned n_hidden_layers=5 present
        key = _NAME_MAP.get(m, m)
        methods_block[key] = {
            "hyperparams": hp,
            "score": {
                "best_value_median": best.get("best_value_median"),
                "winner_trial_number": best.get("winner_trial_number"),
                "source": "hpo_holdout_global",
            },
        }
        print(f"{m:26s} -> {key:16s} median={best.get('best_value_median')}")

    if missing:
        msg = f"no best_hp.json for {len(missing)}/{len(cfg.methods)}: {missing}"
        if not args.allow_missing:
            print(f"\nERROR: {msg}\n"
                  f"holdout is likely still running; re-run when it completes, "
                  f"or pass --allow-missing to emit a partial file.", file=sys.stderr)
            return 1
        print(f"\nWARNING: {msg}")

    with open(_OUT, "w") as f:
        yaml.safe_dump({"methods": methods_block}, f,
                       sort_keys=False, default_flow_style=False)
    print(f"\nwrote {_OUT} with {len(methods_block)} methods (schema A)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
