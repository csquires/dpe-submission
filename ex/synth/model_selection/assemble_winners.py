"""assemble winners.yaml from HPO holdout best_hp.json for a specified variant.

schema A (what load_winners.resolve_hp reads first):
  methods:
    <step2_method_name>:
      hyperparams: { ...best_hp verbatim... }
      score: { ... provenance ... }

one GLOBAL winner per method (slices=None) -> a single `hyperparams`, NO
per_bucket: every kl_idx_* bucket falls through to it. best_hp is copied
VERBATIM (it already uses the current builder keys: n_steps/learning_rate/...),
best_step is -1 (train to full n_steps) so no splicing. only name that differs
between the holdout dir and step2 is MDRE -> MDRE_15.

variant-aware: each tag has isolated data paths and HPO StudyConfig modules.
prevents cross-variant contamination by globbing with correct experiment_name(tag)
and enumerating methods from hpo_config_modules(tag).

usage (env fac, DPE_DATA_ROOT set, cwd=repo, AFTER the holdouts):
  python -m ex.synth.model_selection.assemble_winners
  python -m ex.synth.model_selection.assemble_winners --variant tr8192_te4096
"""
import argparse
import glob
import json
import os

import yaml

from ex.synth.model_selection.variants import (
    experiment_name,
    hpo_config_modules,
    winners_path,
)
from ex.utils.hpo.optuna.study_config import load_config

_NAME_MAP = {"MDRE": "MDRE_15"}  # only name that differs from step2/METHOD_SPECS


def main(variant: str | None = None) -> None:
    """assemble winners.yaml from HPO holdout best_hp.json for the specified variant.

    Args:
        variant: tag in VARIANTS (e.g., "tr8192_te2048"). If None, derives from env or default.

    Procedure:
        1. Resolve variant via variants.resolve(variant) -> (tag, config_dict).
        2. Compute experiment_name = experiment_name(tag).
        3. Load HPO StudyConfig modules via hpo_config_modules(tag); enumerate all methods.
        4. For each method:
           - Glob: $DPE_DATA_ROOT/holdout/{experiment_name}/{method}/**/best_hp.json
           - HARD ERROR (raise) if 0 hits: display exact glob pattern + diagnostic.
           - HARD ERROR (raise) if >1 hit: list all files (stale results from earlier run).
           - Load best_hp.json, merge fixed_hp, apply _NAME_MAP, write to methods_block.
        5. Write methods_block to variants.winners_path(tag), creating winners/ dir as needed.
        6. Print summary: method -> score, final output path.
    """
    from ex.synth.model_selection import variants

    # (1) resolve variant
    tag, _ = variants.resolve(variant)
    exp_name = experiment_name(tag)

    # (2) load HPO modules and enumerate methods
    methods_block: dict = {}
    missing: list[tuple[str, str]] = []  # (method, glob_pattern)
    duplicates: list[tuple[str, list[str]]] = []  # (method, [paths])

    root = os.environ["DPE_DATA_ROOT"]

    for mod in hpo_config_modules(tag):
        cfg = load_config(mod)
        for m in cfg.methods:
            pattern = f"{root}/holdout/{exp_name}/{m}/**/best_hp.json"
            hits = sorted(glob.glob(pattern, recursive=True))

            # (3) validate glob result
            if not hits:
                missing.append((m, pattern))
                continue

            if len(hits) > 1:
                duplicates.append((m, hits))
                continue

            # (4) load and merge
            best = json.load(open(hits[0]))
            hp = dict(best["best_hp"])
            hp.update(cfg.fixed_hp or {})
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

    # (5) fail loudly on errors
    if missing:
        m, pat = missing[0]
        raise RuntimeError(
            f"holdout never ran for variant {tag!r}, method {m!r}.\n"
            f"Expected: {pat}\n"
            f"All missing: {[name for name, _ in missing]}"
        )

    if duplicates:
        m, paths = duplicates[0]
        raise RuntimeError(
            f"stale holdout results for variant {tag!r}, method {m!r}; >1 hit.\n"
            f"Files:\n" + "\n".join(f"  {p}" for p in paths)
        )

    # (6) write output
    out_path = winners_path(tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = {"methods": methods_block}
    with open(out_path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)

    print(f"\nwrote {out_path} with {len(methods_block)} methods")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="assemble winners.yaml from HPO holdout results"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="variant tag (e.g., tr8192_te2048); if None, use env DPE_MS_VARIANT or default"
    )
    args = parser.parse_args()
    main(variant=args.variant)
