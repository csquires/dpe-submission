"""assemble winners.yaml (one per route) from HPO holdout best_hp.json for dokls ablation.

schema A (what load_winners.resolve_hp reads first):
  methods:
    <step2_method_name>:
      hyperparams: { ...best_hp verbatim... }
      score: { ... provenance ... }

one GLOBAL winner per method (slices=None) -> a single `hyperparams`, NO per_bucket:
every kl_idx_* bucket falls through to it. best_hp is copied VERBATIM (it already
uses the current builder keys: n_steps/learning_rate/...), best_step is -1 (train to
full n_steps) so no splicing.

route-aware: each (tag, route) pair has isolated data paths and HPO StudyConfig modules.
prevents cross-route contamination by globbing with correct experiment_name(tag, route)
and enumerating methods from hpo_config_modules(tag, route). for a given tag, this script
builds TWO independent winners files (one per route) since two_leg and direct are fully
separate HPO experiments.

usage (env fac, DPE_DATA_ROOT set, cwd=repo, AFTER the holdouts):
  python -m ex.ablations.dokls.assemble_winners
  python -m ex.ablations.dokls.assemble_winners --variant q0_N1024
"""
import argparse
import glob
import json
import os
from pathlib import Path

import yaml

from ex.ablations.dokls import variants
from ex.utils.hpo.optuna.study_config import load_config

_NAME_MAP = {}  # placeholder; dokls methods do not rename


def main(variant: str | None = None, route: str | None = None) -> None:
    """assemble winners.yaml (one per route) from HPO holdout best_hp.json for the specified tag.

    Procedure: resolve tag via variants.resolve(variant); for EACH route in variants.ROUTES, glob
    $DPE_DATA_ROOT/holdout/{experiment_name(tag, route)}/{method}/**/best_hp.json for each method in
    hpo_config_modules(tag, route); hard-error on 0 or >1 hits; merge fixed_hp; write
    winners_path(tag, route).

    Args:
        variant: tag in VARIANTS (e.g., "q0_N1024"). If None, resolves from
            env (DPE_DOKLS_VARIANT) or default (variants.DEFAULT_TAG).
    """
    # step 1: resolve tag
    tag, _, _ = variants.resolve(variant)

    # ensure DPE_DATA_ROOT is set
    if "DPE_DATA_ROOT" not in os.environ:
        raise RuntimeError("DPE_DATA_ROOT not set")

    root = os.environ["DPE_DATA_ROOT"]

    # step 2-7: for each route, enumerate methods and assemble. `route` selects a
    # single route; the dokls campaign only ran two_leg (the direct-leg
    # comparators are reused from ex.synth.model_selection), and globbing an
    # unrun route would hard-error on 0 hits.
    routes = variants.ROUTES if route is None else [route]
    for route in routes:
        exp_name = variants.experiment_name(tag, route)

        # enumerate methods from HPO config modules
        methods_all = []
        mods = variants.hpo_config_modules(tag, route)

        for mod_path in mods:
            cfg = load_config(mod_path)
            for m in cfg.methods:
                if m not in methods_all:
                    methods_all.append(m)

        # step 3-7: glob, validate, load, and assemble winners for this route
        methods_block = {}
        missing = []  # (method, pattern)
        duplicates = []  # (method, [hit_paths])

        for m in methods_all:
            pattern = f"{root}/holdout/{exp_name}/{m}/**/best_hp.json"
            hits = sorted(glob.glob(pattern, recursive=True))

            # step 3: validate
            if len(hits) == 0:
                missing.append((m, pattern))
                continue

            if len(hits) > 1:
                duplicates.append((m, hits))
                continue

            # step 4: load and merge
            best_file = hits[0]
            best_data = json.load(open(best_file))
            hp = dict(best_data["best_hp"])

            # merge fixed_hp from HPO config (overwrite hp on collision)
            for mod_path in mods:
                cfg = load_config(mod_path)
                if m in cfg.methods and cfg.fixed_hp:
                    hp.update(cfg.fixed_hp)

            # apply name mapping (if any; no known dokls renames yet)
            key = _NAME_MAP.get(m, m)

            # step 5: build score provenance
            methods_block[key] = {
                "hyperparams": hp,
                "score": {
                    "best_value_median": best_data.get("best_value_median"),
                    "winner_trial_number": best_data.get("winner_trial_number"),
                    "best_step": -1,
                    "source": "hpo_holdout_global",
                },
            }

            # log
            print(
                f"[{route}] {m:26s} -> {key:16s} "
                f"median={best_data.get('best_value_median'):.6f}"
            )

        # step 6: error reporting (per route)
        if missing:
            m, pat = missing[0]
            raise RuntimeError(
                f"holdout never ran for variant {tag!r} route {route!r}, method {m!r}.\n"
                f"Expected: {pat}\n"
                f"DPE_DATA_ROOT={root}\n"
                f"All missing: {[name for name, _ in missing]}"
            )

        if duplicates:
            m, paths = duplicates[0]
            msg = (
                f"stale holdout results for variant {tag!r} route {route!r}, method {m!r}; "
                f">1 hit in glob.\n"
                f"Files:\n" + "\n".join(f"  {p}" for p in paths)
            )
            raise RuntimeError(msg)

        # step 7: write output (per route)
        out_path = variants.winners_path(tag, route)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out = {"methods": methods_block}

        with open(out_path, "w") as f:
            yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)

        print(f"wrote {out_path} with {len(methods_block)} methods ({route})")

    print(f"done: wrote 2 winners files for tag={tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="variant tag (e.g., q0_N1024). "
        "If omitted, resolves from env or default.",
    )
    parser.add_argument(
        "--route",
        type=str,
        default=None,
        choices=variants.ROUTES,
        help="assemble only this route. omit for all routes; use two_leg when "
        "the direct comparators come from ex.synth.model_selection.",
    )
    args = parser.parse_args()
    main(args.variant, args.route)
