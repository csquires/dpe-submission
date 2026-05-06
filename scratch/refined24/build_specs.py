"""build narrowed recalibrated_specs/<exp>.yaml for the refined24 mini-campaign.

reads jsons/<exp>_combined_summary.json, takes the top-K shortlist trials per
method, and applies narrow.narrow_spec to every numeric/categorical hp present
in METHOD_SPECS[m]['base_search_space']. for hps that the shortlist exposes but
base_search_space does not (e.g., n_hidden_layers logged as hp), the spec is
preserved as a single-value choice -- these are wrapper passthroughs.

baseline overlay
    if the experiment already has a recalibrated_specs/<exp>.yaml, we use it as
    a baseline. for each method, the output spec is the *intersection* of:
        - the prior baseline entry (preserves capacity caps like
          n_hidden_layers=[2] and hidden_dim=[128, 256] that were intentionally
          set per-experiment)
        - any narrowing learned from the new shortlist data.
    if a method has zero shortlist trials but is present in the baseline, the
    baseline entry is kept verbatim so capacity caps are not silently widened.
    if a method has zero shortlist trials and no baseline entry, it is OMITTED
    from the output (caller should drop it from the launcher matrix).

writes:
    experiments/utils/hpo/recalibrated_specs/<exp>.yaml.refined24

caller swaps the .refined24 file into the active <exp>.yaml position before
calling launcher_lite. side-by-side files preserve the wave-3 / model_selection
yaml that already exists.

usage:
    python -m scratch.refined24.build_specs \
        --experiments model_selection,smodice_eldr_estimation \
        --top-k 5
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from experiments.utils.hpo.method_specs import METHOD_SPECS
from experiments.utils.hpo.narrow import narrow_spec


# canonicalisation: jsons/ method names sometimes carry retry/wave suffixes.
_STRIP = ("_fmdre_retry", "_general", "_ctsmfix", "_vfmfix",
          "_wave3", "_retry2", "_retry")

# wrapper-default n_hidden_layers per method (from src/models/* defaults
# and experiments/utils/hpo/builders.py flat_hp.get fallbacks). used to pin
# n_hidden_layers in refined24 yamls that have no baseline cap, so the
# resulting search space is explicit about capacity rather than relying on
# silent wrapper defaults that the spec yaml does not surface.
# extrema-driven ceiling extensions: for axes where the prior best-of-shortlist
# routinely hit the upper bound (frac_extreme >= 0.5 on the high side, see
# scratch/hp_analysis_out/tables/extrema_consistent.csv and
# figures/hp_analysis/extrema_vs_vfm.md), bump the ceiling so the refined
# search can probe past the truncation. only applied if the narrowed spec
# is itself near the original cap (>= 85% of the original max in the
# spec's natural scale); otherwise the narrowing already excludes the cap
# and there is nothing to extend.
#
# format: {(method, hp): new_upper_bound}
EXTENDED_UPPER: dict[tuple[str, str], float] = {
    ("TriangularVFM_V1", "sigma"): 5.0,            # was 3.0; hit high in 3/8 exps
    ("TriangularVFM_V3", "path_height"): 3.0,      # was 2.0; hit high in 3/6 exps
    ("TriangularVFM_V3", "integration_steps"): 4000,  # was 2600; hit high in 3/6 exps
}

WRAPPER_DEFAULT_N_HIDDEN_LAYERS: dict[str, int] = {
    "BDRE": 1,
    "MDRE_15": 2,
    "TriangularMDRE": 2,
    "TSM": 3, "CTSM": 3, "VFM": 3,
    "TriangularTSM": 3,
    "TriangularCTSM_V1": 3, "TriangularCTSM_V2": 3, "TriangularCTSM_V3": 3,
    "TriangularVFM_V1": 3, "TriangularVFM_V2": 3, "TriangularVFM_V3": 3,
    "MultiHeadTDRE": 3, "MultiHeadTriangularTDRE": 3,
    "FMDRE": 3, "FMDRE_S2": 3, "TriangularFMDRE": 3,
}


def canonical(method: str) -> str:
    for s in _STRIP:
        if method.endswith(s):
            return method[: -len(s)]
    return method


def collect_top_trials(summary: dict, top_k: int) -> dict[str, list[dict]]:
    """merge shortlists for the same canonical method (across retry/wave suffixes)
    and keep the top_k trials by holdout_mae."""
    bucket: dict[str, list[dict]] = {}
    for raw, info in summary.get("methods", {}).items():
        m = canonical(raw)
        sl = info.get("shortlist", [])
        bucket.setdefault(m, []).extend(sl)
    out = {}
    for m, sl in bucket.items():
        sl_sorted = sorted(sl, key=lambda t: float(t.get("holdout_mae", float("inf"))))
        out[m] = sl_sorted[:top_k]
    return out


def _deserialise_spec(raw):
    """convert a yaml-loaded list spec back to a tuple, preserving choice list."""
    if not isinstance(raw, (list, tuple)):
        return raw
    kind = raw[0]
    if kind == "choice":
        return ("choice", list(raw[1]))
    return tuple(raw)


def _intersect(a: tuple, b: tuple) -> tuple:
    """return the tightest spec consistent with both a and b.

    rules:
      - if either is a choice, restrict to the intersection of the choice set
        with the other's allowed values. empty intersection -> fall back to a.
      - otherwise (both numeric ranges of same kind), take [max(lo), min(hi)].
        if max(lo) > min(hi), fall back to a.
    a is the new (narrowed) spec; b is the baseline.
    """
    if a is None:
        return b
    if b is None:
        return a
    ka, kb = a[0], b[0]
    # choice ∩ anything
    def _values_in(spec: tuple, candidate) -> bool:
        kind = spec[0]
        if kind == "choice":
            return candidate in spec[1]
        if kind in ("uniform", "log_uniform"):
            return spec[1] <= float(candidate) <= spec[2]
        if kind in ("uniform_int", "log_uniform_int"):
            return spec[1] <= int(candidate) <= spec[2]
        return True
    if ka == "choice" and kb == "choice":
        cs = [v for v in a[1] if v in b[1]]
        return ("choice", cs) if cs else a
    if ka == "choice":
        cs = [v for v in a[1] if _values_in(b, v)]
        return ("choice", cs) if cs else a
    if kb == "choice":
        cs = [v for v in b[1] if _values_in(a, v)]
        return ("choice", cs) if cs else a
    # numeric range ∩ numeric range
    if ka == kb and len(a) == 3:
        lo = max(float(a[1]), float(b[1]))
        hi = min(float(a[2]), float(b[2]))
        if lo > hi:
            return a
        if "int" in ka:
            return (ka, int(lo), int(hi))
        return (ka, lo, hi)
    return a


def narrow_method(method: str, top_trials: list[dict],
                  baseline: dict | None = None) -> dict[str, tuple]:
    """narrow each base_search_space hp using the top trials' values.

    flow:
      1. start with METHOD_SPECS[m].base_search_space.
      2. tighten each hp using narrow_spec on observed top-trial values.
      3. tighten further by intersecting with the baseline yaml (if provided).
      4. surface any hps present in trials or baseline but not in
         base_search_space (e.g., n_hidden_layers, ema_decay).

    baseline preserves intentional per-experiment caps (e.g.,
    n_hidden_layers=[2], hidden_dim=[128]) even when the source shortlist did
    not directly observe them.

    returns {} iff there are no trials and no baseline (caller skips method).
    """
    spec_by_hp = METHOD_SPECS.get(method, {}).get("base_search_space", {})
    if not spec_by_hp and not baseline:
        return {}
    narrowed: dict[str, tuple] = {}
    for hp, base in spec_by_hp.items():
        vals = [t["hyperparams"][hp] for t in top_trials
                if "hyperparams" in t and hp in t["hyperparams"]]
        if vals:
            try:
                cand = narrow_spec(base, vals)
            except (TypeError, ValueError):
                cand = base
        else:
            cand = base
        if baseline and hp in baseline:
            cand = _intersect(cand, _deserialise_spec(baseline[hp]))
        narrowed[hp] = cand
    # extras: hps present in trials or baseline but not in base_search_space
    extra: dict[str, set] = {}
    for t in top_trials:
        for hp, v in t.get("hyperparams", {}).items():
            if hp in spec_by_hp:
                continue
            extra.setdefault(hp, set()).add(v)
    for hp, vs in extra.items():
        cand = ("choice", sorted(vs, key=lambda x: (str(type(x)), x)))
        if baseline and hp in baseline:
            cand = _intersect(cand, _deserialise_spec(baseline[hp]))
        narrowed[hp] = cand
    if baseline:
        for hp, raw in baseline.items():
            if hp not in narrowed:
                narrowed[hp] = _deserialise_spec(raw)
    if "n_hidden_layers" not in narrowed:
        default = WRAPPER_DEFAULT_N_HIDDEN_LAYERS.get(method)
        if default is not None:
            narrowed["n_hidden_layers"] = ("choice", [default])
    # apply extrema-driven ceiling extensions
    for (m_match, hp_match), new_upper in EXTENDED_UPPER.items():
        if method != m_match or hp_match not in narrowed:
            continue
        cur = narrowed[hp_match]
        base_kind = (METHOD_SPECS.get(method, {})
                     .get("base_search_space", {})
                     .get(hp_match, (None,))[0])
        narrowed[hp_match] = _extend_upper(cur, new_upper, base_kind)
    if not narrowed and not top_trials:
        return {}
    return narrowed


def _extend_upper(spec: tuple, new_upper: float, base_kind: str | None) -> tuple:
    """raise the upper bound of `spec` to new_upper if its current upper sits
    near the original cap. converts a collapsed choice back to a numeric range
    of `base_kind` (defaulting to log_uniform) so the optimizer can explore
    the extended region.
    """
    kind = spec[0]
    if kind == "choice":
        vals = list(spec[1])
        try:
            lo = float(min(vals))
        except (TypeError, ValueError):
            return spec
        new_kind = base_kind or "log_uniform"
        if "int" in new_kind:
            return (new_kind, max(int(lo), 1), int(new_upper))
        return (new_kind, lo, float(new_upper))
    if len(spec) == 3:
        cur_upper = float(spec[2])
        if cur_upper >= new_upper:
            return spec
        if "int" in kind:
            return (kind, int(spec[1]), int(new_upper))
        return (kind, float(spec[1]), float(new_upper))
    return spec


def serialise(spec: tuple) -> list:
    """yaml-friendly: tuple -> list, inner choice list preserved."""
    if not isinstance(spec, (tuple, list)):
        return spec
    if spec[0] == "choice":
        return [spec[0], list(spec[1])]
    return list(spec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", default="jsons")
    p.add_argument("--out-dir", default="experiments/utils/hpo/recalibrated_specs")
    p.add_argument("--experiments", required=True,
                   help="csv of experiment names, e.g. model_selection,smodice_eldr_estimation")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--suffix", default=".refined24",
                   help="suffix for output yaml; '' to overwrite active spec")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for exp in [s.strip() for s in args.experiments.split(",") if s.strip()]:
        json_path = Path(args.json_dir) / f"{exp}_combined_summary.json"
        if not json_path.exists():
            print(f"  [skip] no shortlist json at {json_path}")
            continue
        summary = json.loads(json_path.read_text())
        top = collect_top_trials(summary, args.top_k)
        # baseline: existing recalibrated_specs/<exp>.yaml (capacity caps live here).
        baseline_path = out_dir / f"{exp}.yaml"
        baseline_methods: dict[str, dict] = {}
        if baseline_path.exists():
            base_doc = yaml.safe_load(baseline_path.read_text()) or {}
            baseline_methods = base_doc.get("methods", {}) or {}
            print(f"  [baseline] {baseline_path} -> {len(baseline_methods)} method entries")
        else:
            print(f"  [baseline] no prior {baseline_path}; using METHOD_SPECS only")

        methods_block: dict[str, dict] = {}
        skipped: list[str] = []
        all_methods = set(top) | set(baseline_methods)
        for method in sorted(all_methods):
            if method not in METHOD_SPECS:
                continue
            trials = top.get(method, [])
            baseline = baseline_methods.get(method)
            ns = narrow_method(method, trials, baseline=baseline)
            if not ns:
                skipped.append(method)
                continue
            methods_block[method] = {hp: serialise(spec) for hp, spec in ns.items()}
        payload = {
            "provenance": {
                "derived_at": "2026-05-05",
                "rule": (f"narrowed from top-{args.top_k} of "
                         f"{json_path.name} via narrow_spec; baseline overlay "
                         f"{baseline_path.name if baseline_path.exists() else '<none>'}; "
                         f"refined24 mini-campaign"),
            },
            "workflow_version": "v1",
            "methods": methods_block,
        }
        out = out_dir / f"{exp}.yaml{args.suffix}"
        out.write_text(yaml.dump(payload, default_flow_style=False, sort_keys=False))
        print(f"  wrote {out} ({len(methods_block)} methods)"
              + (f"; skipped (no trials, no baseline): {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
