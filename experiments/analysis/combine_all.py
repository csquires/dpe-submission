"""One-shot script to create all *_combined experiment directories on NFS.

Run on a node with NFS access (e.g. babel-p5-20):
    python3 combine_all.py
"""
from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

BASE = Path("/data/user_data/yizhoulu/dpe-submission")
REPORT_PATH = Path("/home/yizhoulu/dpe-submission/experiments/analysis/results/combine_all_report.json")

# Methods to exclude from model_selection_avi (variant-specific suffixes)
_VARIANT_SUFFIX_RE = re.compile(
    r"_(?:wave3|vfmfix|ctsmfix|retry|fmdre_retry|retry2|retry3)$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_method_dir(path: Path) -> bool:
    return path.is_dir() and ((path / "broad").exists() or (path / "refined").exists())


def load_raw(method_dir: Path, variant: str) -> list[tuple[Path, dict]]:
    d = method_dir / variant
    if not d.exists():
        return []
    pairs = []
    for f in sorted(d.glob("trial_*.json")):
        try:
            pairs.append((f, json.loads(f.read_text())))
        except Exception:
            pass
    return pairs


def dedup_key(trial: dict) -> tuple:
    hp_json = json.dumps(trial.get("hyperparams", {}), sort_keys=True)
    return (trial.get("pilot_variant", "broad"), hp_json, trial.get("eval_sample_seed"))


# ---------------------------------------------------------------------------
# Strategy A: union-with-dedup from multiple sources
# ---------------------------------------------------------------------------

def combine_sources(
    label: str,
    sources: list[tuple[Path, str]],  # (path, src_label)
    output_dir: Path,
) -> dict:
    """Union broad+refined trials from multiple sources, keep best on duplicate."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_methods: set[str] = set()
    for src_path, _ in sources:
        if src_path.exists():
            for d in src_path.iterdir():
                if is_method_dir(d):
                    all_methods.add(d.name)

    method_reports: dict = {}
    for method in sorted(all_methods):
        key_to_best: dict = {}
        key_to_srcs: dict = defaultdict(list)
        per_source: dict[str, int] = {}

        for src_path, src_label in sources:
            mdir = src_path / method
            if not mdir.exists():
                continue
            n = 0
            for variant in ("broad", "refined"):
                for _, trial in load_raw(mdir, variant):
                    key = dedup_key(trial)
                    key_to_srcs[key].append(src_label)
                    score = trial.get("score", float("inf"))
                    if key not in key_to_best:
                        key_to_best[key] = (src_label, trial)
                    elif score < key_to_best[key][1].get("score", float("inf")):
                        key_to_best[key] = (src_label, trial)
                    n += 1
            per_source[src_label] = per_source.get(src_label, 0) + n

        overlap_keys = [
            {
                "hyperparams": json.loads(k[1]),
                "sources": list(set(v)),
            }
            for k, v in key_to_srcs.items()
            if len(set(v)) > 1
        ]
        method_reports[method] = {
            "per_source": per_source,
            "total_unique": len(key_to_best),
            "overlap_count": len(overlap_keys),
            "overlapping_trials": overlap_keys,
        }

        broad_out = output_dir / method / "broad"
        refined_out = output_dir / method / "refined"
        broad_out.mkdir(parents=True, exist_ok=True)
        refined_out.mkdir(parents=True, exist_ok=True)

        bi = ri = 0
        for _, (_, trial) in sorted(key_to_best.items()):
            if trial.get("pilot_variant", "broad") == "refined":
                (refined_out / f"trial_{ri}.json").write_text(json.dumps(trial, indent=2))
                ri += 1
            else:
                (broad_out / f"trial_{bi}.json").write_text(json.dumps(trial, indent=2))
                bi += 1

        print(f"  {method:<35} broad={bi:>4}  refined={ri:>3}  overlap={len(overlap_keys)}")

    return method_reports


# ---------------------------------------------------------------------------
# Strategy B: simple copy (single source, no dedup needed)
# ---------------------------------------------------------------------------

def copy_standard(
    src_path: Path,
    output_dir: Path,
    method_filter=None,
) -> dict:
    """Copy broad/refined/trial_*.json from src_path into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    method_reports: dict = {}

    for mdir in sorted(src_path.iterdir()):
        if not is_method_dir(mdir):
            continue
        if method_filter and not method_filter(mdir.name):
            continue
        per_variant: dict[str, int] = {}
        for variant in ("broad", "refined"):
            src_var = mdir / variant
            if not src_var.exists():
                continue
            dst_var = output_dir / mdir.name / variant
            dst_var.mkdir(parents=True, exist_ok=True)
            count = 0
            for f in sorted(src_var.glob("trial_*.json")):
                shutil.copy2(f, dst_var / f.name)
                count += 1
            if count:
                per_variant[variant] = count
        if per_variant:
            method_reports[mdir.name] = per_variant
            print(f"  {mdir.name:<35} {per_variant}")

    return method_reports


# ---------------------------------------------------------------------------
# Strategy C: extract from legacy hpo_results flat layout (mnist_eldr_estimation)
# ---------------------------------------------------------------------------

def extract_hpo_results(
    src_path: Path,
    output_dir: Path,
) -> dict:
    """Extract hpo_results/<method>/trial_*.json → <method>/broad/trial_*.json.

    Adds missing fields (score, pilot_variant, training_cells, eval_sample_seed)
    so our load_trials() can ingest them.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    hpo_dir = src_path / "hpo_results"
    if not hpo_dir.exists():
        print(f"  WARNING: {hpo_dir} not found")
        return {}

    method_reports: dict = {}
    for mdir in sorted(hpo_dir.iterdir()):
        if not mdir.is_dir():
            continue
        # skip logs and refined subdir at top level
        if mdir.name in ("logs", "refined"):
            continue

        files = sorted(mdir.glob("trial_*.json"))
        if not files:
            continue

        dst = output_dir / mdir.name / "broad"
        dst.mkdir(parents=True, exist_ok=True)
        count = 0
        for i, f in enumerate(files):
            try:
                trial = json.loads(f.read_text())
            except Exception:
                continue
            # Patch missing standard fields
            if "score" not in trial:
                score = trial.get("mean_mae") or trial.get("mean_metric")
                if score is None or not math.isfinite(float(score)):
                    continue
                trial["score"] = float(score)
            if "pilot_variant" not in trial:
                trial["pilot_variant"] = "broad"
            if "eval_sample_seed" not in trial:
                trial["eval_sample_seed"] = 1729
            if "training_cells" not in trial:
                per = trial.get("per_pair_mae", {})
                training_cells = []
                for k in per:
                    parts = k.split(":")
                    training_cells.append([int(p) for p in parts])
                trial["training_cells"] = training_cells
            (dst / f"trial_{i}.json").write_text(json.dumps(trial, indent=2))
            count += 1

        method_reports[mdir.name] = {"broad": count}
        print(f"  {mdir.name:<35} broad={count:>4}")

    return method_reports


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

report: dict = {}

# 1. smodice_eldr_estimation_combined
print("\n=== smodice_eldr_estimation_combined ===")
r = combine_sources(
    "smodice_eldr_estimation_combined",
    [(BASE / "smodice_eldr_estimation", "yizhoulu"),
     (BASE / "smodice_eldr_estimation_avi", "avi")],
    BASE / "smodice_eldr_estimation_combined",
)
report["smodice_eldr_estimation_combined"] = r

# 2. dbpedia_cond_flow_combined
print("\n=== dbpedia_cond_flow_combined ===")
r = combine_sources(
    "dbpedia_cond_flow_combined",
    [(BASE / "dbpedia_cond_flow", "yizhoulu"),
     (BASE / "dbpedia_cond_flow_avi", "avi")],
    BASE / "dbpedia_cond_flow_combined",
)
report["dbpedia_cond_flow_combined"] = r

# 3. mnist_cond_flow_combined  (just avi, no overlap to resolve)
print("\n=== mnist_cond_flow_combined ===")
r = copy_standard(BASE / "mnist_cond_flow_avi", BASE / "mnist_cond_flow_combined")
report["mnist_cond_flow_combined"] = r

# 4. eig_estimation_combined
print("\n=== eig_estimation_combined ===")
r = copy_standard(BASE / "eig_estimation_avi", BASE / "eig_estimation_combined")
report["eig_estimation_combined"] = r

# 5. pendulum_eldr_estimation_combined
print("\n=== pendulum_eldr_estimation_combined ===")
r = copy_standard(BASE / "pendulum_eldr_estimation_avi", BASE / "pendulum_eldr_estimation_combined")
report["pendulum_eldr_estimation_combined"] = r

# 6. dre_sample_complexity_combined
print("\n=== dre_sample_complexity_combined ===")
r = copy_standard(BASE / "dre_sample_complexity", BASE / "dre_sample_complexity_combined")
report["dre_sample_complexity_combined"] = r

# 7. plugin_dre_combined
print("\n=== plugin_dre_combined ===")
r = copy_standard(BASE / "plugin_dre", BASE / "plugin_dre_combined")
report["plugin_dre_combined"] = r

# 8. pstar_sample_complexity_combined
print("\n=== pstar_sample_complexity_combined ===")
r = copy_standard(BASE / "pstar_sample_complexity", BASE / "pstar_sample_complexity_combined")
report["pstar_sample_complexity_combined"] = r

# 9. elbo_estimation_combined
print("\n=== elbo_estimation_combined ===")
r = copy_standard(BASE / "elbo_estimation", BASE / "elbo_estimation_combined")
report["elbo_estimation_combined"] = r

# 10. model_selection_combined  (exclude variant-suffix methods)
print("\n=== model_selection_combined ===")
r = copy_standard(
    BASE / "model_selection_avi",
    BASE / "model_selection_combined",
    method_filter=lambda m: not _VARIANT_SUFFIX_RE.search(m),
)
report["model_selection_combined"] = r

# 11. mnist_eldr_estimation_combined  (legacy flat layout)
print("\n=== mnist_eldr_estimation_combined ===")
r = extract_hpo_results(BASE / "mnist_eldr_estimation_avi", BASE / "mnist_eldr_estimation_combined")
report["mnist_eldr_estimation_combined"] = r

# Write local report
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=2))
print(f"\nOverlap report → {REPORT_PATH}")
print("\nDone.")
