"""HPO analysis pipeline.

Four-stage pipeline per method:
  Stage 1  – Score subsampled primary pool on train_cells → top-10 by mean
             MAE → promote 2 with lowest cross-cell std.
  Stage 2  – Neighbourhood search in secondary broad pool (anchored on
             stage-1 top-10) → top-10 → promote 2 lowest cross-cell std.
  Stage 2.5– [only when <method>_refined24/ exists] Neighbourhood search in
             the refined24 pool (anchored on all promoted-so-far) → top-10 →
             promote 2 lowest cross-cell std.
  Stage 3  – Holdout evaluation of all promoted candidates; ranked by
             holdout cross-cell std; at-least-5 guarantee (see below).

At-least-5 guarantee:
  Promoted candidates = 2 (S1) + 2 (S2) + 0-2 (S2.5) = 4-6.  If fewer than
  MIN_RETURN_K survive to holdout, the pipeline pads with the next-best
  (lowest train_cell_std) candidates from the last completed stage's top-10,
  excluding already-promoted ones, until MIN_RETURN_K are available.

At each stage the top-10 is scanned for dominant HP values: any HP whose
single value appears in >50 % of the top-10 trials is reported.

Primary/secondary pool selection:
  • Method has ≥ REFINED_MIN_TRIALS refined trials → primary = ALL refined,
    secondary = ALL broad.
  • Otherwise → primary = subsample N from broad (N depends on method family),
    secondary = remaining broad.

Usage:
    cd /home/yizhoulu/dpe-submission
    python -m ex.analysis.analyze \\
        --experiment smodice_eldr_estimation \\
        --data-root /data/user_data/yizhoulu/dpe-submission \\
        --output-dir ex/analysis/results

    # both experiments
    for exp in smodice_eldr_estimation elbo; do
        python -m ex.analysis.analyze --experiment $exp
    done
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .config import (
    ANOMALY_DIFF_HP_THRESH,
    ANOMALY_DIFF_MAE_THRESH,
    ANOMALY_SIMILAR_HP_THRESH,
    ANOMALY_SIMILAR_MAE_THRESH,
    CATEGORICAL_DOMINANCE_THRESH,
    EXP_CONFIGS,
    EXCLUDED_METHODS,
    FINAL_SHORTLIST_K,
    MIN_RETURN_K,
    NEIGHBOR_MIN_MATCHES,
    NEIGHBOR_TOL_MAX,
    NEIGHBOR_TOL_START,
    NEIGHBOR_TOL_STEP,
    REFINED_MIN_TRIALS,
    STAGE1_TOP_K,
    STAGE2_TOP_K,
    STAGE_PROMOTE_K,
    SUBSAMPLE_SEED,
    get_subsample_n,
)
from .utils import (
    dominant_categoricals,
    filter_trials_adaptive,
    hp_distance,
    is_method_dir,
    load_trials,
    metric_on_cells,
    neighbor_search,
    split_cells,
    variance_on_cells,
)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_pool(
    pool: list[dict],
    cells: list[tuple],
    metric_key: str,
) -> list[tuple[float, float, dict]]:
    """Score each trial as (mean_mae, cell_std). Drop trials missing any cell."""
    out: list[tuple[float, float, dict]] = []
    for t in pool:
        mean = metric_on_cells(t, cells, metric_key)
        std = variance_on_cells(t, cells, metric_key)
        if mean is not None and std is not None:
            out.append((mean, std, t))
    return out


def _top_k_by_mae(
    scored: list[tuple[float, float, dict]],
    k: int,
) -> list[tuple[float, float, dict]]:
    return sorted(scored, key=lambda x: x[0])[:k]


def _promote(
    top_k: list[tuple[float, float, dict]],
    n: int = STAGE_PROMOTE_K,
) -> list[tuple[float, float, dict]]:
    """From top-K, select *n* with lowest cross-cell std."""
    return sorted(top_k, key=lambda x: x[1])[:n]


def _stage_record(
    top_k: list[tuple[float, float, dict]],
    promoted: list[tuple[float, float, dict]],
    extra: dict | None = None,
) -> dict:
    promoted_ids = {id(t) for _, _, t in promoted}
    rec: dict = {
        **(extra or {}),
        "top_k": len(top_k),
        "promoted": len(promoted),
        "dominant_categoricals": dominant_categoricals(
            [t for _, _, t in top_k], threshold=CATEGORICAL_DOMINANCE_THRESH
        ),
        "top_trials": [
            {
                "trial_id": t["trial_id"],
                "variant": t.get("pilot_variant"),
                "train_mae": round(mean, 6),
                "train_cell_std": round(std, 6),
                "promoted": id(t) in promoted_ids,
                "hyperparams": t["hyperparams"],
            }
            for mean, std, t in top_k
        ],
    }
    return rec


# ---------------------------------------------------------------------------
# Per-method analysis
# ---------------------------------------------------------------------------

def analyze_method(
    method: str,
    method_dir: Path,
    train_cells: list[tuple],
    holdout_cells: list[tuple],
    metric_key: str,
    refined24_dir: Path | None = None,
    subsample_seed: int = SUBSAMPLE_SEED,
) -> dict:
    """Run the full pipeline for one (method, experiment) pair."""
    broad = filter_trials_adaptive(load_trials(method_dir, "broad"), method)
    refined = filter_trials_adaptive(load_trials(method_dir, "refined"), method)
    refined24 = (
        filter_trials_adaptive(load_trials(refined24_dir, "broad"), method)
        if refined24_dir is not None and refined24_dir.exists()
        else []
    )

    result: dict = {
        "method": method,
        "n_broad": len(broad),
        "n_refined": len(refined),
        "n_refined24": len(refined24),
        "strategy": "",
        "stages": {},
        "shortlist": [],
        "variance": {},
        "anomalies": {"similar_hp_diff_mae": [], "diff_hp_similar_mae": []},
    }

    if not broad and not refined:
        result["error"] = "no trials found"
        return result

    # ── pool assignment ──────────────────────────────────────────────────────
    if len(refined) >= REFINED_MIN_TRIALS:
        primary_pool = refined
        secondary_pool = broad
        result["strategy"] = (
            f"refined_primary: all {len(refined)} refined as stage-1 pool, "
            f"{len(broad)} broad as stage-2 search pool"
        )
    else:
        n = min(get_subsample_n(method), len(broad))
        rng = random.Random(subsample_seed)
        indices = set(rng.sample(range(len(broad)), n))
        primary_pool = [broad[i] for i in range(len(broad)) if i in indices]
        secondary_pool = [broad[i] for i in range(len(broad)) if i not in indices]
        result["strategy"] = (
            f"broad_subsample: sampled {n}/{len(broad)} for stage-1, "
            f"{len(secondary_pool)} remaining for stage-2"
        )

    # ── stage 1 ─────────────────────────────────────────────────────────────
    scored_s1 = _score_pool(primary_pool, train_cells, metric_key)
    top10_s1 = _top_k_by_mae(scored_s1, STAGE1_TOP_K)
    promoted_s1 = _promote(top10_s1)
    result["stages"]["stage1"] = _stage_record(
        top10_s1, promoted_s1,
        {"pool_size": len(primary_pool), "scored": len(scored_s1)},
    )

    # ── stage 2: NN in secondary broad pool ─────────────────────────────────
    neighbors_s2, tol_s2 = neighbor_search(
        top_trials=[t for _, _, t in top10_s1],
        candidate_pool=secondary_pool,
        start_tol=NEIGHBOR_TOL_START,
        step=NEIGHBOR_TOL_STEP,
        max_tol=NEIGHBOR_TOL_MAX,
        min_matches=NEIGHBOR_MIN_MATCHES,
    )
    scored_s2 = _score_pool(neighbors_s2, train_cells, metric_key)
    top10_s2 = _top_k_by_mae(scored_s2, STAGE2_TOP_K)
    promoted_s2 = _promote(top10_s2)
    result["stages"]["stage2"] = _stage_record(
        top10_s2, promoted_s2,
        {
            "tolerance_used": tol_s2,
            "neighbors_found": len(neighbors_s2),
            "scored": len(scored_s2),
        },
    )

    all_promoted = promoted_s1 + promoted_s2
    last_top10 = top10_s2 if top10_s2 else top10_s1

    # ── stage 2.5: NN in refined24 pool (if available) ──────────────────────
    if refined24:
        neighbors_r24, tol_r24 = neighbor_search(
            top_trials=[t for _, _, t in all_promoted],
            candidate_pool=refined24,
            start_tol=NEIGHBOR_TOL_START,
            step=NEIGHBOR_TOL_STEP,
            max_tol=NEIGHBOR_TOL_MAX,
            min_matches=NEIGHBOR_MIN_MATCHES,
        )
        scored_r24 = _score_pool(neighbors_r24, train_cells, metric_key)
        top10_r24 = _top_k_by_mae(scored_r24, STAGE1_TOP_K)
        promoted_r24 = _promote(top10_r24)
        result["stages"]["stage2_refined24"] = _stage_record(
            top10_r24, promoted_r24,
            {
                "tolerance_used": tol_r24,
                "neighbors_found": len(neighbors_r24),
                "scored": len(scored_r24),
            },
        )
        all_promoted = all_promoted + promoted_r24
        last_top10 = top10_r24 if top10_r24 else last_top10

    # ── deduplicate promoted by object identity ──────────────────────────────
    seen_ids: set[int] = set()
    unique_promoted: list[tuple[float, float, dict]] = []
    for item in all_promoted:
        tid = id(item[2])
        if tid not in seen_ids:
            seen_ids.add(tid)
            unique_promoted.append(item)

    # ── pad to MIN_RETURN_K if needed ────────────────────────────────────────
    if len(unique_promoted) < MIN_RETURN_K:
        # sort last stage's top-10 by train_cell_std; add non-promoted entries
        for mean, std, t in sorted(last_top10, key=lambda x: x[1]):
            if id(t) not in seen_ids:
                seen_ids.add(id(t))
                unique_promoted.append((mean, std, t))
            if len(unique_promoted) >= MIN_RETURN_K:
                break

    # ── stage 3: holdout evaluation ─────────────────────────────────────────
    holdout_scored: list[dict] = []
    for mean_train, std_train, t in unique_promoted:
        hold_mean = metric_on_cells(t, holdout_cells, metric_key)
        hold_std = variance_on_cells(t, holdout_cells, metric_key)
        if hold_mean is not None and hold_std is not None:
            holdout_scored.append({
                "trial_id": t["trial_id"],
                "variant": t.get("pilot_variant"),
                "train_mae": round(mean_train, 6),
                "train_cell_std": round(std_train, 6),
                "holdout_mae": round(hold_mean, 6),
                "holdout_cell_std": round(hold_std, 6),
                "hyperparams": t["hyperparams"],
            })

    # primary sort: holdout cross-cell std (variance goal); tiebreak: holdout mae
    holdout_scored.sort(key=lambda x: (x["holdout_cell_std"], x["holdout_mae"]))

    result["stages"]["stage3"] = {
        "candidates_in": len(unique_promoted),
        "scored": len(holdout_scored),
    }
    shortlist = holdout_scored[:FINAL_SHORTLIST_K]
    result["shortlist"] = shortlist

    # ── summary variance ─────────────────────────────────────────────────────
    if shortlist:
        maes = [e["holdout_mae"] for e in shortlist]
        stds = [e["holdout_cell_std"] for e in shortlist]
        mean_mae = sum(maes) / len(maes)
        result["variance"] = {
            "n": len(shortlist),
            "mean_holdout_mae": round(mean_mae, 6),
            "std_holdout_mae": round(
                (sum((x - mean_mae) ** 2 for x in maes) / len(maes)) ** 0.5, 6
            ),
            "min_holdout_mae": round(min(maes), 6),
            "min_holdout_cell_std": round(min(stds), 6),
        }

        # ── anomaly detection ────────────────────────────────────────────────
        similar_hp_diff_mae: list[dict] = []
        diff_hp_similar_mae: list[dict] = []
        for i in range(len(shortlist)):
            for j in range(i + 1, len(shortlist)):
                a, b = shortlist[i], shortlist[j]
                dist = hp_distance(a["hyperparams"], b["hyperparams"])
                rel_mae_diff = abs(a["holdout_mae"] - b["holdout_mae"]) / (mean_mae + 1e-8)
                if dist <= ANOMALY_SIMILAR_HP_THRESH and rel_mae_diff > ANOMALY_DIFF_MAE_THRESH:
                    similar_hp_diff_mae.append({
                        "trial_ids": [a["trial_id"], b["trial_id"]],
                        "hp_dist": round(dist, 4),
                        "rel_mae_diff": round(rel_mae_diff, 4),
                        "holdout_maes": [a["holdout_mae"], b["holdout_mae"]],
                    })
                if dist > ANOMALY_DIFF_HP_THRESH and rel_mae_diff < ANOMALY_SIMILAR_MAE_THRESH:
                    diff_hp_similar_mae.append({
                        "trial_ids": [a["trial_id"], b["trial_id"]],
                        "hp_dist": round(dist, 4),
                        "rel_mae_diff": round(rel_mae_diff, 4),
                        "holdout_maes": [a["holdout_mae"], b["holdout_mae"]],
                    })
        result["anomalies"]["similar_hp_diff_mae"] = similar_hp_diff_mae
        result["anomalies"]["diff_hp_similar_mae"] = diff_hp_similar_mae

    return result


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    experiment: str,
    data_root: Path,
    output_dir: Path,
    methods: list[str] | None = None,
) -> dict:
    """Run the full pipeline for *experiment*. Returns the results dict."""
    cfg = EXP_CONFIGS[experiment]
    metric_key = cfg["metric_key"]
    n_train = cfg["n_train_cells"]
    cell_split_seed = cfg["cell_split_seed"]

    exp_dir = data_root / experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Derive canonical cell list from the first available trial.
    all_cells: list[tuple] | None = None
    for mdir in sorted(exp_dir.iterdir()):
        if not is_method_dir(mdir) or mdir.name.endswith("_refined24"):
            continue
        for variant in ("broad", "refined"):
            trials = load_trials(mdir, variant)
            if trials:
                all_cells = [tuple(c) for c in trials[0].get("training_cells", [])]
                break
        if all_cells is not None:
            break

    if all_cells is None:
        raise RuntimeError(f"No trial data found under {exp_dir}")

    train_cells, holdout_cells = split_cells(all_cells, n_train, seed=cell_split_seed)
    print(
        f"[{experiment}] {len(all_cells)} cells → "
        f"{len(train_cells)} train + {len(holdout_cells)} holdout  "
        f"(metric: {metric_key})"
    )

    method_dirs = sorted(
        d for d in exp_dir.iterdir()
        if is_method_dir(d) and not d.name.endswith("_refined24")
    )
    if methods:
        method_dirs = [d for d in method_dirs if d.name in methods]

    results: dict = {
        "experiment": experiment,
        "data_root": str(data_root),
        "metric_key": metric_key,
        "n_train_cells": len(train_cells),
        "n_holdout_cells": len(holdout_cells),
        "train_cells": [list(c) for c in train_cells],
        "holdout_cells": [list(c) for c in holdout_cells],
        "methods": {},
    }

    for mdir in method_dirs:
        method = mdir.name
        if method in EXCLUDED_METHODS:
            print(f"  {method:<35}  (excluded)")
            continue
        refined24_dir = exp_dir / f"{method}_refined24"
        has_r24 = refined24_dir.exists()
        print(f"  {method:<35} {'[+r24]' if has_r24 else '      '}", end="", flush=True)
        try:
            res = analyze_method(
                method=method,
                method_dir=mdir,
                train_cells=train_cells,
                holdout_cells=holdout_cells,
                metric_key=metric_key,
                refined24_dir=refined24_dir if has_r24 else None,
            )
        except Exception as exc:
            res = {"method": method, "error": str(exc)}
            print(f" ERROR: {exc}")
        else:
            n_short = len(res.get("shortlist", []))
            strat = res.get("strategy", "")[:40]
            print(f" shortlist={n_short}  [{strat}]")
        results["methods"][method] = res

    output_dir.mkdir(parents=True, exist_ok=True)
    # out_path = output_dir / f"{experiment}_analysis.json"
    # out_path.write_text(json.dumps(results, indent=2))
    # print(f"\nFull results  → {out_path}")

    summary_path = output_dir / f"{experiment}_std_noDRE.json"
    summary = _build_summary(results)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {summary_path}")

    _print_summary(results)
    return results


def _build_summary(results: dict) -> dict:
    summary: dict = {
        "experiment": results["experiment"],
        "metric": results["metric_key"],
        "methods": {},
    }
    # collect all methods with a valid min_holdout_cell_std, sort by it
    method_rows = []
    for method, res in results["methods"].items():
        if "error" in res:
            method_rows.append((float("inf"), method, res))
        else:
            std = res.get("variance", {}).get("min_holdout_cell_std", float("inf"))
            method_rows.append((std, method, res))
    method_rows.sort(key=lambda x: x[0])

    for _, method, res in method_rows:
        if "error" in res:
            summary["methods"][method] = {"error": res["error"]}
            continue
        shortlist = res.get("shortlist", [])
        # within each method shortlist: rank by holdout_mae ascending
        shortlist_sorted = sorted(shortlist, key=lambda s: s["holdout_mae"])
        entry: dict = {
            "variance": res.get("variance", {}),
            "shortlist": [
                {
                    "rank": i + 1,
                    "holdout_mae": s["holdout_mae"],
                    "holdout_cell_std": s["holdout_cell_std"],
                    "train_mae": s["train_mae"],
                    "train_cell_std": s["train_cell_std"],
                    "trial_id": s["trial_id"],
                    "variant": s.get("variant"),
                    "hyperparams": s["hyperparams"],
                }
                for i, s in enumerate(shortlist_sorted)
            ],
            "dominant_categoricals_per_stage": {
                stage: info.get("dominant_categoricals", {})
                for stage, info in res.get("stages", {}).items()
                if stage != "stage3"
            },
        }
        summary["methods"][method] = entry
    return summary


def _print_summary(results: dict) -> None:
    exp = results["experiment"]
    print(f"\n{'='*70}")
    print(f"SUMMARY  {exp}")
    print(f"{'='*70}")
    print(f"  {'Method':<35} {'MinCellStd':>10} {'BestMAE':>10} {'N':>3}  Strategy")
    print(f"  {'-'*68}")
    rows = []
    for method, res in results["methods"].items():
        if "error" in res:
            rows.append((float("inf"), method, res))
        else:
            std = res.get("variance", {}).get("min_holdout_cell_std", float("nan"))
            rows.append((std, method, res))
    rows.sort(key=lambda x: x[0])
    for _, method, res in rows:
        if "error" in res:
            print(f"  {method:<35}  ERROR: {res['error']}")
            continue
        var = res.get("variance", {})
        best = var.get("min_holdout_mae", float("nan"))
        min_std = var.get("min_holdout_cell_std", float("nan"))
        n = var.get("n", 0)
        strat = "refined" if "refined_primary" in res.get("strategy", "") else "broad"
        print(f"  {method:<35} {min_std:>10.5f} {best:>10.5f} {n:>3}  {strat}")

    # dominant categoricals summary
    print(f"\n  {'Method':<35}  Dominant HPs per stage")
    print(f"  {'-'*73}")
    for method, res in results["methods"].items():
        if "error" in res:
            continue
        stage_cats = {
            stage: info.get("dominant_categoricals", {})
            for stage, info in res.get("stages", {}).items()
            if stage != "stage3"
        }
        non_empty = {s: d for s, d in stage_cats.items() if d}
        if non_empty:
            parts = []
            for stage, cats in non_empty.items():
                for hp, info in cats.items():
                    parts.append(f"{stage}/{hp}={info['value']} ({info['fraction']:.0%})")
            print(f"  {method:<35}  {', '.join(parts)}")

    total_sim = sum(
        len(r.get("anomalies", {}).get("similar_hp_diff_mae", []))
        for r in results["methods"].values()
        if "error" not in r
    )
    total_diff = sum(
        len(r.get("anomalies", {}).get("diff_hp_similar_mae", []))
        for r in results["methods"].values()
        if "error" not in r
    )
    print(f"\n  Anomalies: similar-HP/different-MAE={total_sim}  "
          f"different-HP/similar-MAE={total_diff}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HPO analysis pipeline: subsample → top-10 → NN → [refined24 NN] → holdout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=list(EXP_CONFIGS.keys()),
        help="Experiment to analyse",
    )
    parser.add_argument(
        "--data-root",
        default="/data/user_data/yizhoulu/dpe-submission",
        help="NFS root containing experiment directories (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="ex/analysis/results",
        help="Directory to write output JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Restrict analysis to these methods (default: all)",
    )
    args = parser.parse_args()

    run_experiment(
        experiment=args.experiment,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        methods=args.methods,
    )


if __name__ == "__main__":
    main()
