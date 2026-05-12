"""HPO analysis pipeline — per-cell rank variant.

Identical pipeline to analyze.py except the selection criterion at every
stage is **average per-cell MAE rank** instead of cross-cell std.

Per-cell rank score for a trial:
  For each cell in the evaluation set, rank all trials in the pool by their
  MAE on that cell (rank 1 = best).  Average those ranks across cells.  A
  trial with a low average rank is consistently good *relative to the other
  candidates* across cells, even if its absolute MAE is not the lowest.

Stage summary:
  Stage 1   – rank_pool(primary) → top-10 by avg_rank → promote 2 lowest avg_rank
  Stage 2   – NN in secondary broad → rank_pool → top-10 → promote 2
  Stage 2.5 – [if _refined24/ exists] NN in refined24 → rank_pool → top-10 → promote 2
  Stage 3   – rank_pool(promoted, holdout_cells) → sort by holdout avg_rank → return ≥5

Output: <experiment>_rank.json (same schema as _std.json with
        holdout_cell_std replaced by holdout_avg_rank, train_cell_std by
        train_avg_rank).
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
    rank_pool,
    split_cells,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _top_k(scored: list[tuple], k: int) -> list[tuple]:
    """top-k from rank_pool output (already sorted by avg_rank asc)."""
    return scored[:k]


def _promote(top_k: list[tuple], n: int = STAGE_PROMOTE_K) -> list[tuple]:
    """top-k is already sorted by avg_rank asc; just take first n."""
    return top_k[:n]


def _stage_record(
    top_k: list[tuple],
    promoted: list[tuple],
    extra: dict | None = None,
) -> dict:
    promoted_ids = {id(t) for _, _, t in promoted}
    return {
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
                "train_avg_rank": round(avg_rank, 4),
                "train_mae": round(mean_mae, 6),
                "promoted": id(t) in promoted_ids,
                "hyperparams": t["hyperparams"],
            }
            for avg_rank, mean_mae, t in top_k
        ],
    }


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
    refined24_as_primary: bool = False,
) -> dict:
    broad = filter_trials_adaptive(load_trials(method_dir, "broad"), method)
    refined = filter_trials_adaptive(load_trials(method_dir, "refined"), method)
    _r24_raw = (
        load_trials(refined24_dir, "broad")
        if refined24_dir is not None and refined24_dir.exists()
        else []
    )
    # skip compute-budget filter for refined24 when it is the primary pool —
    # those trials are already curated around known-good configs
    refined24 = _r24_raw if refined24_as_primary else filter_trials_adaptive(_r24_raw, method)

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

    # ── pool assignment ──────────────────────────────────────────────────────
    if refined24_as_primary:
        if not refined24:
            result["skip"] = "no refined24 data"
            return result
        primary_pool = refined24
        secondary_pool = broad
        result["strategy"] = (
            f"refined24_primary: {len(refined24)} refined24 trials as stage-1 pool, "
            f"{len(broad)} broad as stage-2 search pool"
        )
    elif len(refined) >= REFINED_MIN_TRIALS:
        primary_pool = refined
        secondary_pool = broad
        result["strategy"] = (
            f"refined_primary: all {len(refined)} refined as stage-1 pool, "
            f"{len(broad)} broad as stage-2 search pool"
        )
    else:
        if not broad and not refined:
            result["error"] = "no trials found"
            return result
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
    ranked_s1 = rank_pool(primary_pool, train_cells, metric_key)
    top10_s1 = _top_k(ranked_s1, STAGE1_TOP_K)
    promoted_s1 = _promote(top10_s1)
    result["stages"]["stage1"] = _stage_record(
        top10_s1, promoted_s1,
        {"pool_size": len(primary_pool), "scored": len(ranked_s1)},
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
    ranked_s2 = rank_pool(neighbors_s2, train_cells, metric_key)
    top10_s2 = _top_k(ranked_s2, STAGE2_TOP_K)
    promoted_s2 = _promote(top10_s2)
    result["stages"]["stage2"] = _stage_record(
        top10_s2, promoted_s2,
        {
            "tolerance_used": tol_s2,
            "neighbors_found": len(neighbors_s2),
            "scored": len(ranked_s2),
        },
    )

    all_promoted = promoted_s1 + promoted_s2
    last_top10 = top10_s2 if top10_s2 else top10_s1

    # ── stage 2.5: NN in refined24 pool (if available, and not already primary)
    if refined24 and not refined24_as_primary:
        neighbors_r24, tol_r24 = neighbor_search(
            top_trials=[t for _, _, t in all_promoted],
            candidate_pool=refined24,
            start_tol=NEIGHBOR_TOL_START,
            step=NEIGHBOR_TOL_STEP,
            max_tol=NEIGHBOR_TOL_MAX,
            min_matches=NEIGHBOR_MIN_MATCHES,
        )
        ranked_r24 = rank_pool(neighbors_r24, train_cells, metric_key)
        top10_r24 = _top_k(ranked_r24, STAGE1_TOP_K)
        promoted_r24 = _promote(top10_r24)
        result["stages"]["stage2_refined24"] = _stage_record(
            top10_r24, promoted_r24,
            {
                "tolerance_used": tol_r24,
                "neighbors_found": len(neighbors_r24),
                "scored": len(ranked_r24),
            },
        )
        all_promoted = all_promoted + promoted_r24
        last_top10 = top10_r24 if top10_r24 else last_top10

    # ── deduplicate by object identity ──────────────────────────────────────
    seen_ids: set[int] = set()
    unique_promoted: list[tuple] = []
    for item in all_promoted:
        tid = id(item[2])
        if tid not in seen_ids:
            seen_ids.add(tid)
            unique_promoted.append(item)

    # ── pad to MIN_RETURN_K if needed ────────────────────────────────────────
    if len(unique_promoted) < MIN_RETURN_K:
        for item in last_top10:
            if id(item[2]) not in seen_ids:
                seen_ids.add(id(item[2]))
                unique_promoted.append(item)
            if len(unique_promoted) >= MIN_RETURN_K:
                break

    # ── stage 3: holdout evaluation ─────────────────────────────────────────
    # re-rank promoted candidates against each other on holdout cells
    holdout_ranked = rank_pool(
        [t for _, _, t in unique_promoted], holdout_cells, metric_key
    )

    holdout_scored: list[dict] = []
    for avg_rank, mean_mae, t in holdout_ranked:
        train_entry = next(
            (e for e in unique_promoted if id(e[2]) == id(t)), None
        )
        train_avg_rank = train_entry[0] if train_entry else float("nan")
        train_mae = train_entry[1] if train_entry else float("nan")
        holdout_scored.append({
            "trial_id": t["trial_id"],
            "variant": t.get("pilot_variant"),
            "train_avg_rank": round(train_avg_rank, 4),
            "train_mae": round(train_mae, 6),
            "holdout_avg_rank": round(avg_rank, 4),
            "holdout_mae": round(mean_mae, 6),
            "hyperparams": t["hyperparams"],
        })
    # primary sort: holdout avg_rank asc; tiebreak: holdout mae
    holdout_scored.sort(key=lambda x: (x["holdout_avg_rank"], x["holdout_mae"]))

    result["stages"]["stage3"] = {
        "candidates_in": len(unique_promoted),
        "scored": len(holdout_scored),
    }
    shortlist = holdout_scored[:FINAL_SHORTLIST_K]
    result["shortlist"] = shortlist

    # ── summary variance ─────────────────────────────────────────────────────
    if shortlist:
        maes = [e["holdout_mae"] for e in shortlist]
        ranks = [e["holdout_avg_rank"] for e in shortlist]
        mean_mae = sum(maes) / len(maes)
        result["variance"] = {
            "n": len(shortlist),
            "mean_holdout_mae": round(mean_mae, 6),
            "std_holdout_mae": round(
                (sum((x - mean_mae) ** 2 for x in maes) / len(maes)) ** 0.5, 6
            ),
            "min_holdout_mae": round(min(maes), 6),
            "min_holdout_avg_rank": round(min(ranks), 4),
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
    refined24_as_primary: bool = False,
) -> dict:
    cfg = EXP_CONFIGS[experiment]
    metric_key = cfg["metric_key"]
    n_train = cfg["n_train_cells"]
    cell_split_seed = cfg["cell_split_seed"]

    exp_dir = data_root / experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

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

    n_total_cfg = cfg["n_total_cells"]

    for mdir in method_dirs:
        method = mdir.name
        refined24_dir = exp_dir / f"{method}_refined24"
        has_r24 = refined24_dir.exists()

        # When using refined24 as primary, the trials may cover fewer cells
        # than the full experiment universe.  Re-derive train/holdout from
        # whatever cells those trials actually evaluated.
        method_train = train_cells
        method_holdout = holdout_cells
        if refined24_as_primary and has_r24:
            r24_sample = load_trials(refined24_dir, "broad")
            if r24_sample:
                r24_cells = [tuple(c) for c in r24_sample[0].get("training_cells", [])]
                if r24_cells and set(r24_cells) != set(all_cells):
                    n_r24 = len(r24_cells)
                    n_train_r24 = max(1, round(n_r24 * n_train / n_total_cfg))
                    method_train, method_holdout = split_cells(
                        r24_cells, n_train_r24, seed=cell_split_seed
                    )

        if method in EXCLUDED_METHODS:
            print(f"  {method:<35}  (excluded)")
            continue

        print(f"  {method:<35} {'[+r24]' if has_r24 else '      '}", end="", flush=True)
        try:
            res = analyze_method(
                method=method,
                method_dir=mdir,
                train_cells=method_train,
                holdout_cells=method_holdout,
                metric_key=metric_key,
                refined24_dir=refined24_dir if has_r24 else None,
                refined24_as_primary=refined24_as_primary,
            )
        except Exception as exc:
            res = {"method": method, "error": str(exc)}
            print(f" ERROR: {exc}")
        else:
            if "skip" in res:
                print(f" (skipped: {res['skip']})")
                continue
            n_short = len(res.get("shortlist", []))
            strat = res.get("strategy", "")[:40]
            print(f" shortlist={n_short}  [{strat}]")
        results["methods"][method] = res

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{experiment}_rank_noDRE.json"
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
    method_rows = []
    for method, res in results["methods"].items():
        if "error" in res:
            method_rows.append((float("inf"), method, res))
        else:
            rank = res.get("variance", {}).get("min_holdout_avg_rank", float("inf"))
            method_rows.append((rank, method, res))
    method_rows.sort(key=lambda x: x[0])

    for _, method, res in method_rows:
        if "error" in res:
            summary["methods"][method] = {"error": res["error"]}
            continue
        shortlist = res.get("shortlist", [])
        if not shortlist:
            continue
        shortlist_sorted = sorted(shortlist, key=lambda s: s["holdout_mae"])
        entry: dict = {
            "variance": res.get("variance", {}),
            "shortlist": [
                {
                    "rank": i + 1,
                    "holdout_mae": s["holdout_mae"],
                    "holdout_avg_rank": s["holdout_avg_rank"],
                    "train_mae": s["train_mae"],
                    "train_avg_rank": s["train_avg_rank"],
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
    print(f"SUMMARY (rank)  {exp}")
    print(f"{'='*70}")
    print(f"  {'Method':<35} {'MinAvgRank':>10} {'BestMAE':>10} {'N':>3}  Strategy")
    print(f"  {'-'*68}")
    rows = []
    for method, res in results["methods"].items():
        if "error" in res:
            rows.append((float("inf"), method, res))
        else:
            rank = res.get("variance", {}).get("min_holdout_avg_rank", float("nan"))
            rows.append((rank, method, res))
    rows.sort(key=lambda x: x[0])
    for _, method, res in rows:
        if "error" in res:
            continue
        if not res.get("shortlist"):
            continue
        var = res.get("variance", {})
        best = var.get("min_holdout_mae", float("nan"))
        min_rank = var.get("min_holdout_avg_rank", float("nan"))
        n = var.get("n", 0)
        s = res.get("strategy", "")
        strat = "r24" if "refined24_primary" in s else ("refined" if "refined_primary" in s else "broad")
        print(f"  {method:<35} {min_rank:>10.2f} {best:>10.5f} {n:>3}  {strat}")

    print(f"\n  {'Method':<35}  Dominant HPs per stage")
    print(f"  {'-'*68}")
    for _, method, res in rows:
        if "error" in res or not res.get("shortlist"):
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
        for r in results["methods"].values() if "error" not in r
    )
    total_diff = sum(
        len(r.get("anomalies", {}).get("diff_hp_similar_mae", []))
        for r in results["methods"].values() if "error" not in r
    )
    print(f"\n  Anomalies: similar-HP/different-MAE={total_sim}  "
          f"different-HP/similar-MAE={total_diff}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HPO analysis pipeline (per-cell rank variant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment", required=True, choices=list(EXP_CONFIGS.keys()))
    parser.add_argument(
        "--data-root", default="/data/user_data/yizhoulu/dpe-submission",
    )
    parser.add_argument(
        "--output-dir", default="experiments/analysis/results",
    )
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--refined24-as-primary",
        action="store_true",
        default=False,
        help="Use refined24 trials as stage-1 primary pool (only for exps with real refined24 data)",
    )
    args = parser.parse_args()

    run_experiment(
        experiment=args.experiment,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        methods=args.methods,
        refined24_as_primary=args.refined24_as_primary,
    )


if __name__ == "__main__":
    main()
