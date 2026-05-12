# HPO Analysis Pipeline

| File | Purpose |
|---|---|
| `config.py` | All tunable constants (cell splits, subsample counts, tolerances) |
| `utils.py` | Shared helpers (load trials, split cells, metric, neighbourhood search) |
| `analyze.py` | Main analysis pipeline (stages 1–3 + optional refined24 stage) |
| `analyze_rank.py` | Per-cell rank variant of `analyze.py` |

---

## Analysis pipeline (`analyze.py`)

### What it does

**Per-experiment (global):**
- Reads `training_cells` from any trial to get the full eval-cell list.
- Splits cells deterministically (seed 42) into:
  - **train cells** (16 for smodice / 24 for elbo) — used for stage-1/2/2.5 selection
  - **holdout cells** (8 for both) — withheld until stage 3

**Per-method:**

| Stage | Condition | Input pool | Action | Promotes |
|---|---|---|---|---|
| **1** | always | primary pool (see below) | score on train cells → top-10 by mean MAE | 2 with lowest cross-cell std |
| **2** | always | secondary broad pool | NN search anchored on stage-1 top-10 → top-10 by mean MAE | 2 with lowest cross-cell std |
| **2.5** | `<method>_refined24/` exists | refined24 broad trials | NN search anchored on all promoted-so-far → top-10 by mean MAE | 2 with lowest cross-cell std |
| **3** | always | all promoted candidates | holdout evaluation | ranked by holdout cross-cell std |

**Cross-cell std** (the variance metric): for a given trial, evaluate on all train/holdout cells, compute the population std of per-cell MAE values. A low cross-cell std means the method is consistently good across different cells, not just lucky on a subset.

**Selection criterion at each stage:** from the top-10 by mean MAE, the 2 candidates with the *lowest* cross-cell std are promoted to the next stage. This keeps only the most consistent configs moving forward.

**Primary/secondary pool selection:**

| Condition | Primary pool | Secondary pool |
|---|---|---|
| Method has ≥10 refined trials | ALL refined trials (no subsampling) | ALL broad trials |
| Otherwise | Subsample N from broad (see below) | Remaining broad trials |

**Broad-trial subsample counts (N):**

| Method family | N |
|---|---|
| `*VFM*` | 50 |
| `*FMDRE*` | 30 |
| everything else | 100 |

**At-least-5 guarantee:** promoted candidates = 2 (S1) + 2 (S2) + 0–2 (S2.5) = 4–6 total. If fewer than 5 survive to holdout, the pipeline pads with the next-best candidates from the last completed stage's top-10 (sorted by train cross-cell std, excluding already-promoted ones) until 5 are available.

**Final shortlist ordering:** holdout cross-cell std ascending (tiebreak: holdout mean MAE). The best entry is the most variance-consistent on unseen cells.

**Dominant HP reporting:** at each stage (1, 2, 2.5), any HP whose single value appears in >50% of that stage's top-10 trials is reported. This surfaces HPs that are effectively pinned by the data regardless of search-space width.

**Neighbourhood search:** starts at 10% relative tolerance; relaxes by 10% per step until ≥3 neighbours found (max tolerance configurable in `config.py`). Categorical values must match exactly.

### How to run

Set `DPE_DATA_ROOT` before running (see the root `README.md` for environment-variable setup):

```bash
export DPE_DATA_ROOT=/path/to/dpe/data
source venv/bin/activate

# smodice (automatically picks up BDRE_refined24/, FMDRE_refined24/, etc.)
python -m ex.ablations.analysis.analyze \
    --experiment smodice_eldr_estimation \
    --output-dir ex/ablations/analysis/results

# elbo
python -m ex.ablations.analysis.analyze \
    --experiment elbo_estimation \
    --output-dir ex/ablations/analysis/results

# restrict to specific methods
python -m ex.ablations.analysis.analyze \
    --experiment smodice_eldr_estimation \
    --methods BDRE FMDRE VFM \
    --output-dir ex/ablations/analysis/results
```

`--data-root` defaults to `$DPE_DATA_ROOT`; pass it explicitly to point at a different location.

### Output files

`ex/ablations/analysis/results/<experiment>_analysis.json` (full) and
`ex/ablations/analysis/results/<experiment>_summary.json` (concise).

**Summary schema:**
```json
{
  "experiment": "smodice_eldr_estimation",
  "metric": "per_cell_ldr_mae",
  "methods": {
    "BDRE": {
      "n_refined24": 24,
      "variance": {
        "n": 5,
        "mean_holdout_mae": 0.41,
        "std_holdout_mae": 0.03,
        "min_holdout_mae": 0.37,
        "min_holdout_cell_std": 0.02
      },
      "shortlist": [
        {
          "rank": 1,
          "holdout_mae": 0.37,
          "holdout_cell_std": 0.02,
          "train_mae": 0.35,
          "train_cell_std": 0.018,
          "trial_id": 42,
          "variant": "broad",
          "hyperparams": {"learning_rate": 0.001, ...}
        }
      ],
      "dominant_categoricals_per_stage": {
        "stage1": {"n_hidden_layers": {"value": 3, "fraction": 0.8, "count": 8, "total": 10}},
        "stage2": {},
        "stage2_refined24": {}
      }
    }
  }
}
```

**Full analysis schema** (`_analysis.json`) additionally contains per-stage details:
```json
"stages": {
  "stage1": {
    "pool_size": 100, "scored": 98, "top_k": 10, "promoted": 2,
    "dominant_categoricals": {"n_hidden_layers": {"value": 3, "fraction": 0.8, ...}},
    "top_trials": [{"trial_id": 42, "train_mae": 0.35, "train_cell_std": 0.018,
                    "promoted": true, "hyperparams": {...}}, ...]
  },
  "stage2": {"tolerance_used": 0.1, "neighbors_found": 12, ...},
  "stage2_refined24": {"tolerance_used": 0.2, "neighbors_found": 5, ...},
  "stage3": {"candidates_in": 6, "scored": 6}
}
```

---

## Adding a new experiment

1. Add an entry in `config.py -> EXP_CONFIGS`:
   ```python
   "my_new_exp": {
       "metric_key": "per_cell_something",
       "n_total_cells": 24,
       "n_train_cells": 16,
       "n_holdout_cells": 8,
       "cell_split_seed": 42,
   },
   ```
2. Run `analyze.py --experiment my_new_exp` — no other changes needed.
3. If `<method>_refined24/` directories exist alongside the regular method dirs, they are picked up automatically.
