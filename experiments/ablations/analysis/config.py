"""Per-experiment and per-method configuration for the analysis pipeline."""
from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Experiment-level config
# ---------------------------------------------------------------------------
# metric_key      : field name inside trial JSON that maps cell_key → score
# n_total_cells   : number of eval cells used per trial (all same within an exp)
# n_train_cells   : cells used for stage-1/2 selection (total - n_holdout_cells)
# n_holdout_cells : cells withheld for stage-3 evaluation
# cell_split_seed : RNG seed for deterministic train/holdout split

EXP_CONFIGS: dict[str, dict] = {
    "smodice_eldr_estimation_combined": {
        "metric_key": "per_cell_ldr_mae",
        "n_total_cells": 24,
        "n_train_cells": 16,
        "n_holdout_cells": 8,
        "cell_split_seed": 42,
    },
    "elbo_estimation_combined": {
        "metric_key": "per_cell_eldr_abs_err",
        "n_total_cells": 32,
        "n_train_cells": 24,
        "n_holdout_cells": 8,
        "cell_split_seed": 42,
    },
    "dbpedia_cond_flow_combined": {
        "metric_key": "per_pair_mae",
        "n_total_cells": 32,
        "n_train_cells": 24,
        "n_holdout_cells": 8,
        "cell_split_seed": 42,
    },
    "mnist_cond_flow_combined": {
        "metric_key": "per_pair_mae",
        "n_total_cells": 32,
        "n_train_cells": 24,
        "n_holdout_cells": 8,
        "cell_split_seed": 42,
    },
    "eig_estimation_combined": {
        "metric_key": "per_design_eig_abs_err",
        "n_total_cells": 32,
        "n_train_cells": 24,
        "n_holdout_cells": 8,
        "cell_split_seed": 42,
    },
    "pendulum_eldr_estimation_combined": {
        "metric_key": "per_cell_ldr_mae",
        "n_total_cells": 32,
        "n_train_cells": 24,
        "n_holdout_cells": 8,
        "cell_split_seed": 42,
    },
    "dre_sample_complexity_combined": {
        "metric_key": "per_cell_mae",
        "n_total_cells": 9,
        "n_train_cells": 6,
        "n_holdout_cells": 3,
        "cell_split_seed": 42,
    },
    "plugin_dre_combined": {
        "metric_key": "per_cell_mae",
        "n_total_cells": 4,
        "n_train_cells": 3,
        "n_holdout_cells": 1,
        "cell_split_seed": 42,
    },
    "pstar_sample_complexity_combined": {
        "metric_key": "per_cell_mae",
        "n_total_cells": 20,
        "n_train_cells": 15,
        "n_holdout_cells": 5,
        "cell_split_seed": 42,
    },
    "model_selection_combined": {
        "metric_key": "per_row_ldr_mean_ae",
        "n_total_cells": 7,
        "n_train_cells": 5,
        "n_holdout_cells": 2,
        "cell_split_seed": 42,
    },
    "mnist_eldr_estimation_combined": {
        "metric_key": "per_pair_mae",
        "n_total_cells": 4,
        "n_train_cells": 3,
        "n_holdout_cells": 1,
        "cell_split_seed": 42,
    },
}

# ---------------------------------------------------------------------------
# Per-method broad-trial subsampling counts
# ---------------------------------------------------------------------------
# VFM / TriangularVFM_*   → budget 100 broad  → subsample 50
# FMDRE / TriangularFMDRE / FMDRE_S2 → budget 50 broad → subsample 30
# everything else          → budget 200 broad → subsample 100

_VFM_RE = re.compile(r"vfm", re.IGNORECASE)
_FMDRE_RE = re.compile(r"fmdre", re.IGNORECASE)

SUBSAMPLE_DEFAULT = 100
SUBSAMPLE_VFM = 50
SUBSAMPLE_FMDRE = 30


def get_subsample_n(method: str) -> int:
    """Broad-trial subsample count for *method*."""
    if _VFM_RE.search(method):
        return SUBSAMPLE_VFM
    if _FMDRE_RE.search(method):
        return SUBSAMPLE_FMDRE
    return SUBSAMPLE_DEFAULT


# ---------------------------------------------------------------------------
# Refined-trial threshold
# ---------------------------------------------------------------------------
# If a method has >= this many refined trials, use ALL refined as the primary
# pool and skip broad subsampling for stage-1.
REFINED_MIN_TRIALS = 10

# ---------------------------------------------------------------------------
# Stage parameters
# ---------------------------------------------------------------------------
STAGE1_TOP_K = 10       # top-K selected from primary pool (stage 1)
STAGE2_TOP_K = 10       # top-K selected from NN search (stage 2)
STAGE_PROMOTE_K = 2     # candidates promoted from each stage (by lowest cell-std)
FINAL_SHORTLIST_K = 10  # max entries in the final shortlist (stage 3)
MIN_RETURN_K = 5        # minimum hyperparams returned per method

# Fraction threshold for categorical HP dominance reporting
CATEGORICAL_DOMINANCE_THRESH = 0.50

# Broad-trial subsample seed (for reproducibility)
SUBSAMPLE_SEED = 0

# ---------------------------------------------------------------------------
# Classifier methods excluded from analysis
# ---------------------------------------------------------------------------
# These methods use a classifier/discriminator and ran with far more epochs
# than the submission budget allows; exclude them entirely.
EXCLUDED_METHODS: frozenset[str] = frozenset({
    "BDRE",
    "MDRE_15",
    "MultiHeadTDRE",
    "MultiHeadTriangularTDRE",
})

# ---------------------------------------------------------------------------
# Neighbourhood search
# ---------------------------------------------------------------------------
NEIGHBOR_TOL_START = 0.10   # initial relative tolerance (10 %)
NEIGHBOR_TOL_STEP = 0.10    # relax by this amount per iteration
NEIGHBOR_TOL_MAX = 0.30     # give up after this tolerance
NEIGHBOR_MIN_MATCHES = 3    # minimum neighbours before we stop relaxing

# ---------------------------------------------------------------------------
# Anomaly-detection thresholds
# ---------------------------------------------------------------------------
# "Similar HP → different MAE": hp_dist ≤ SIMILAR_HP and rel_mae_diff > DIFF_MAE
ANOMALY_SIMILAR_HP_THRESH = 0.10
ANOMALY_DIFF_MAE_THRESH = 0.20

# "Different HP → similar MAE": hp_dist > DIFF_HP and rel_mae_diff ≤ SIMILAR_MAE
ANOMALY_DIFF_HP_THRESH = 0.30
ANOMALY_SIMILAR_MAE_THRESH = 0.05
