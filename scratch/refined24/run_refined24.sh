#!/usr/bin/env bash
# refined24 mini-campaign launcher.
#
# preflight checklist (do not run blindly):
#   - DPE_DATA_ROOT and DPE_CKPT_ROOT are set in your environment.
#   - conda env `fac` is on PATH (the launcher activates it inside sbatch wraps).
#   - the prior recalibrated_specs/{model_selection,smodice_eldr_estimation}.yaml
#     are saved in case you need to revert (this script does that automatically).
#
# what it does, in order:
#   1. (re)builds narrowed yamls from jsons/<exp>_combined_summary.json with top-5 trials.
#      output goes to recalibrated_specs/<exp>_refined24.yaml (does not touch
#      the active yaml yet).
#   2. backs up the active recalibrated_specs/<exp>.yaml -> <exp>.yaml.before_refined24.bak
#      and swaps in the refined24 version.
#   3. exports DPE_BUDGET_PROFILE=refined24 and DPE_TRAINING_M=DPE_HOLDOUT_M=16.
#   4. invokes launcher_lite with --budget 25 --output-suffix _refined24
#      --experiments model_selection,smodice_eldr_estimation. defaults to dry-run
#      so you can see the matrix before committing. set REAL=1 to actually submit.
#   5. always restores the backed-up yaml before exiting (even on error/abort)
#      so the repo is left clean.
#
# usage:
#   bash scratch/refined24/run_refined24.sh             # dry-run; prints matrix
#   REAL=1 bash scratch/refined24/run_refined24.sh      # actually submits
#
set -euo pipefail

WORKDIR="/home/aviamala/dpe-submission"
EXPS="${EXPS:-model_selection,smodice_eldr_estimation,elbo_estimation,mnist_cond_flow}"
SPEC_DIR="${WORKDIR}/experiments/utils/hpo/recalibrated_specs"
PYTHON="/home/aviamala/miniconda3/envs/fac/bin/python"
SKIP_PAIRS_FILE="${WORKDIR}/scratch/refined24/skip_pairs.json"

# 0. detect clear winners & emit skip-pairs.json + winners.<exp>.refined24.yaml
# tier-2 cutoff (gap>=2%, spread<=12%); override via GAP_THR / SPREAD_THR env.
"$PYTHON" -m scratch.refined24.clear_winners \
    --gap-thr "${GAP_THR:-0.02}" \
    --spread-thr "${SPREAD_THR:-0.12}"

# 1. (re)build narrowed yamls (output: recalibrated_specs/<exp>_refined24.yaml)
"$PYTHON" -m scratch.refined24.build_specs \
    --experiments "$EXPS" \
    --suffix _refined24

# 2. swap in refined24 yaml; ensure revert on exit
declare -a SWAPS=()
restore_swaps() {
    for swap in "${SWAPS[@]}"; do
        bak="${swap}.before_refined24.bak"
        if [[ -f "$bak" ]]; then
            mv -f "$bak" "$swap"
            echo "[restore] $swap"
        fi
    done
}
trap restore_swaps EXIT

for exp in $(echo "$EXPS" | tr , ' '); do
    active="${SPEC_DIR}/${exp}.yaml"
    refined="${SPEC_DIR}/${exp}_refined24.yaml"
    if [[ ! -f "$refined" ]]; then
        echo "ERROR: missing $refined -- build_specs step failed." >&2
        exit 2
    fi
    if [[ -f "$active" ]]; then
        cp "$active" "${active}.before_refined24.bak"
        SWAPS+=("$active")
    fi
    cp "$refined" "$active"
    echo "[swap] $active <- $refined"
done

# 3. environment overrides
export DPE_BUDGET_PROFILE="refined24"
export DPE_TRAINING_M="16"
export DPE_HOLDOUT_M="16"
export DPE_WINNERS_SUFFIX=".refined24"   # persist writes winners.<exp>.refined24.yaml

# 4. invoke launcher (dry-run by default)
DRY_FLAG="--dry-run"
if [[ "${REAL:-0}" == "1" ]]; then
    DRY_FLAG=""
fi

cd "$WORKDIR"
# preempt + cpu_array (omnibus) launcher. faster fanout when slurm finally
# queues; trials may preempt and watchdog requeues. set LITE=1 to fall back
# to the general-partition lite path.
RUN_ID="refined24_$(date +%Y%m%d_%H%M%S)"
DATA_ROOT_FOR_Q="${DPE_DATA_ROOT:-${WORKDIR}/scratch/refined24/_nfs_test}"
QUEUE_FILE="${DATA_ROOT_FOR_Q}/${RUN_ID}_watchdog_queue.txt"
mkdir -p "$(dirname "$QUEUE_FILE")"

if [[ "${LITE:-0}" == "1" ]]; then
    "$PYTHON" -m experiments.utils.hpo.launcher_lite \
        --experiments "$EXPS" \
        --methods all \
        --budget 25 \
        --output-suffix "_refined24" \
        --skip-pairs-file "$SKIP_PAIRS_FILE" \
        --queue-file "$QUEUE_FILE" \
        --run-id "$RUN_ID" \
        $DRY_FLAG
else
    "$PYTHON" -m experiments.utils.hpo.launcher \
        --experiments "$EXPS" \
        --methods all \
        --budget 25 \
        --output-suffix "_refined24" \
        --skip-pairs-file "$SKIP_PAIRS_FILE" \
        --queue-file "$QUEUE_FILE" \
        --my-cap "${MY_CAP:-96}" \
        --total-cap "${TOTAL_CAP:-200}" \
        --cpu-concurrency "${CPU_CONCURRENCY:-64}" \
        $DRY_FLAG
fi

# 6. stage pinned winners onto NFS so downstream consumers find them
DATA_ROOT="${DPE_DATA_ROOT:-${WORKDIR}/scratch/refined24/_nfs_test}"
for exp in $(echo "$EXPS" | tr , ' '); do
    src="${WORKDIR}/scratch/refined24/winners_pinned/winners.${exp}.refined24.yaml"
    if [[ -f "$src" ]]; then
        dst_dir="${DATA_ROOT}/${exp}"
        mkdir -p "$dst_dir"
        cp "$src" "${dst_dir}/winners.${exp}.refined24.yaml"
        echo "[stage] $src -> $dst_dir/"
    fi
done

# 5. trap above restores the original yamls.
echo "done. (DPE_BUDGET_PROFILE=$DPE_BUDGET_PROFILE, DPE_TRAINING_M=$DPE_TRAINING_M)"
