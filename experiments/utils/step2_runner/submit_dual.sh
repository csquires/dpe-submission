#!/bin/bash
# submit_dual.sh
# submit BOTH a watchdog_lite (front-pop -> preempt GPU partition) AND a cpu
# array dispatcher (back-pop -> array CPU partition) against the same step2
# queue file. cpu-eligible methods are at the BACK of the queue (per
# adapter.is_cpu_eligible); slow GPU methods at the FRONT. atomic flock on
# the shared queue ensures no overlap.
#
# usage:
#   bash submit_dual.sh <QUEUE_FILE> [WATCHDOG_MY_CAP] [WATCHDOG_TOTAL_CAP] \\
#                       [CPU_ARRAY_SIZE] [CPU_CONCURRENCY] [N_PER_ELEMENT] \\
#                       [METHOD_FILTER]
#
# defaults:
#   WATCHDOG_MY_CAP=22  WATCHDOG_TOTAL_CAP=800
#   CPU_ARRAY_SIZE=64   CPU_CONCURRENCY=100  N_PER_ELEMENT=2
#   METHOD_FILTER="" (all cpu-eligible methods drain via array)

set -e
set -u

QUEUE_FILE="${1:?usage: $0 <queue_file> [watchdog_my_cap] [watchdog_total_cap] [array_size] [array_concurrency] [n_per_element] [method_filter]}"
WATCHDOG_MY_CAP="${2:-22}"
WATCHDOG_TOTAL_CAP="${3:-800}"
CPU_ARRAY_SIZE="${4:-64}"
CPU_CONCURRENCY="${5:-100}"
N_PER_ELEMENT="${6:-2}"
METHOD_FILTER="${7:-}"

WORKDIR="/home/aviamala/dpe-submission"
export DPE_DATA_ROOT="${DPE_DATA_ROOT:-/data/user_data/$USER/dpe-submission}"
export DPE_CKPT_ROOT="${DPE_CKPT_ROOT:-/scratch/$USER/ckpt/dpe-submission}"

# derive log root from queue filename: $DPE_DATA_ROOT/<exp>/step2_dual/<run_id>/
QF_BASE="$(basename "$QUEUE_FILE")"
QF_STEM="${QF_BASE%.*}"
# stem looks like step2_<exp>_queue
EXP="${QF_STEM#step2_}"
EXP="${EXP%_queue}"
RUN_ID="${WATCHDOG_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="$DPE_DATA_ROOT/$EXP/step2_dual/$RUN_ID"
mkdir -p "$LOG_ROOT"

echo "=== step2 dual-drain ==="
echo "queue:           $QUEUE_FILE"
echo "experiment:      $EXP"
echo "run_id:          $RUN_ID"
echo "log_root:        $LOG_ROOT"
echo "watchdog_caps:   my=$WATCHDOG_MY_CAP total=$WATCHDOG_TOTAL_CAP"
echo "cpu_array:       size=$CPU_ARRAY_SIZE concurrency=$CPU_CONCURRENCY n_per_element=$N_PER_ELEMENT"
echo "method_filter:   ${METHOD_FILTER:-(none)}"

# 1. front-pop: lite watchdog drains GPU lines into preempt partition.
echo
echo "[1/2] submitting watchdog_lite (front-pop -> preempt GPU)..."
WATCHDOG_RUN_ID="${RUN_ID}_lite" \
    bash "$WORKDIR/experiments/utils/submit_watchdog_lite.sh" \
        "$QUEUE_FILE" "$WATCHDOG_MY_CAP" "$WATCHDOG_TOTAL_CAP" 60

# 2. back-pop: cpu array dispatcher drains cpu-eligible lines into array partition.
echo
echo "[2/2] submitting cpu_dispatcher (back-pop -> array CPU)..."
FILTER_ARG=""
if [[ -n "$METHOD_FILTER" ]]; then
    FILTER_ARG="--method-filter $METHOD_FILTER"
fi

source ~/.bashrc && conda activate fac
cd "$WORKDIR"
python -m experiments.utils.step2_runner.cpu_dispatcher \
    --queue-file "$QUEUE_FILE" \
    --array-size "$CPU_ARRAY_SIZE" \
    --concurrency "$CPU_CONCURRENCY" \
    --n-per-element "$N_PER_ELEMENT" \
    --output-root "$LOG_ROOT/cpu_array" \
    --device cpu \
    $FILTER_ARG

echo
echo "=== dual-drain submitted ==="
echo "watchdog_lite log: $DPE_DATA_ROOT/$EXP/watchdog_lite/${RUN_ID}_lite/watchdog.log"
echo "cpu_array logs:    $LOG_ROOT/cpu_array/elem_*.out"
echo
echo "monitor:           squeue -u $USER --name=watchdog_lite,step2_cpuarr"
