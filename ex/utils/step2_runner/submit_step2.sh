#!/bin/bash
# submit_step2.sh
# unified step2 entry point with two non-colliding modes.
#
# usage:
#   bash submit_step2.sh --mode {general|dual} --queue <queue_file> [options]
#
# modes:
#   general: single drain, partition=general, lite watchdog only.
#            for low-cap, no-preempt runs (qos=normal MaxSubmitJobsPerUser=50).
#   dual:    double drain, partition=preempt + cpu array partition.
#            lite watchdog (front-pop, preserves --requeue for preempt) +
#            cpu_dispatcher (back-pop, cpu array). for high-throughput runs.
#
# both modes use distinct logdirs ($DPE_DATA_ROOT/<exp>/step2_<mode>/<run_id>/)
# so they can run concurrently without colliding on state.
#
# options:
#   --my-cap         per-user cap (default: general=45, dual=22)
#   --total-cap      total cluster cap (default: 1500)
#   --array-size     cpu array element count (dual mode, default: 64)
#   --concurrency    cpu array concurrency (dual mode, default: 100)
#   --n-per-element  trials per array element (dual mode, default: 2)
#   --method-filter  cpu array method whitelist (dual mode, default: all eligible)
#   --orphan         orphan-scan interval (default: 60)

set -e

MODE=""
QUEUE=""
MY_CAP=""
TOTAL_CAP=1500
ARRAY_SIZE=64
ARRAY_CONCURRENCY=100
N_PER_ELEMENT=2
METHOD_FILTER=""
ORPHAN=60

usage() {
    sed -n '2,30p' "$0" | sed 's/^# //; s/^#//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)          MODE="$2"; shift 2 ;;
        --queue)         QUEUE="$2"; shift 2 ;;
        --my-cap)        MY_CAP="$2"; shift 2 ;;
        --total-cap)     TOTAL_CAP="$2"; shift 2 ;;
        --array-size)    ARRAY_SIZE="$2"; shift 2 ;;
        --concurrency)   ARRAY_CONCURRENCY="$2"; shift 2 ;;
        --n-per-element) N_PER_ELEMENT="$2"; shift 2 ;;
        --method-filter) METHOD_FILTER="$2"; shift 2 ;;
        --orphan)        ORPHAN="$2"; shift 2 ;;
        -h|--help)       usage ;;
        *) echo "unknown arg: $1"; usage ;;
    esac
done

if [[ "$MODE" != "general" && "$MODE" != "dual" ]]; then
    echo "ERROR: --mode must be 'general' or 'dual'"
    exit 2
fi
if [[ -z "$QUEUE" ]]; then
    echo "ERROR: --queue required"
    exit 2
fi
if [[ ! -f "$QUEUE" ]]; then
    echo "ERROR: queue file not found: $QUEUE"
    exit 2
fi

# default my-cap by mode
if [[ -z "$MY_CAP" ]]; then
    if [[ "$MODE" == "general" ]]; then MY_CAP=45; else MY_CAP=22; fi
fi

WORKDIR="${DPE_WORKDIR:-$PWD}"
export DPE_DATA_ROOT="${DPE_DATA_ROOT:-$HOME/dpe-data}"
export DPE_CKPT_ROOT="${DPE_CKPT_ROOT:-$HOME/dpe-ckpt}"

# derive experiment name from queue filename: step2_<exp>_queue.txt
QF_BASE="$(basename "$QUEUE")"
QF_STEM="${QF_BASE%.*}"
EXP="${QF_STEM#step2_}"
EXP="${EXP%_queue}"
RUN_ID="${WATCHDOG_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="$DPE_DATA_ROOT/$EXP/step2_${MODE}/$RUN_ID"
mkdir -p "$LOG_ROOT"

echo "=== step2 ${MODE}-drain ==="
echo "queue:           $QUEUE"
echo "experiment:      $EXP"
echo "run_id:          $RUN_ID"
echo "log_root:        $LOG_ROOT"
echo "watchdog caps:   my=$MY_CAP total=$TOTAL_CAP"
[[ "$MODE" == "dual" ]] && echo "cpu_array:       size=$ARRAY_SIZE concurrency=$ARRAY_CONCURRENCY n_per_element=$N_PER_ELEMENT filter=${METHOD_FILTER:-(none)}"

# 1. lite watchdog: partition switches by mode
WATCHDOG_PARTITION="general"
if [[ "$MODE" == "dual" ]]; then WATCHDOG_PARTITION="preempt"; fi
echo "watchdog partition: $WATCHDOG_PARTITION"

WATCHDOG_RUN_ID="${RUN_ID}_lite_${MODE}" \
    bash "$WORKDIR/experiments/utils/submit_watchdog_lite.sh" \
        "$QUEUE" "$MY_CAP" "$TOTAL_CAP" "$ORPHAN" \
        -- --partition="$WATCHDOG_PARTITION"

# 2. cpu_dispatcher (dual mode only)
if [[ "$MODE" == "dual" ]]; then
    echo
    echo "[dual] submitting cpu_dispatcher (back-pop -> array CPU)..."

    set +u
    source ~/.bashrc 2>/dev/null
    conda activate fac
    set -u
    cd "$WORKDIR"

    FILTER_ARG=""
    if [[ -n "$METHOD_FILTER" ]]; then
        FILTER_ARG="--method-filter $METHOD_FILTER"
    fi

    python -m experiments.utils.step2_runner.cpu_dispatcher \
        --queue-file "$QUEUE" \
        --array-size "$ARRAY_SIZE" \
        --concurrency "$ARRAY_CONCURRENCY" \
        --n-per-element "$N_PER_ELEMENT" \
        --output-root "$LOG_ROOT/cpu_array" \
        --device cpu \
        $FILTER_ARG
fi

echo
echo "=== step2 ${MODE}-drain submitted ==="
echo "watchdog log: $DPE_DATA_ROOT/$EXP/watchdog_lite/${RUN_ID}_lite_${MODE}/watchdog.log  (NB: legacy log path; logs land here even though logdir is $LOG_ROOT)"
[[ "$MODE" == "dual" ]] && echo "cpu_array logs:    $LOG_ROOT/cpu_array/elem_*.out"
echo
echo "monitor:           squeue -u \$USER"
