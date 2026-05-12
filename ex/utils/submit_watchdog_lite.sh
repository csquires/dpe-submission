#!/bin/bash
# submit_watchdog_lite.sh
# submit lite watchdog (general partition, no cpu_array drain) to cpu partition.
# usage: bash submit_watchdog_lite.sh [--reset] <WATCHDOG_QUEUE_FILE> [MY_CAP] [TOTAL_CAP] [ORPHAN_INTERVAL] [-- <extra watchdog flags...>]
# defaults: MY_CAP=80, TOTAL_CAP=200, ORPHAN_INTERVAL=60

set -e
set -u

RESET_FLAG=""
if [[ "${1:-}" == "--reset" ]]; then
    RESET_FLAG="--reset-exclude-on-startup"
    shift
fi
WATCHDOG_QUEUE_FILE="${1:?"usage: $0 [--reset] <WATCHDOG_QUEUE_FILE> [MY_CAP] [TOTAL_CAP] [ORPHAN_INTERVAL] [-- <extra watchdog flags...>]"}"
MY_CAP="${2:-80}"
TOTAL_CAP="${3:-200}"
ORPHAN_INTERVAL="${4:-60}"
shift 4 2>/dev/null || true
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS="$*"

WORKDIR="${DPE_WORKDIR:-/home/aviamala/dpe-submission}"
CONDA_ENV="${DPE_CONDA_ENV:-fac}"
export DPE_DATA_ROOT="${DPE_DATA_ROOT:-/data/user_data/$USER/dpe-submission}"
export DPE_CKPT_ROOT="${DPE_CKPT_ROOT:-/scratch/$USER/ckpt/dpe-submission}"

RUN_ID="${WATCHDOG_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

QF_BASE="$(basename "$WATCHDOG_QUEUE_FILE")"
QF_STEM="${QF_BASE%.*}"
QF_PARENT="$(basename "$(dirname "$WATCHDOG_QUEUE_FILE")")"
if [[ "$QF_STEM" == *_watchdog* ]]; then
    EXP="${QF_STEM%%_watchdog*}"
elif [[ "$QF_PARENT" == "watchdog" || "$QF_PARENT" == "queues" ]]; then
    EXP="$(basename "$(dirname "$(dirname "$WATCHDOG_QUEUE_FILE")")")"
else
    EXP="$QF_PARENT"
fi
LOGDIR="$DPE_DATA_ROOT/$EXP/watchdog_lite/$RUN_ID"
mkdir -p "$LOGDIR"

EXCLUDE_FILE="$LOGDIR/exclude.txt"
LOG_FILE="$LOGDIR/watchdog.log"
STATE_FILE="$LOGDIR/state.json"
touch "$EXCLUDE_FILE"
touch "$LOG_FILE"

JID=$(sbatch --parsable \
    --partition=cpu \
    --time=24:00:00 \
    --mem=4G \
    --cpus-per-task=2 \
    --requeue \
    --job-name=watchdog_lite \
    --output="$LOGDIR/watchdog.out" \
    --error="$LOGDIR/watchdog.err" \
    --wrap="export DPE_WAVE3='${DPE_WAVE3:-}' DPE_DATA_ROOT='$DPE_DATA_ROOT' DPE_CKPT_ROOT='$DPE_CKPT_ROOT' && \
            source ~/.bashrc && conda activate $CONDA_ENV && cd $WORKDIR && \
            python -m ex.utils.watchdog_lite \
              --queue-file '$WATCHDOG_QUEUE_FILE' \
              --my-cap $MY_CAP \
              --total-cap $TOTAL_CAP \
              --exclude-file '$EXCLUDE_FILE' \
              --log-file '$LOG_FILE' \
              --state-file '$STATE_FILE' \
              --orphan-scan-interval $ORPHAN_INTERVAL \
              $RESET_FLAG \
              $EXTRA_ARGS")

echo "=== watchdog_lite submitted ==="
echo "jobid:          $JID"
echo "run id:         $RUN_ID"
echo "queue file:     $WATCHDOG_QUEUE_FILE"
echo "logdir:         $LOGDIR"
echo ""
echo "monitor:        squeue -j $JID"
echo "tail log:       tail -f $LOGDIR/watchdog.log"
echo "cancel:         scancel $JID"
