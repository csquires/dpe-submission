#!/bin/bash
# submit_watchdog.sh
# submit watchdog variant to cpu partition. one watchdog per pilot variant.
# usage: bash submit_watchdog.sh [--reset] <WATCHDOG_QUEUE_FILE> [MY_CAP] [TOTAL_CAP] [ORPHAN_INTERVAL]
# MY_CAP defaults 80, TOTAL_CAP defaults 200, ORPHAN_INTERVAL defaults 60
# (cycles between orphan-recovery scans; pass 0 to disable).
# --reset: truncate exclude.txt on startup (fresh launch only; NOT for requeue path).

set -e
set -u

# parse positional + optional args
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
# remaining args (after optional `--`) are forwarded to watchdog.py verbatim.
# allows callers to pass --cpu-array-relaunch and friends.
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS="$*"

# environment setup
WORKDIR="${DPE_WORKDIR:-$PWD}"
CONDA_ENV="${DPE_CONDA_ENV:-fac}"
export DPE_DATA_ROOT="${DPE_DATA_ROOT:-/data/user_data/$USER/dpe-submission}"
export DPE_CKPT_ROOT="${DPE_CKPT_ROOT:-/scratch/$USER/ckpt/dpe-submission}"

# directory + artifact creation
# WATCHDOG_RUN_ID env override allows relaunching into an existing LOGDIR
# (preserves state.json + exclude.txt across restart).
RUN_ID="${WATCHDOG_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

# derive experiment name for readable logdir.
# try filename first (top-level "<exp>_watchdog_queue.txt" pattern), then
# parent-dir layout (.../<exp>/watchdog/queue.jsonl), else fall back to
# the immediate parent dir name.
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
# state + log on NFS (DPE_DATA_ROOT) so requeue across nodes survives.
# /scratch (DPE_CKPT_ROOT) is node-local; cross-node requeue would lose state.
LOGDIR="$DPE_DATA_ROOT/$EXP/watchdog/$RUN_ID"
mkdir -p "$LOGDIR"

# ensure artifact files exist
EXCLUDE_FILE="$LOGDIR/exclude.txt"
LOG_FILE="$LOGDIR/watchdog.log"
STATE_FILE="$LOGDIR/state.json"
SUBMITTED_TSV="$LOGDIR/submitted.tsv"
touch "$EXCLUDE_FILE"
touch "$LOG_FILE"

# sbatch invocation
JID=$(sbatch --parsable \
    --partition=cpu \
    --time=24:00:00 \
    --mem=4G \
    --cpus-per-task=2 \
    --requeue \
    --job-name=v735e_watchdog \
    --output="$LOGDIR/watchdog.out" \
    --error="$LOGDIR/watchdog.err" \
    --wrap="source ~/.bashrc && conda activate $CONDA_ENV && cd $WORKDIR && \
            python -m ex.utils.watchdog \
              --queue-file '$WATCHDOG_QUEUE_FILE' \
              --my-cap $MY_CAP \
              --total-cap $TOTAL_CAP \
              --exclude-file '$EXCLUDE_FILE' \
              --log-file '$LOG_FILE' \
              --state-file '$STATE_FILE' \
              --orphan-scan-interval $ORPHAN_INTERVAL \
              $RESET_FLAG \
              $EXTRA_ARGS")

# output and user feedback
echo "=== watchdog submitted ==="
echo "jobid:          $JID"
echo "run id:         $RUN_ID"
echo "queue file:     $WATCHDOG_QUEUE_FILE"
echo "logdir:         $LOGDIR"
echo ""
echo "monitor:        squeue -j $JID"
echo "tail log:       tail -f $LOGDIR/watchdog.log"
echo "cancel:         scancel $JID"
