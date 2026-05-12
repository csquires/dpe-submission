#!/bin/bash
# gen_tramp.sh
# generate per-method trampoline scripts that funnel HPO sweeps from the array
# partition to preempt. each trampoline submits its method's missing trials to
# preempt, chains the next trampoline via --dependency=afterany on the preempt
# jobid, and scancels the predecessor's array equivalent on entry. final tramp
# scancels the last method's array job.
#
# discovery: squeue --me -p array, names matching ^hpo(cf)?_<METHOD>$, deduped
# by parent jobid, sorted DESC by jobid (highest jobid first).
#
# usage:
#   bash gen_tramp.sh         # dry-run: print discovered order, do not write/submit
#   bash gen_tramp.sh GO      # write trampoline scripts and submit tramp_00
set -e
set -u

ACK="${1:-}"

WORKDIR="/home/aviamala/dpe-submission"
# centralized roots: never let heavy artifacts leak under $HOME.
export DPE_DATA_ROOT="${DPE_DATA_ROOT:-/data/user_data/$USER/dpe-submission}"
export DPE_CKPT_ROOT="${DPE_CKPT_ROOT:-/scratch/$USER/ckpt/dpe-submission}"
RUN_ID=$(date +%Y%m%d_%H%M%S)
TRAMP_ROOT="$DPE_DATA_ROOT/trampolines"
TRAMP_DIR="$TRAMP_ROOT/$RUN_ID"

# halved per-method walltime caps (originals in submit_hpo.sh; halved here for preempt).
declare -A TIME_LIMITS=(
    [TSM]="1:00:00"
    [CTSM]="1:00:00"
    [VFM]="1:00:00"
    [FMDRE]="1:30:00"
    [FMDRE_S2]="1:30:00"
    [MHTTDRE]="1:00:00"
    [TriangularCTSM_V1]="1:30:00"
    [TriangularCTSM_V2]="1:30:00"
    [TriangularCTSM_V3]="3:00:00"
    [TriangularVFM_V1]="2:00:00"
    [TriangularVFM_V2]="2:00:00"
    [TriangularVFM_V3]="3:00:00"
)
TIME_DEFAULT="1:30:00"

echo "=== discovering hpo array jobs ==="
SQUEUE_OUT=$(squeue --me -p array -t PD,R --noheader -o "%i %j" \
             | grep -E '^[0-9].* hpo(cf)?_[A-Za-z0-9_]+' || true)

if [ -z "$SQUEUE_OUT" ]; then
    echo "no hpo array jobs found in queue (PD or R). nothing to do."
    exit 0
fi

declare -A METHOD_OF
declare -A EXP_OF
PARENT_JIDS=()

while IFS= read -r line; do
    raw_jid=$(echo "$line" | awk '{print $1}')
    name=$(echo "$line"   | awk '{print $2}')
    parent_jid="${raw_jid%%_*}"
    if [ -n "${METHOD_OF[$parent_jid]:-}" ]; then continue; fi
    if [[ "$name" == hpocf_* ]]; then
        method="${name#hpocf_}"; exp="mnist_eldr_cond_flow"
    elif [[ "$name" == hpo_* ]]; then
        method="${name#hpo_}";   exp="mnist_eldr_estimation"
    else
        continue
    fi
    METHOD_OF[$parent_jid]="$method"
    EXP_OF[$parent_jid]="$exp"
    PARENT_JIDS+=("$parent_jid")
done <<< "$SQUEUE_OUT"

# sort parent jids descending
mapfile -t SORTED < <(printf '%s\n' "${PARENT_JIDS[@]}" | sort -rn)
N=${#SORTED[@]}

echo "discovered $N method-arrays (descending jobid):"
for i in "${!SORTED[@]}"; do
    jid="${SORTED[$i]}"
    method="${METHOD_OF[$jid]}"
    exp="${EXP_OF[$jid]}"
    tl="${TIME_LIMITS[$method]:-$TIME_DEFAULT}"
    printf "  [%02d] jid=%s exp=%-25s method=%-25s time=%s\n" \
        "$i" "$jid" "$exp" "$method" "$tl"
done

if [ "$ACK" != "GO" ]; then
    echo
    echo "dry-run only. to write and submit, run:  bash $0 GO"
    exit 0
fi

mkdir -p "$TRAMP_DIR/logs"
echo
echo "=== writing trampolines to $TRAMP_DIR ==="

for i in $(seq 0 $((N - 1))); do
    jid="${SORTED[$i]}"
    method="${METHOD_OF[$jid]}"
    exp="${EXP_OF[$jid]}"
    tl="${TIME_LIMITS[$method]:-$TIME_DEFAULT}"

    if [ "$i" -gt 0 ]; then
        prev_jid="${SORTED[$((i - 1))]}"
    else
        prev_jid=""
    fi

    printf -v idx_pad "%02d" "$i"
    out_path="$TRAMP_DIR/tramp_${idx_pad}_${exp}_${method}.sh"

    if [ "$i" -lt $((N - 1)) ]; then
        next_idx=$((i + 1))
        next_jid="${SORTED[$next_idx]}"
        next_method="${METHOD_OF[$next_jid]}"
        next_exp="${EXP_OF[$next_jid]}"
        printf -v next_idx_pad "%02d" "$next_idx"
        next_path="$TRAMP_DIR/tramp_${next_idx_pad}_${next_exp}_${next_method}.sh"
    else
        next_path=""
    fi

    DATA_ROOT="$DPE_DATA_ROOT/$exp"
    LOGDIR="$DPE_CKPT_ROOT/$exp"
    HPO_CONFIG_DIR="$DATA_ROOT/hpo_configs/$method"
    HPO_RESULTS_DIR="$DATA_ROOT/hpo_results/$method"

    cat > "$out_path" <<TRAMPEOF
#!/bin/bash
#SBATCH --job-name=tramp_${idx_pad}_${method}
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=$TRAMP_DIR/logs/tramp_${idx_pad}_${method}_%j.out
#SBATCH --error=$TRAMP_DIR/logs/tramp_${idx_pad}_${method}_%j.err

THIS_INDEX=$i
THIS_METHOD="$method"
THIS_EXP="$exp"
THIS_ARRAY_JID="$jid"
PREV_ARRAY_JID="$prev_jid"
NEXT_TRAMP="$next_path"
TIME_LIMIT="$tl"
WORKDIR="$WORKDIR"
HPO_CONFIG_DIR="$HPO_CONFIG_DIR"
HPO_RESULTS_DIR="$HPO_RESULTS_DIR"
LOGDIR="$LOGDIR"
TRAMP_DIR="$TRAMP_DIR"

source ~/.bashrc
conda activate fac

echo "[tramp \$THIS_INDEX] method=\$THIS_METHOD exp=\$THIS_EXP own_array_jid=\$THIS_ARRAY_JID prev_array_jid=\$PREV_ARRAY_JID"

# kill predecessor's array equivalent (its preempt batch just finished)
if [ -n "\$PREV_ARRAY_JID" ]; then
    echo "[tramp \$THIS_INDEX] scancel \$PREV_ARRAY_JID"
    scancel "\$PREV_ARRAY_JID" || true
fi

# compute sparse array spec from missing trial json files
SPEC=\$(cd "\$WORKDIR" && python -m src.utils.missing_array \\
       --results-dir "\$HPO_RESULTS_DIR" --num-trials 100)

if [ -z "\$SPEC" ]; then
    echo "[tramp \$THIS_INDEX] no missing trials; chaining"
    if [ -n "\$NEXT_TRAMP" ]; then
        sbatch "\$NEXT_TRAMP"
    else
        echo "[tramp \$THIS_INDEX] last tramp; scancel \$THIS_ARRAY_JID"
        scancel "\$THIS_ARRAY_JID" || true
    fi
    exit 0
fi

echo "[tramp \$THIS_INDEX] preempt array spec: \$SPEC"
mkdir -p "\$LOGDIR/logs"

PJID=\$(sbatch --parsable \\
    --job-name="pre_\${THIS_METHOD}" \\
    --partition=preempt \\
    --qos=preempt_qos \\
    --time="\$TIME_LIMIT" \\
    --mem=32G \\
    --gpus=1 \\
    --cpus-per-task=4 \\
    --array="\${SPEC}%24" \\
    --requeue \\
    --exclude=babel-u5-32 \\
    --output="\$LOGDIR/logs/pre_\${THIS_METHOD}_%A_%a.out" \\
    --error="\$LOGDIR/logs/pre_\${THIS_METHOD}_%A_%a.err" \\
    --wrap="source ~/.bashrc && conda activate fac && cd \$WORKDIR && export HDF5_USE_FILE_LOCKING=FALSE && python -m ex.\${THIS_EXP}.hpo_trial --method \$THIS_METHOD --config-file \$HPO_CONFIG_DIR/trial_\\\${SLURM_ARRAY_TASK_ID}.json --eval-pairs 0:0,1:0,2:0,3:0 --output-dir \$HPO_RESULTS_DIR")

echo "[tramp \$THIS_INDEX] submitted preempt array: \$PJID"

# wait for preempt to reach RUNNING (or terminate) before freeing array fallback
echo "[tramp \$THIS_INDEX] waiting for preempt \$PJID to reach RUNNING..."
while true; do
    if squeue -j "\$PJID" -h -t R 2>/dev/null | grep -q .; then
        echo "[tramp \$THIS_INDEX] preempt \$PJID is now RUNNING"
        break
    fi
    if ! squeue -j "\$PJID" -h 2>/dev/null | grep -q .; then
        echo "[tramp \$THIS_INDEX] preempt \$PJID terminated before reaching RUNNING"
        break
    fi
    sleep 30
done

# scancel own array now that preempt is running
echo "[tramp \$THIS_INDEX] scancel own array: \$THIS_ARRAY_JID"
scancel "\$THIS_ARRAY_JID" || true

# chain next trampoline (or final cleanup if last)
if [ -n "\$NEXT_TRAMP" ]; then
    sbatch --parsable --dependency=afterany:\$PJID "\$NEXT_TRAMP"
    echo "[tramp \$THIS_INDEX] chained next: \$NEXT_TRAMP (afterany:\$PJID)"
else
    sbatch --parsable --dependency=afterany:\$PJID \\
        --job-name="tramp_final" \\
        --partition=cpu --qos=cpu_qos \\
        --time=00:05:00 --cpus-per-task=1 --mem=1G \\
        --output="\$TRAMP_DIR/logs/tramp_final_%j.out" \\
        --error="\$TRAMP_DIR/logs/tramp_final_%j.err" \\
        --wrap="echo 'final cleanup: scancel \$THIS_ARRAY_JID'; scancel \$THIS_ARRAY_JID || true"
    echo "[tramp \$THIS_INDEX] last tramp; scheduled final scancel for \$THIS_ARRAY_JID"
fi
TRAMPEOF
    chmod +x "$out_path"
done

FIRST_TRAMP=$(ls "$TRAMP_DIR"/tramp_00_*.sh)
echo "wrote $N trampolines."
echo
echo "=== submitting tramp_00 ==="
JOB=$(sbatch --parsable "$FIRST_TRAMP")
echo "submitted: $JOB ($FIRST_TRAMP)"
echo "monitor:   squeue -u $USER"
echo "tramp dir: $TRAMP_DIR"
