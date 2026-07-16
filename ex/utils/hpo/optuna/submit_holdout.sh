#!/bin/bash
# fan-out launcher: sbatch a single array job whose elements are
# (candidate, cell) pairs (see run_holdout.py), plus an aggregator job that
# runs after the array finishes (afterany).
#
# per-element job is small (~one full-budget training on one holdout cell),
# so a ~20-min walltime on preempt is safe -- preemption is unlikely on a
# job that short, and --requeue + the idempotent-skip in run_holdout.py
# resumes a preempted element cleanly.
#
#   bash submit_holdout.sh --config <dotted.module> --method <name>

# activate conda before `set -u` -- /etc/bashrc references unbound vars.
source ~/.bashrc
conda activate fac

set -e -u

config=""
method=""
lane=""
while [[ $# -gt 0 ]]; do
	case "$1" in
		--config) config="$2"; shift 2 ;;
		--method) method="$2"; shift 2 ;;
		--lane)   lane="$2";   shift 2 ;;
		*) echo "unknown arg: $1"; exit 2 ;;
	esac
done
[[ -z "$config" || -z "$method" ]] && {
	echo "usage: $0 --config <dotted.module> --method <name> [--lane <name>]"; exit 2
}
[[ -z "${DPE_DATA_ROOT:-}" ]] && { echo "error: DPE_DATA_ROOT not set"; exit 1; }

# resolve holdout lane: explicit --lane wins; else the config's holdout_lane
# (default "holdout" cpu; gpu-only campaigns set "array_gpu"). read via an
# UNQUOTED heredoc so $config expands.
if [ -z "$lane" ]; then
	lane=$(python - <<PY
from ex.utils.hpo.optuna.study_config import load_config
print(getattr(load_config("$config"), "holdout_lane", "holdout"))
PY
)
fi
echo "holdout lane: $lane"

workdir="$(pwd)"
mkdir -p logs

# query slices from config; empty string means "no slices, use current behavior".
# UNQUOTED heredoc so $config expands (quoted <<'PY' left it literal -> the
# load_config('$config') ModuleNotFoundError killed every auto-holdout under set -e).
slices_csv=$(python - <<PY
from ex.utils.hpo.optuna.study_config import load_config
from ex.utils.hpo.optuna.storage import _serialize_slice
cfg = load_config("$config")
if cfg.slices is None:
    print("")
else:
    print(",".join(_serialize_slice(s) for s in cfg.slices))
PY
)

if [ -z "$slices_csv" ]; then
    # no slices: current behavior (byte-identical)

# 1. total element count = |pool| * |holdout_pool|.
n=$(python -m ex.utils.hpo.optuna.run_holdout \
		--config "$config" --method "$method" --list-elements 2>/dev/null \
		| tail -1)
if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -eq 0 ]; then
	echo "error: could not determine element count (got '$n')"; exit 1
fi
echo "$config/$method: $n elements"

base="holdout_${config##*.}_${method}"

# 2. ONE array job consuming the selected lane profile in
# ex/utils/hpo/optuna/lanes.py as the single source of truth for the
# (partition, qos, gpus, cpus_per_task, B, cores, mem, walltime, max_concurrent)
# shape. default lane is "holdout" (cpu/array). pass --lane preempt to target the
# preempt gpu pool. update the lane profile; this script picks it up.
eval "$(python - "$lane" <<'PY'
import sys
from ex.utils.hpo.optuna.lanes import get_lane
l = get_lane(sys.argv[1])
cores = l.cores_per_trial if l.cores_per_trial is not None else 0
B = l.batch_size if l.batch_size is not None else 0
print(f"partition={l.partition}")
print(f"qos={l.qos}")
print(f"gpus={l.gpus}")
print(f"cpus_per_task={l.cpus_per_task}")
print(f"mem={l.mem}")
print(f"B={B}")
print(f"cores={cores}")
print(f"walltime={l.worker_walltime}")
print(f"throttle={l.max_concurrent}")
print(f"constraint={l.constraint}")
PY
)"
for v in partition cpus_per_task mem B walltime throttle; do
	[[ -z "${!v:-}" || "${!v}" == "None" ]] && { echo "error: lane field '$v' not set"; exit 1; }
done
[[ "$B" -eq 0 ]] && { echo "error: lane batch_size not pinned (set lane.batch_size)"; exit 1; }
[[ "$cores" -eq 0 ]] && cores=1  # default if lane leaves it None
n_chunks=$(( (n + B - 1) / B ))

# per-lane extra sbatch flags (qos, gres).
extra=()
[[ -n "$qos" ]] && extra+=(--qos="$qos")
[[ "$gpus" -gt 0 ]] && extra+=(--gres=gpu:"$gpus")
[[ -n "$constraint" && "$constraint" != "None" ]] && extra+=(--constraint="$constraint")

arr_jid=$(sbatch --parsable \
	--job-name="${base}" \
	--array="0-$((n_chunks-1))%${throttle}" \
	--partition="${partition}" \
	"${extra[@]}" \
	--cpus-per-task="${cpus_per_task}" \
	--mem="${mem}" \
	--time="${walltime}" \
	--requeue \
	--output="logs/${base}_%A_%a.out" \
	--wrap="source ~/.bashrc && conda activate fac && cd ${workdir} && python -m ex.utils.hpo.optuna.run_holdout --config ${config} --method ${method} --chunk-idx \${SLURM_ARRAY_TASK_ID} --chunk-size ${B} --cores-per-trial ${cores}")

# 3. aggregator: phase-1 (per-candidate) + phase-2 (per-study winner).
agg_jid=$(sbatch --parsable \
	--job-name="${base}_aggregate" \
	--partition=cpu \
	--qos=cpu_qos \
	--cpus-per-task=2 \
	--mem=4G \
	--time=00:30:00 \
	--requeue \
	--dependency="afterany:${arr_jid}" \
	--output="logs/${base}_aggregate_%j.out" \
	--wrap="source ~/.bashrc && conda activate fac && cd ${workdir} && python -m ex.utils.hpo.optuna.aggregate_holdout --config ${config} --method ${method}")

    echo "submitted array ${arr_jid} (size ${n}, throttle %${throttle}) + aggregator ${agg_jid}"

else
    # slices present: iterate per slice
    for s in $(echo "$slices_csv" | tr ',' ' '); do

        # re-query element count per slice
        n=$(python -m ex.utils.hpo.optuna.run_holdout \
                --config "$config" --method "$method" --slice "$s" --list-elements 2>/dev/null \
                | tail -1)
        if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -eq 0 ]; then
            echo "error: could not determine element count (got '$n') for slice '$s'"; exit 1
        fi
        echo "$config/$method/slice_$s: $n elements"

        # update base name (unique per slice)
        base="holdout_${config##*.}_${method}_slice_${s}"

        # read lane profile (same as no-slice case)
        eval "$(python - "$lane" <<'PY'
import sys
from ex.utils.hpo.optuna.lanes import get_lane
l = get_lane(sys.argv[1])
cores = l.cores_per_trial if l.cores_per_trial is not None else 0
B = l.batch_size if l.batch_size is not None else 0
print(f"partition={l.partition}")
print(f"qos={l.qos}")
print(f"gpus={l.gpus}")
print(f"cpus_per_task={l.cpus_per_task}")
print(f"mem={l.mem}")
print(f"B={B}")
print(f"cores={cores}")
print(f"walltime={l.worker_walltime}")
print(f"throttle={l.max_concurrent}")
print(f"constraint={l.constraint}")
PY
)"
        for v in partition cpus_per_task mem B walltime throttle; do
            [[ -z "${!v:-}" || "${!v}" == "None" ]] && { echo "error: lane field '$v' not set"; exit 1; }
        done
        [[ "$B" -eq 0 ]] && { echo "error: lane batch_size not pinned (set lane.batch_size)"; exit 1; }
        [[ "$cores" -eq 0 ]] && cores=1  # default if lane leaves it None
        n_chunks=$(( (n + B - 1) / B ))

        # per-lane extra sbatch flags (qos, gres).
        extra=()
        [[ -n "$qos" ]] && extra+=(--qos="$qos")
        [[ "$gpus" -gt 0 ]] && extra+=(--gres=gpu:"$gpus")
        [[ -n "$constraint" && "$constraint" != "None" ]] && extra+=(--constraint="$constraint")

        arr_jid=$(sbatch --parsable \
            --job-name="${base}" \
            --array="0-$((n_chunks-1))%${throttle}" \
            --partition="${partition}" \
            "${extra[@]}" \
            --cpus-per-task="${cpus_per_task}" \
            --mem="${mem}" \
            --time="${walltime}" \
            --requeue \
            --output="logs/${base}_%A_%a.out" \
            --wrap="source ~/.bashrc && conda activate fac && cd ${workdir} && python -m ex.utils.hpo.optuna.run_holdout --config ${config} --method ${method} --slice ${s} --chunk-idx \${SLURM_ARRAY_TASK_ID} --chunk-size ${B} --cores-per-trial ${cores}")

        agg_jid=$(sbatch --parsable \
            --job-name="${base}_aggregate" \
            --partition=cpu \
            --qos=cpu_qos \
            --cpus-per-task=2 \
            --mem=4G \
            --time=00:30:00 \
            --requeue \
            --dependency="afterany:${arr_jid}" \
            --output="logs/${base}_aggregate_%j.out" \
            --wrap="source ~/.bashrc && conda activate fac && cd ${workdir} && python -m ex.utils.hpo.optuna.aggregate_holdout --config ${config} --method ${method} --slice ${s}")

        echo "submitted slice_${s}: array ${arr_jid} (size ${n}, throttle %${throttle}) + aggregator ${agg_jid}"
    done
fi
