#!/bin/bash
# submit.sh: thin wrapper that submits a slurm array job, activates conda env,
# and invokes python -m ex.utils.hpo.optuna.submit with study config.

set -e -u

config="" lane="" replicas="1" concurrency=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--config) config="$2"; shift 2 ;;
		--lane) lane="$2"; shift 2 ;;
		--replicas) replicas="$2"; shift 2 ;;
		--concurrency) concurrency="$2"; shift 2 ;;
		*) echo "usage: $0 --config PYMODULE_PATH --lane LANE_NAME [--replicas R] [--concurrency K]"; exit 2 ;;
	esac
done

[[ -z "$config" ]] && echo "usage: $0 --config PYMODULE_PATH --lane LANE_NAME [--replicas R] [--concurrency K]" && exit 2
[[ -z "$lane" ]] && echo "usage: $0 --config PYMODULE_PATH --lane LANE_NAME [--replicas R] [--concurrency K]" && exit 2
[[ -z "${DPE_DATA_ROOT:-}" ]] && echo "error: DPE_DATA_ROOT not set" && exit 1

# resolve lane profile
lane_data=$(python -c "import sys; sys.path.insert(0, '.'); from ex.utils.hpo.optuna.lanes import get_lane; p = get_lane('${lane}'); print(p.partition, p.qos, p.gpus, p.cpus_per_task, p.mem, p.worker_walltime)" 2>/dev/null) || {
	echo "error: failed to resolve lane '${lane}'"; exit 1; }

# split into vars
read -r partition qos gpus cpus_per_task mem worker_walltime <<< "$lane_data"

[[ -z "$partition" ]] && echo "error: lane resolution returned empty partition" && exit 1
[[ -z "$gpus" ]] && echo "error: lane resolution returned empty gpus" && exit 1
[[ -z "$cpus_per_task" ]] && echo "error: lane resolution returned empty cpus_per_task" && exit 1
[[ -z "$mem" ]] && echo "error: lane resolution returned empty mem" && exit 1
[[ -z "$worker_walltime" ]] && echo "error: lane resolution returned empty worker_walltime" && exit 1

n_methods=$(python -c "import sys; sys.path.insert(0, '.'); from ${config} import CONFIG; print(len(CONFIG.methods))" 2>/dev/null) || {
	echo "error: failed to resolve n_methods from config"; exit 1; }
[[ ! "$n_methods" =~ ^[0-9]+$ ]] || [[ "$n_methods" -le 0 ]] && echo "error: n_methods must be positive integer" && exit 1

array_size=$((replicas * n_methods))
[[ "$array_size" -le 0 ]] && echo "error: array_size (replicas * n_methods) must be positive integer" && exit 1

concurrency="${concurrency:-${SLURM_CONCURRENCY:-16}}"

mkdir -p logs || exit 1

echo "config: $config"
echo "lane: $lane"
echo "replicas: $replicas"
echo "n_methods: $n_methods"
echo "array_size: $array_size"
echo "partition: $partition"
echo "qos: ${qos:-<empty>}"
echo "gpus: $gpus"
echo "cpus_per_task: $cpus_per_task"
echo "mem: $mem"
echo "worker_walltime: $worker_walltime"
echo "concurrency: $concurrency"

export DPE_DATA_ROOT

job_name="optuna_${config##*.}"

sbatch_args=(
	--array="0-$((array_size - 1))%${concurrency}"
	--partition="${partition}"
	--time="${worker_walltime}"
	--cpus-per-task="${cpus_per_task}"
	--mem="${mem}"
	--job-name="${job_name}"
	--output="logs/optuna_%A_%a.out"
)

# add qos flag only if non-empty
[[ -n "$qos" ]] && sbatch_args+=(--qos="${qos}")

# add gpus flag only if gpus > 0
[[ "$gpus" -gt 0 ]] && sbatch_args+=(--gpus="${gpus}")

sbatch_args+=(--wrap="source ~/.bashrc && conda activate fac && cd ${PWD} && python -m ex.utils.hpo.optuna.submit --config ${config} --lane ${lane}")

sbatch_output=$(sbatch "${sbatch_args[@]}" 2>&1) || { echo "error: sbatch submission failed"; exit 1; }

echo "$sbatch_output"
