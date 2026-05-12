#!/bin/bash
# submit.sh: thin wrapper that submits a slurm array job, activates conda env,
# and invokes python -m experiments.utils.hpo.optuna.submit with study config.

set -e -u

config="" array_size="" partition="" time="" cpus="" mem="" concurrency=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--config) config="$2"; shift 2 ;;
		--array-size) array_size="$2"; shift 2 ;;
		--partition) partition="$2"; shift 2 ;;
		--time) time="$2"; shift 2 ;;
		--cpus-per-task) cpus="$2"; shift 2 ;;
		--mem) mem="$2"; shift 2 ;;
		--concurrency) concurrency="$2"; shift 2 ;;
		*) echo "usage: $0 --config PYMODULE_PATH [--array-size N] [--partition P] [--time HH:MM:SS] [--cpus-per-task C] [--mem M] [--concurrency K]"; exit 2 ;;
	esac
done

[[ -z "$config" ]] && echo "usage: $0 --config PYMODULE_PATH [--array-size N] [--partition P] [--time HH:MM:SS] [--cpus-per-task C] [--mem M] [--concurrency K]" && exit 2
[[ -z "${DPE_DATA_ROOT:-}" ]] && echo "error: DPE_DATA_ROOT not set" && exit 1

if [[ -z "$array_size" ]]; then
	array_size=$(python -c "import sys; sys.path.insert(0, '.'); from ${config} import cfg; print(len(cfg.methods))" 2>/dev/null) || {
		echo "error: failed to resolve array size from config"; exit 1; }
	[[ ! "$array_size" =~ ^[0-9]+$ ]] || [[ "$array_size" -le 0 ]] && echo "error: array size must be positive integer" && exit 1
fi

partition="${partition:-${SLURM_PARTITION:-cpu}}"
time="${time:-${SLURM_TIME:-06:00:00}}"
cpus="${cpus:-$(nproc 2>/dev/null || echo 16)}"
cpus="${cpus:-${DPE_CORES_PER_NODE:-16}}"
mem="${mem:-${DPE_MEM_PER_NODE:-32G}}"
concurrency="${concurrency:-${SLURM_CONCURRENCY:-16}}"

mkdir -p logs || exit 1

echo "config: $config"
echo "array_size: $array_size"
echo "partition: $partition"
echo "time: $time"
echo "cpus: $cpus"
echo "mem: $mem"
echo "concurrency: $concurrency"

export DPE_DATA_ROOT

job_name="optuna_${config##*.}"

sbatch_output=$(sbatch \
	--array="0-$((array_size - 1))%${concurrency}" \
	--partition="${partition}" \
	--time="${time}" \
	--cpus-per-task="${cpus}" \
	--mem="${mem}" \
	--job-name="${job_name}" \
	--output="logs/optuna_%A_%a.out" \
	--wrap="source activate fac && python -m experiments.utils.hpo.optuna.submit --config ${config}" \
	2>&1) || { echo "error: sbatch submission failed"; exit 1; }

echo "$sbatch_output"
