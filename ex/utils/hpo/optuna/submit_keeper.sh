#!/bin/bash
# submit_keeper.sh: sbatch the optuna preempt-keeper to the cpu partition.
#
# the keeper tops up preempt-partition optuna workers until every study in the
# given StudyConfig reaches target_trials. it dispatches workers to the preempt
# partition using its own internal LaneProfile config. it is the preempt half of
# the double-ended drain; submit.sh provides the stable array-partition baseline:
#
#   bash submit.sh        --config <module> --partition array   # baseline
#   bash submit_keeper.sh --config <module>                     # preempt top-up
#
# any flags after --config are forwarded verbatim to keeper.py (e.g.
# --jobs-per-method, --my-cap, --max-cycles, --dry-run).

set -e -u

config=""
extra=()
while [[ $# -gt 0 ]]; do
	case "$1" in
		--config) config="$2"; shift 2 ;;
		*) extra+=("$1"); shift ;;
	esac
done

[[ -z "$config" ]] && echo "usage: $0 --config PYMODULE_PATH [keeper flags...]" && exit 2
[[ -z "${DPE_DATA_ROOT:-}" ]] && echo "error: DPE_DATA_ROOT not set" && exit 1

export DPE_DATA_ROOT
workdir="$(pwd)"
mkdir -p logs

# the keeper runs on cpu; --requeue gives free auto-restart on cpu preemption
# (the keeper is idempotent -- it recomputes all state from the journals).
sbatch_output=$(sbatch --parsable \
	--job-name="optkeeper_${config##*.}" \
	--partition="${KEEPER_PARTITION:-cpu}" \
	--qos="${KEEPER_QOS:-cpu_qos}" \
	--time="${KEEPER_TIME:-24:00:00}" \
	--cpus-per-task=2 \
	--mem=4G \
	--requeue \
	--output="logs/optkeeper_%j.out" \
	--wrap="source ~/.bashrc && conda activate fac && cd ${workdir} && python -m ex.utils.hpo.optuna.keeper --config ${config} --workdir ${workdir} ${extra[*]:-}") \
	|| { echo "error: sbatch submission failed"; exit 1; }

echo "submitted keeper: ${sbatch_output}"
echo "config:  ${config}"
echo "monitor: squeue -u \$USER -n optkeeper_${config##*.}"
