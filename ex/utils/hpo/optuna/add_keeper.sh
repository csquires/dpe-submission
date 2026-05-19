#!/bin/bash
# register a campaign keeper into the already-running redis job.
#
# appends the StudyConfig module to $DPE_DATA_ROOT/redis/keepers.txt; the redis
# job's keeper supervisor (see redis_server.sh) picks it up within ~30s and
# spawns the keeper as a child process. no new slurm job, no redis restart --
# campaigns can be started while others are already running.
#
#   bash add_keeper.sh ex.utils.hpo.optuna.configs.eig_reg
#
set -u

CFG="${1:-}"
[[ -z "$CFG" ]] && { echo "usage: $0 <dotted.config.module>"; exit 2; }
[[ -z "${DPE_DATA_ROOT:-}" ]] && { echo "error: DPE_DATA_ROOT not set"; exit 1; }

RDIR="$DPE_DATA_ROOT/redis"
JOBF="$RDIR/jobid"
REG="$RDIR/keepers.txt"

[[ -f "$JOBF" ]] || {
	echo "error: no redis jobid at $JOBF; launch redis_server.sh first"
	exit 1
}
JOBID="$(cat "$JOBF")"
squeue -h -j "$JOBID" -o "%T" 2>/dev/null | grep -q RUNNING || {
	echo "error: redis job $JOBID is not RUNNING"
	exit 1
}

mkdir -p "$RDIR/done"
if grep -qxF "$CFG" "$REG" 2>/dev/null; then
	echo "$CFG already registered"
else
	echo "$CFG" >> "$REG"
	echo "registered $CFG"
fi
# clear any stale done-marker so a re-added campaign is picked up again.
rm -f "$RDIR/done/$CFG"
echo "redis job $JOBID supervisor will spawn its keeper within ~30s"
echo "  log: logs/keeper_${CFG##*.}_${JOBID}.out"
