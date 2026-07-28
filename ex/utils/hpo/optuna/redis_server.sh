#!/bin/bash
# long-lived redis-server for optuna hpo journal storage, with an embedded
# keeper supervisor.
#
# redis is the single process that writes hpo journals to disk (its own AOF),
# which removes the concurrent-NFS-append corruption JournalFileBackend
# suffered. this one job also hosts every campaign keeper: a watcher loop
# spawns one keeper per config listed in $DPE_DATA_ROOT/redis/keepers.txt, so
# a keeper is a child process here rather than its own slurm job. add_keeper.sh
# appends to that registry to start a campaign into an already-running redis
# job without a restart or an extra slurm job.
#
#   bash redis_server.sh                                    # sbatch this job
#   bash add_keeper.sh ex.utils.hpo.optuna.configs.eig_reg  # add a campaign
#
#SBATCH --job-name=optredis
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --output=logs/optredis_%j.out
source ~/.bashrc
conda activate fac
set -u

[[ -z "${DPE_DATA_ROOT:-}" ]] && { echo "error: DPE_DATA_ROOT not set"; exit 1; }
WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"
mkdir -p logs

PORT=6390
RDIR="$DPE_DATA_ROOT/redis"
REG="$RDIR/keepers.txt"
HOST="$(hostname -s)"
mkdir -p "$RDIR/data" "$RDIR/done"

# publish endpoint + jobid for storage.py and add_keeper.sh to discover
# (rewritten on every requeue, since the host may change).
echo "${HOST}:${PORT}" > "$RDIR/endpoint"
echo "${SLURM_JOB_ID}" > "$RDIR/jobid"
echo "redis endpoint: ${HOST}:${PORT}   jobid: ${SLURM_JOB_ID}"

# self-chain: sbatch a successor that fires when this job ends for any
# reason (TIMEOUT after 24h, preempt, scancel). slurm's --requeue does not
# fire on TIMEOUT, which is the regular exit mode of a long-lived job, so
# self-chaining is the only way to keep an uninterrupted optredis line up.
# guarded by OPTREDIS_NOCHAIN=1 for manual one-shot runs.
if [[ "${OPTREDIS_NOCHAIN:-0}" != "1" ]]; then
    # --export=ALL + --chdir keep DPE_* env and a VALID working dir across the
    # chain. without them a successor lost DPE_DATA_ROOT and ran with a broken
    # cwd (/dpe-submission), so keepers' sbatch died on "getcwd failed" and
    # dispatched nothing. WORKDIR is this job's (valid) dir, propagated forward.
    SUCC=$(sbatch --parsable --dependency=afterany:"${SLURM_JOB_ID}" \
        --export=ALL --chdir="$WORKDIR" \
        "${BASH_SOURCE[0]}" 2>/dev/null || echo "")
    if [[ -n "$SUCC" ]]; then
        echo "[redis] self-chained successor jobid: $SUCC"
    else
        echo "[redis] WARNING: self-chain sbatch failed; no successor queued"
    fi
fi

# minimal config: reachable from all nodes, AOF persistence so a requeue
# replays cleanly. redis is single-threaded -> its AOF append is the only
# disk write and is never concurrent.
#
# RDB snapshots are DISABLED (save ""): the journal DB grows to tens of millions
# of keys, whose 6GB+ bgsave forks fail/get-killed and, with the default
# stop-writes-on-bgsave-error yes, that BLOCKS all writes (workers can't record
# trials) and litters temp-*.rdb files. AOF is the intended persistence anyway,
# so drop RDB and also disable the write-block as belt-and-suspenders.
cat > "$RDIR/redis.conf" <<EOF
bind 0.0.0.0
port ${PORT}
protected-mode no
dir ${RDIR}/data
appendonly yes
appendfsync everysec
save ""
stop-writes-on-bgsave-error no
EOF

redis-server "$RDIR/redis.conf" &
REDIS_PID=$!

# wait for redis to accept connections before spawning keepers.
for _ in $(seq 1 30); do
	redis-cli -h "$HOST" -p "$PORT" ping 2>/dev/null | grep -q PONG && break
	sleep 1
done

# keeper supervisor: each tick, spawn a keeper for every registered config
# that has no live keeper and has not already finished. a crashed keeper
# (nonzero exit) is respawned next tick; a keeper that exits cleanly (campaign
# reached target) drops a done/ marker and is left alone. the registry and
# done/ markers live on $DPE_DATA_ROOT, so this all survives a job requeue.
declare -A KPID
echo "[redis] keeper supervisor watching ${REG}"
while kill -0 "$REDIS_PID" 2>/dev/null; do
	if [[ -f "$REG" ]]; then
		while read -r cfg; do
			[[ -z "$cfg" || "$cfg" == \#* ]] && continue
			[[ -f "$RDIR/done/$cfg" ]] && continue
			pid="${KPID[$cfg]:-}"
			[[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && continue
			(
				python -m ex.utils.hpo.optuna.keeper \
					--config "$cfg" --workdir "$WORKDIR" \
					>> "logs/keeper_${cfg##*.}_${SLURM_JOB_ID}.out" 2>&1
				if [[ $? -eq 0 ]]; then
					# campaign hit target -> sbatch a holdout job per method,
					# then mark done. holdout failures are logged but do not
					# block the done-marker.
					ms=$(python -c "from ex.utils.hpo.optuna.study_config import load_config; print(' '.join(load_config('$cfg').methods))")
					for m in $ms; do
						bash ex/utils/hpo/optuna/submit_holdout.sh \
							--config "$cfg" --method "$m" \
							>> "logs/keeper_${cfg##*.}_${SLURM_JOB_ID}.out" 2>&1 || true
					done
					touch "$RDIR/done/$cfg"
				fi
			) &
			KPID[$cfg]=$!
			echo "[redis] spawned keeper for ${cfg} (pid ${KPID[$cfg]})"
		done < "$REG"
	fi
	sleep 30
done
echo "[redis] redis-server exited; job ending"
