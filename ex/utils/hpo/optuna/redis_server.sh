#!/bin/bash
# launch the long-lived redis-server that backs optuna hpo journal storage.
#
# every hpo worker writes its journal through this one redis server over tcp,
# so redis is the only process that ever appends to disk (its own AOF). this
# removes the concurrent-NFS-file-append corruption that the JournalFileBackend
# suffered. on (re)start the server publishes its <host>:<port> to
# $DPE_DATA_ROOT/redis/endpoint, which storage.py reads to build the url.
#
#   bash redis_server.sh        # sbatch the server
#
#SBATCH --job-name=optredis
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --requeue
#SBATCH --output=logs/optredis_%j.out
source ~/.bashrc
conda activate fac
set -u

[[ -z "${DPE_DATA_ROOT:-}" ]] && { echo "error: DPE_DATA_ROOT not set"; exit 1; }

PORT=6390
RDIR="$DPE_DATA_ROOT/redis"
HOST="$(hostname -s)"
mkdir -p "$RDIR/data"

# publish the endpoint for workers to discover (rewritten on every requeue).
echo "${HOST}:${PORT}" > "$RDIR/endpoint"
echo "redis endpoint: ${HOST}:${PORT}   data dir: $RDIR/data"

# minimal config: reachable from all nodes, AOF persistence so a requeue
# replays cleanly. redis is single-threaded -> appends to its AOF are the
# only disk writes and never concurrent.
cat > "$RDIR/redis.conf" <<EOF
bind 0.0.0.0
port ${PORT}
protected-mode no
dir ${RDIR}/data
appendonly yes
appendfsync everysec
save 300 1
EOF

# exec so SIGTERM at requeue reaches redis directly (it saves and exits).
exec redis-server "$RDIR/redis.conf"
