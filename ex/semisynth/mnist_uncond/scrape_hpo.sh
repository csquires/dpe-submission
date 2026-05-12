#!/bin/bash
# scrape_hpo.sh
# collect hpo results from compute node NFS mounts after slurm jobs finish.
# usage: bash scrape_hpo.sh <JOBID>
set -eu

JOBID=${1:?"usage: $0 <JOBID>"}
REMOTE_SUFFIX="aviamala/ckpt/dpe-submission/mnist_eldr_estimation"
LOCAL="ex/semisynth/mnist_uncond/hpo_results"

mkdir -p "$LOCAL"

nodes=$(sacct -j "$JOBID" --format=NodeList%30 --noheader | sort -u | tr -d ' ')

for node in $nodes; do
    [ -z "$node" ] && continue
    src="/compute/$node/$REMOTE_SUFFIX"
    if [ -d "$src" ]; then
        echo "copying from $src..."
        cp -rn "$src/"* "$LOCAL/" 2>/dev/null || echo "  empty or failed: $node"
    else
        echo "  not mounted or missing: $src"
    fi
done

echo "done. results in $LOCAL/"
