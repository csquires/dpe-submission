#!/bin/bash
# Archive the pre-peak data + holdout artifacts to $DPE_DATA_ROOT/archive/<date>/.
#
# Redis prefixes are NOT moved — peak studies use *_peak method names so they
# coexist with the in-flight `experiment:method` prefixes. Inspect the old
# Optuna journals at any time via the existing redis-server.
#
# Usage:
#   bash scripts/archive_pre_peak.sh           # uses today's date
#   bash scripts/archive_pre_peak.sh 2026-06-21
#
# Idempotent: if the destination already exists, the script aborts before
# touching anything.

set -e -u

DATE="${1:-$(date -u +%Y-%m-%d)}"

if [[ -z "${DPE_DATA_ROOT:-}" ]]; then
    echo "error: DPE_DATA_ROOT not set" >&2
    exit 1
fi

ARCHIVE_ROOT="${DPE_DATA_ROOT}/archive/${DATE}"
if [[ -e "$ARCHIVE_ROOT" ]]; then
    echo "error: archive dir already exists: $ARCHIVE_ROOT" >&2
    echo "       remove or pass a different date argument" >&2
    exit 1
fi

mkdir -p "$ARCHIVE_ROOT"
echo "archiving pre-peak data into $ARCHIVE_ROOT"

# move the per-experiment data directories.
for exp in mnist eig occupancy pendulum; do
    src="${DPE_DATA_ROOT}/${exp}"
    if [[ -e "$src" ]]; then
        echo "  mv $src -> $ARCHIVE_ROOT/$exp"
        mv "$src" "$ARCHIVE_ROOT/$exp"
    else
        echo "  skip $src (does not exist)"
    fi
done

# move the holdout artifact tree (pre-peak best_hp.json/aggregate CSVs).
if [[ -e "${DPE_DATA_ROOT}/holdout" ]]; then
    echo "  mv ${DPE_DATA_ROOT}/holdout -> $ARCHIVE_ROOT/holdout"
    mv "${DPE_DATA_ROOT}/holdout" "$ARCHIVE_ROOT/holdout"
fi

# step2 results from previous campaigns (per-experiment subdirs).
for exp in mnist eig occupancy pendulum; do
    src="${DPE_DATA_ROOT}/${exp}_step2"
    if [[ -e "$src" ]]; then
        echo "  mv $src -> $ARCHIVE_ROOT/${exp}_step2"
        mv "$src" "$ARCHIVE_ROOT/${exp}_step2"
    fi
done

echo "done. archive at $ARCHIVE_ROOT"
echo "NOTE: redis journals were left in place. peak studies use *_peak method"
echo "      names so their redis prefixes never collide with the archived ones."
