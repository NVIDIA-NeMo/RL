#!/bin/bash
# Cross-node NVLink: does MNNVL fabric actually span nodes in one NVL72 domain?
#
# The same-node result (104 GB/s at 512MB) only proves NVLink within a chassis.
# MNNVL is supposed to reach the whole domain, and that difference decides
# whether this covers ~13% of our traffic or all of it. One task per node,
# rendezvous over the shared filesystem.
set -uo pipefail
export PYTHONPATH=$PWD
export NRL_IGNORE_VERSION_MISMATCH=1
RV="$1"; MB="$2"; PROTO="${3:-nvlink}"
case "$SLURM_PROCID" in
  0) ROLE=producer ;;
  *) ROLE=consumer ;;   # _FileQueue.get polls for .meta; no sleep needed
esac
echo "[$(hostname)] procid=$SLURM_PROCID role=$ROLE proto=$PROTO ${MB}MB"
python tools/smoke_nvlink_normal_mode.py \
  --protocol "$PROTO" --mb "$MB" --role "$ROLE" --rendezvous "$RV" 2>&1 \
  | grep -viE "FutureWarning|import pynvml|topology.cpp|GID table" | tail -14
