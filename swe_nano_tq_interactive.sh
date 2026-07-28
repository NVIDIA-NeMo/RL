#!/bin/bash
# =============================================================================
# swe_nano_tq_interactive.sh — interactive nano SWE run with the TransferQueue
# data plane ENABLED (sync GRPO).
#
# Same 6-node nano shape as swe_nano_interactive.sh, but points at the sync + TQ
# config (nano_swe_teacher_sync_tq.yaml): async_grpo disabled, data_plane.enabled
# =true, impl=transfer_queue. Use this to CHECK the data plane end to end.
#
# Run from a NETWORKED shell:
#     bash swe_nano_tq_interactive.sh
#     bash swe_nano_tq_interactive.sh grpo.max_num_steps=1     # extra hydra overrides pass through
#
# Then: bash <jobid>-attach.sh ; source <jobid>-run-cmd.sh   (edit + re-source to iterate)
#
# Verify TQ engaged: the driver log must show "🚀 Running synchronous GRPO
# training" (not "Running async GRPO training") plus TransferQueueController /
# SimpleStorageUnit actors. For async + TQ use swe_nano_sc_interactive.sh.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -a
# shellcheck disable=SC1091
source "${HERE}/swe_nano.env"

# --- override for the sync + TransferQueue variant --------------------------
EXP_NAME=nano-swe-tq-zhiyul
CONFIG_PATH="${CODE_DIR}/examples/configs/ultra/nano_swe_teacher_sync_tq.yaml"
RESULTS_DIR="${WORKSPACE_DIR}/results/${EXP_NAME}"
BASE_LOG_DIR="${WORKSPACE_DIR}/ray_logs/${EXP_NAME}"
set +a

INTERACTIVE=1 DRY_RUN=0 INTERACTIVE_WAIT="${INTERACTIVE_WAIT:-1}" \
  bash "${HERE}/ultra_launch.sh" "$@"
