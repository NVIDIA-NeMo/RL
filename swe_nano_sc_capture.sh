#!/bin/bash
# =============================================================================
# swe_nano_sc_capture.sh — BATCH launch of the nano SWE SingleController +
# TransferQueue recipe with gate-authoritative token capture enabled.
#
# Batch counterpart of swe_nano_sc_capture_interactive.sh, exactly as
# swe_nano_sc.sh is the batch counterpart of swe_nano_sc_interactive.sh:
# same capture env block (driver PYTHONPATH + orjson, vllm worker venv
# rebuild) and the same two hydra appends; ray.sub runs the driver directly.
#
# Run from a NETWORKED shell at the repo root:
#     DRY_RUN=0 SC_EXP_NAME=<exp> bash swe_nano_sc_capture.sh [extra overrides]
# Capture-canonical fingerprints: export NG_TIC_FP_CANONICAL=1 (recommended).
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_DRY_RUN_IN="${DRY_RUN:-}"
_WALLTIME_IN="${WALLTIME:-}"
_USE_SNAPSHOT_IN="${USE_SNAPSHOT:-}"

set -a
# shellcheck disable=SC1091
source "${HERE}/swe_nano.env"

EXP_NAME="${SC_EXP_NAME:-nano-swe-sc-tq-capture}"
NRL_ENTRYPOINT="${CODE_DIR}/examples/run_grpo_single_controller.py"
CONFIG_PATH="${CODE_DIR}/examples/configs/ultra/nano_swe_teacher_sc.yaml"
RESULTS_DIR="${WORKSPACE_DIR}/results/${EXP_NAME}"
BASE_LOG_DIR="${WORKSPACE_DIR}/ray_logs/${EXP_NAME}"

NRL_FORCE_REBUILD_VENVS_LIST="nemo_rl.environments.nemo_gym.NemoGym,nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
NRL_DRIVER_PYTHONPATH="/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
NRL_DRIVER_PIP_INSTALL="orjson"
# Ray is started from the container's prefetched environment. Keep the driver on
# that exact Python/Ray pair; only class-specific worker venvs are rebuilt below.
NRL_DRIVER_UV_RUN_FLAGS="--locked --no-sync"

# --- Per-call latency breakdown (CALL_TIMING=0 to disable) --------------------
if [ "${CALL_TIMING:-1}" = "1" ]; then
  NRL_CALL_TIMING_DIR="${NRL_CALL_TIMING_DIR:-${WORKSPACE_DIR}/call_timing/${EXP_NAME}}"
  NG_CALL_TIMING_DIR="${NG_CALL_TIMING_DIR:-${NRL_CALL_TIMING_DIR}}"
  mkdir -p "${NRL_CALL_TIMING_DIR}"
fi
[ -n "${_DRY_RUN_IN}" ] && DRY_RUN="${_DRY_RUN_IN}"
[ -n "${_WALLTIME_IN}" ] && WALLTIME="${_WALLTIME_IN}"
[ -n "${_USE_SNAPSHOT_IN}" ] && USE_SNAPSHOT="${_USE_SNAPSHOT_IN}"
set +a

bash "${HERE}/ultra_launch.sh" \
  token_capture.enabled=true \
  +env.nemo_gym.rollout_max_attempts_to_avoid_lp_nan=1 \
  "$@"
