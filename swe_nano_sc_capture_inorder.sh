#!/bin/bash
# =============================================================================
# swe_nano_sc_capture.sh — BATCH launch of the nano SWE SingleController +
# TransferQueue recipe with gate-authoritative token capture enabled.
#
# Batch counterpart of swe_nano_sc_capture_interactive.sh, exactly as
# swe_nano_sc.sh is the batch counterpart of swe_nano_sc_interactive.sh:
# same capture env block (driver PYTHONPATH + orjson, orjson into the prefetched
# vllm worker venv) and the same two hydra appends; ray.sub runs the driver directly.
#
# Run from a NETWORKED shell at the repo root:
#     DRY_RUN=0 SC_EXP_NAME=<exp> bash swe_nano_sc_capture.sh [extra overrides]
# Capture-canonical fingerprints: export NG_TIC_FP_CANONICAL=1 (recommended).
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_DRY_RUN_IN="${DRY_RUN:-}"
_WALLTIME_IN="${WALLTIME:-}"

set -a
# shellcheck disable=SC1091
# source "${HERE}/swe_nano.env"
# source /lustre/fsw/portfolios/coreai/users/amahishi/projects/nemo-rl-workspace/resiliency/partial-rollout-checkpointing/swe_nano.env

NRL_ENTRYPOINT="${CODE_DIR}/examples/run_grpo_single_controller.py"

# Canonical history fingerprints. Without this, fingerprints never match and
# token_in_rate degrades to ~0 (docs/guides/nano-swe-token-capture.md).
NG_TIC_FP_CANONICAL=1
NRL_DRIVER_PYTHONPATH="/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
NRL_DRIVER_PIP_INSTALL="orjson"
NRL_VLLM_WORKER_PIP_INSTALL="orjson"
# Must equal the cluster's ray (uv.lock); Gym's own floor spec would drift.
NRL_GYM_VENV_PIP_INSTALL="ray[default]==2.56.1"

# --- Per-call latency breakdown (CALL_TIMING=0 to disable) --------------------
if [ "${CALL_TIMING:-1}" = "1" ]; then
  NRL_CALL_TIMING_DIR="${NRL_CALL_TIMING_DIR:-${WORKSPACE_DIR}/call_timing/${EXP_NAME}}"
  NG_CALL_TIMING_DIR="${NG_CALL_TIMING_DIR:-${NRL_CALL_TIMING_DIR}}"
  mkdir -p "${NRL_CALL_TIMING_DIR}"
fi
[ -n "${_DRY_RUN_IN}" ] && DRY_RUN="${_DRY_RUN_IN}"
[ -n "${_WALLTIME_IN}" ] && WALLTIME="${_WALLTIME_IN}"
set +a

bash "${HERE}/ultra_launch.sh" \
  +env.nemo_gym.rollout_max_attempts_to_avoid_lp_nan=1 \
  "$@"
# +token_capture.enabled=true \