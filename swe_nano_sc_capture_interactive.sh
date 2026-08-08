#!/bin/bash
# =============================================================================
# swe_nano_sc_capture_interactive.sh — the CAPTURE arm of the nano SWE TQ A/B:
# swe_nano_sc_interactive.sh + gate-authoritative token capture enabled.
#
# Same 6-node SingleController shape as swe_nano_sc_interactive.sh; adds the
# posture the capture path needs (learned on job 5764598, 2026-08-01):
#   token_capture.enabled=true       the recipe yaml defines the block (enabled: false)
#                                    defines the key (pydantic default only)
#   +env.nemo_gym.rollout_max_attempts_to_avoid_lp_nan=1
#                                    capture hard-errors otherwise; pin on the
#                                    legacy arm too when running an A/B pair
#   NRL_DRIVER_PYTHONPATH            driver imports nemo_gym staging records;
#                                    the baked driver venv has no nemo_gym
#   NRL_DRIVER_PIP_INSTALL=orjson    Gym's token_id_capture/__init__ eagerly
#                                    imports consumer->store->orjson (purity
#                                    gap; PR #2278 feedback)
#   VllmAsyncGenerationWorker in NRL_FORCE_REBUILD_VENVS_LIST
#                                    the capture leg needs the VLLM_GYM venv
#                                    (--extra nemo_gym); venv caching is not
#                                    spec-aware (PR #3456 known issue), so a
#                                    legacy-leg venv would be silently reused
#
# Usage (same contract as the other nano launchers):
#     bash swe_nano_sc_capture_interactive.sh [hydra overrides...]
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
set +a

INTERACTIVE=1 DRY_RUN=0 INTERACTIVE_WAIT="${INTERACTIVE_WAIT:-1}" \
  bash "${HERE}/ultra_launch.sh" \
  token_capture.enabled=true \
  +env.nemo_gym.rollout_max_attempts_to_avoid_lp_nan=1 \
  "$@"
