#!/bin/bash
# =============================================================================
# swe_legacy.sh — 48-node Ultra SWE via the LEGACY async path (in-memory
# ReplayBuffer, NO TransferQueue), for a head-to-head throughput comparison
# against the TQ/SingleController run (swe_sc.sh).
#
# Uses the default async entrypoint (run_grpo_nemo_gym.py, async_grpo.enabled)
# + base tiny_swe_teacher.yaml. Batch is matched DOWN to the TQ run's GBS=32
# (num_prompts_per_step=2 x num_generations=16) so the two are on par. Logs to
# the SAME W&B project as swe_sc.sh (nemorl-dataplane-zhiyul) under a distinct
# run name, so both series show side-by-side.
#
#     DRY_RUN=0 bash swe_legacy.sh
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_DRY_RUN_IN="${DRY_RUN:-}"

set -a
# shellcheck disable=SC1091
source "${HERE}/swe.env"

# API keys (WANDB_API_KEY, HF_TOKEN) → W&B auto-enables. Kept out of the repo.
_SECRETS="/lustre/fs1/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/secrets.sh"
# shellcheck disable=SC1090
[ -f "${_SECRETS}" ] && source "${_SECRETS}"

# --- Legacy async overrides (NO NRL_ENTRYPOINT → default run_grpo_nemo_gym.py,
#     NO CONFIG_PATH override → swe.env's base tiny_swe_teacher.yaml) -----------
EXP_NAME=ultra-swe-legacy-async-gbs32-zhiyul
WANDB_PROJ=nemorl-dataplane-zhiyul   # same project as swe_sc.sh for side-by-side
RESULTS_DIR="${WORKSPACE_DIR}/results/${EXP_NAME}"
BASE_LOG_DIR="${WORKSPACE_DIR}/ray_logs/${EXP_NAME}"
# Restore the caller's DRY_RUN (swe.env just reset it to 1).
[ -n "${_DRY_RUN_IN}" ] && DRY_RUN="${_DRY_RUN_IN}"
set +a

# Matched to the TQ run: GBS=32 (num_prompts_per_step=2 x num_generations=16).
EXTRA_OVERRIDES=()
if [ -n "${WANDB_API_KEY:-}" ]; then
  EXTRA_OVERRIDES+=(logger.wandb_enabled=true)
  echo "[swe_legacy] W&B enabled → project=${WANDB_PROJ} name=${EXP_NAME}"
fi

# Match the SC/TQ run's non-batch knobs so the ONLY difference is the data path
# (async in-memory ReplayBuffer here vs TQ+SingleController there):
#   grpo.val_period=0        — SC run has 0 (base has 10000); no val either way in
#                              the compare window, set explicit for cleanliness.
#   checkpointing.enabled=false — SC run disables it; legacy base enables it →
#                              would add checkpoint I/O and skew throughput.
bash "${HERE}/ultra_launch.sh" \
  grpo.num_prompts_per_step=2 \
  grpo.num_generations_per_prompt=16 \
  policy.train_global_batch_size=32 \
  grpo.val_period=0 \
  checkpointing.enabled=false \
  ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"} \
  "$@"
