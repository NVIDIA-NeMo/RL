#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# swe_sc.sh — 48-node Ultra SWE via SingleController + HONOURED TransferQueue.
#
# Batch launch (ray.sub runs the SC driver directly — no interactive idle).
# Reuses swe.env's 48-node Ultra shape (train 32 / gen 16, 550B, 65k) but points
# the launcher at run_grpo_single_controller.py + tiny_swe_teacher_sc.yaml.
#
# Run from a NETWORKED shell (or via the slurm bridge):
#     bash swe_sc.sh                       # DRY_RUN inherited from swe.env (=1): inspect first
#     DRY_RUN=0 bash swe_sc.sh             # submit the batch job
#     DRY_RUN=0 bash swe_sc.sh async_rl.max_inflight_prompts=4   # the inflight=4 arm
#
# NRL_ENTRYPOINT is honoured by ultra_launch.sh (uv run ${NRL_ENTRYPOINT:-...}).
# The SC script is spawned from /lustre (the container's baked examples/ lacks it).
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Capture DRY_RUN passed on the command line BEFORE sourcing swe.env (which
# forces export DRY_RUN=1 and would otherwise clobber the caller's value).
_DRY_RUN_IN="${DRY_RUN:-}"

set -a
# shellcheck disable=SC1091
source "${HERE}/swe.env"

# API keys (WANDB_API_KEY, HF_TOKEN) — sourced so W&B logging auto-enables and
# gated HF downloads work. Kept out of the repo; skipped silently if absent.
_SECRETS="/lustre/fs1/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/secrets.sh"
# shellcheck disable=SC1090
[ -f "${_SECRETS}" ] && source "${_SECRETS}"

# --- SingleController + TransferQueue overrides ------------------------------
NRL_ENTRYPOINT="${CODE_DIR}/examples/run_grpo_single_controller.py"
CONFIG_PATH="${CODE_DIR}/examples/configs/ultra/tiny_swe_teacher_sc.yaml"
EXP_NAME=ultra-swe-sc-tq-zhiyul
WANDB_PROJ=nemorl-dataplane-zhiyul   # W&B project for the TransferQueue data-plane experiments
RESULTS_DIR="${WORKSPACE_DIR}/results/${EXP_NAME}"
BASE_LOG_DIR="${WORKSPACE_DIR}/ray_logs/${EXP_NAME}"
# Restore the caller's DRY_RUN (swe.env just reset it to 1).
[ -n "${_DRY_RUN_IN}" ] && DRY_RUN="${_DRY_RUN_IN}"
set +a

# GBS=32 override set (num_prompts_per_step=2, num_generations=16). The LEGACY
# comparison run is matched DOWN to this same GBS=32 so the two are on par —
# this keeps the TQ/SC run at the shape it's proven at. The SC split path REQUIRES
#   grpo.num_prompts_per_step * grpo.num_generations_per_prompt == policy.train_global_batch_size
# (single_controller_utils/config.py:103), so these three move together. Passed
# as one unit here (not split on the command line) to avoid a half-applied
# mismatch. "$@" comes last so a caller can still override the whole set.
bash "${HERE}/ultra_launch.sh" \
  grpo.num_prompts_per_step=2 \
  grpo.num_generations_per_prompt=16 \
  policy.train_global_batch_size=32 \
  "$@"
