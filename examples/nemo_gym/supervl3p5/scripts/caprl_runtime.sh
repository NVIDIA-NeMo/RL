#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Container entrypoint; called by the CapRL launcher, not submitted directly.
set -euo pipefail

cd /opt/nemo-rl
: "${CAPRL_RUN_DIR:?The launcher must set CAPRL_RUN_DIR}"
export PYTHONUNBUFFERED=1
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://forge.coreweave.com/api/wandb}"
export PYTHONPATH="/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src${PYTHONPATH:+:${PYTHONPATH}}"
export NEMO_RL_VIDEO_MEDIA_ROOT=/lustre
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_IGNORE_VERSION_MISMATCH=1
export NRL_FORCE_REBUILD_VENVS=false
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export UV_NO_SYNC=1
export HF_HOME="${CAPRL_RUN_DIR}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CAPRL_RUN_DIR}/cache/hf-datasets"
export HF_MODULES_CACHE="${CAPRL_RUN_DIR}/cache/hf-modules"
export CAPRL_CONVERSION_CACHE=/opt/caprl-initial-cache
export NRL_MEGATRON_CHECKPOINT_DIR="${CAPRL_CONVERSION_CACHE}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/caprl-${SLURM_JOB_ID}/inductor"
export TRITON_CACHE_DIR="/tmp/caprl-${SLURM_JOB_ID}/triton"
mkdir -p "${HF_MODULES_CACHE}" "${HF_DATASETS_CACHE}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

GEN_PYTHON=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
CONFIG=/opt/nemo-rl/examples/configs/recipes/vlm/vlm_grpo-supervl3p5-caprl-video-32n4g-megatron-prefix-smoke.yaml

if [[ "${1:-}" == setup ]]; then
  for interpreter in \
    /opt/nemo_rl_venv/bin/python \
    "${GEN_PYTHON}" \
    /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python \
    /opt/gym_venvs/responses_api_models/vllm_model/.venv/bin/python \
    /opt/gym_venvs/responses_api_agents/simple_agent/.venv/bin/python \
    /opt/gym_venvs/resources_servers/mcqa/.venv/bin/python \
    /opt/gym_venvs/environment_servers/legacy_agent/.venv/bin/python; do
    [[ -x "${interpreter}" ]] || { echo "Missing image environment: ${interpreter}" >&2; exit 1; }
  done
  echo "CapRL image environments present on $(hostname)"
  exit 0
fi

uv run --no-project --no-sync --python "${GEN_PYTHON}" python \
  examples/nemo_gym/supervl3p5/scripts/preflight_caprl.py --runtime --config "${CONFIG}" "$@"

exec uv run --no-project --no-sync --python /opt/nemo_rl_venv/bin/python python \
  examples/run_grpo_single_controller.py --config "${CONFIG}" "$@"
