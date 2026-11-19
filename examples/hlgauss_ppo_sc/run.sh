#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run inside your allocated environment; this script does not submit Slurm jobs.
set -euo pipefail
recipe="${1:-nano_ctrunc.yaml}"
if (( $# > 0 )); then shift; fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_dir}"
if [[ -n "${ENV_FILE:-}" ]]; then
  source "${ENV_FILE}"
fi
# Must follow every environment file and precede Ray / uv startup.
unset RAY_LOG_SYNC_FREQUENCY
: "${MODEL_PATH:?Set MODEL_PATH to the actor HF model}"
: "${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH to your prepared SWE training data}"
: "${OPENSANDBOX_DOMAIN:?Set OPENSANDBOX_DOMAIN for your sandbox service}"
: "${OPENSANDBOX_API_KEY:?Set OPENSANDBOX_API_KEY in your environment}"
if [[ "${recipe}" != */* ]]; then
  recipe="examples/hlgauss_ppo_sc/${recipe}"
fi
args=(--config "${recipe}" "policy.model_name=${MODEL_PATH}"
      "data.train.data_path=${TRAIN_DATA_PATH}")
if [[ -n "${CRITIC_CHECKPOINT:-}" ]]; then
  args+=("++ppo.warm_start_value_checkpoint=${CRITIC_CHECKPOINT}")
fi
# User overrides are last, including checkpointing.checkpoint_dir for full resume.
exec uv run examples/run_grpo_single_controller.py "${args[@]}" "$@"
