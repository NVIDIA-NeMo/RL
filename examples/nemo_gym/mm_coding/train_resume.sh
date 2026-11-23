#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# Run inside an existing GPU allocation with the documented actor environments.
set -euo pipefail
umask 077

: "${MM_CODING_BASE_CONFIG:?Provide an absolute path to the validated Megatron/vLLM base config}"
: "${MM_CODING_TRAIN_DATA:?Provide the training split}"
: "${MM_CODING_VALIDATION_DATA:?Provide the disjoint validation split}"
: "${MM_CODING_REFERENCE_ROOT:?Provide the original image root}"
: "${MM_CODING_ACCEPTANCE_ROOT:?Provide a new, empty output directory}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(realpath "${script_dir}/../../..")
cd "${repo_root}"
if [[ -e "${MM_CODING_ACCEPTANCE_ROOT}" ]]; then
    echo "Refusing to overwrite an earlier acceptance attempt" >&2
    exit 2
fi
mkdir -p "${MM_CODING_ACCEPTANCE_ROOT}"
# One total keeps Megatron train_iters and the saved scheduler state compatible.
total_steps=4
mode=${1:?Choose single, multi or mixed with matching adapted data}
shift
case "${mode}" in single|multi|mixed) ;; *) echo "Unknown mode: ${mode}" >&2; exit 2 ;; esac
if [[ "${1:-}" == -- ]]; then shift; fi
output="${MM_CODING_ACCEPTANCE_ROOT}/${mode}"
mkdir -p "${output}"
checkpoint_dir=${MM_CODING_CHECKPOINT_ROOT:-${output}/checkpoints}
if [[ -e "${checkpoint_dir}" || -L "${checkpoint_dir}" ]]; then
    echo "Refusing to reuse an earlier checkpoint directory: ${checkpoint_dir}" >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "${checkpoint_dir}")"
mkdir -- "${checkpoint_dir}"
recipe="examples/nemo_gym/mm_coding/mm_coding_${mode}.yaml"
overrides=(
    "checkpointing.checkpoint_dir=${checkpoint_dir}"
    checkpointing.enabled=true
    checkpointing.save_period=1
    checkpointing.save_optimizer=true
    checkpointing.save_data_plane=true
    checkpointing.load_replay_buffer=true
    # Keep small splits from exhausting before the fourth step.
    "grpo.max_num_epochs=${total_steps}"
    async_rl.log_full_train_data=true
    logger.wandb_enabled=false
    "$@"
)
save_overrides=(
    "${overrides[@]}"
    "grpo.max_num_steps=${total_steps}"
    checkpointing.checkpoint_must_save_by=00:00:00:01
    "logger.log_dir=${output}/save"
)
resume_overrides=(
    "${overrides[@]}"
    "grpo.max_num_steps=${total_steps}"
    checkpointing.checkpoint_must_save_by=null
    "logger.log_dir=${output}/resume"
)
# Phase settings can change interpolated paths; check both before training.
uv run --extra nemo_gym python examples/nemo_gym/mm_coding/check_mm_coding_runtime.py \
    --mode "${mode}" --config "${recipe}" "${save_overrides[@]}" \
    --expected-checkpoint-dir "${checkpoint_dir}" > "${output}/runtime-check.json"
uv run --extra nemo_gym python examples/nemo_gym/mm_coding/check_mm_coding_runtime.py \
    --mode "${mode}" --config "${recipe}" "${resume_overrides[@]}" \
    --expected-checkpoint-dir "${checkpoint_dir}" > "${output}/runtime-check-resume.json"
common=(examples/run_grpo_single_controller.py --config "${recipe}")
# These are separate interpreter/driver processes. Both phases use the same split and checkpoint root.
uv run --no-sync python "${common[@]}" "${save_overrides[@]}" \
    > "${output}/save.log" 2>&1
if ! grep -q "Timeout has been reached, stopping training early" "${output}/save.log"; then
    echo "Run 1 did not stop after its first saved step" >&2
    exit 1
fi
# Use the native inspector: Megatron embeds optimizer state in its weights checkpoint.
uv run --no-sync python - "${checkpoint_dir}/step_1" <<'CHECKPOINT'
import sys
from nemo_rl.utils.checkpoint import CheckpointManager
weights, optimizer = CheckpointManager.get_resume_paths(sys.argv[1])
assert weights is not None and weights.is_dir(), "Missing policy weights"
assert optimizer is not None, "Missing native optimizer state"
CHECKPOINT
uv run --no-sync python "${common[@]}" "${resume_overrides[@]}" \
    > "${output}/resume.log" 2>&1
if ! grep -q "Restoring dataloader state from checkpoint" "${output}/resume.log"; then
    echo "Run 2 did not restore the step_1 checkpoint" >&2
    exit 1
fi
uv run --no-sync python - "${checkpoint_dir}/step_${total_steps}" <<'CHECKPOINT'
import sys
from nemo_rl.utils.checkpoint import CheckpointManager
weights, optimizer = CheckpointManager.get_resume_paths(sys.argv[1])
assert weights is not None and weights.is_dir(), "Missing resumed policy weights"
assert optimizer is not None, "Missing resumed native optimizer state"
CHECKPOINT
echo "${mode}: save and resume processes exited successfully; inspect native restore, update and data-cursor evidence."
