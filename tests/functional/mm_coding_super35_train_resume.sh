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

# Run inside an approved GPU allocation with the documented actor environments.
set -euo pipefail
umask 077

: "${MM_CODING_TRAIN_DATA:?Provide the approved training split}"
: "${MM_CODING_VALIDATION_DATA:?Provide the disjoint validation split}"
: "${MM_CODING_REFERENCE_ROOT:?Provide the original image root}"
: "${MM_CODING_ACCEPTANCE_ROOT:?Provide a new, empty output directory}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(realpath "${script_dir}/../..")
cd "${repo_root}"
if [[ -e "${MM_CODING_ACCEPTANCE_ROOT}" ]]; then
    echo "Refusing to overwrite an earlier acceptance attempt" >&2
    exit 2
fi
mkdir -p "${MM_CODING_ACCEPTANCE_ROOT}"
save_steps=${MM_CODING_SAVE_STEPS:-3}
if [[ ! "${save_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MM_CODING_SAVE_STEPS must be a positive integer" >&2
    exit 2
fi
resume_steps=$((save_steps + 1))
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
uv run --no-sync python examples/nemo_gym/nemotron-3.5-super-vl/check_mm_coding_runtime.py \
    --mode "${mode}" > "${output}/runtime-check.json"
common=(
    examples/run_grpo_single_controller.py
    --config "examples/nemo_gym/nemotron-3.5-super-vl/mm_coding_${mode}.yaml"
    "checkpointing.checkpoint_dir=${checkpoint_dir}"
    checkpointing.enabled=true
    checkpointing.save_period=1
    checkpointing.save_optimizer=true
    checkpointing.save_data_plane=true
    checkpointing.load_replay_buffer=true
    async_rl.log_full_train_data=true
    logger.wandb_enabled=false
    "$@"
)
# These are separate interpreter/driver processes. Both phases use the same split and checkpoint root.
uv run --no-sync python "${common[@]}" \
    "grpo.max_num_steps=${save_steps}" "logger.log_dir=${output}/save" > "${output}/save.log" 2>&1
test -d "${checkpoint_dir}/step_${save_steps}/policy/weights"
test -d "${checkpoint_dir}/step_${save_steps}/policy/optimizer"
uv run --no-sync python "${common[@]}" \
    "grpo.max_num_steps=${resume_steps}" "logger.log_dir=${output}/resume" > "${output}/resume.log" 2>&1
test -d "${checkpoint_dir}/step_${resume_steps}/policy/optimizer"
echo "${mode}: save and resume processes exited successfully; inspect native restore, update and data-cursor evidence."
