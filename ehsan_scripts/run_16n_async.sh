#!/bin/bash

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

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
readonly SBATCH_PARTITION="batch_block1,backfill"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is unset: ${name}" >&2
    exit 2
  fi
}

for name in \
  CONTAINER \
  MOUNTS \
  NEMO_RL_CHAT_TEMPLATE \
  NEMO_RL_MODEL \
  NEMO_RL_VIDEO_TRAIN_JSONL \
  NEMO_RL_VIDEO_VAL_JSONL \
  NEMO_RL_VIDEO_MEDIA_ROOT \
  NEMO_RL_RUN_ROOT \
  SBATCH_ACCOUNT \
  WANDB_API_KEY \
  WANDB_ENTITY \
  WANDB_PROJECT; do
  require_env "${name}"
done

case "${SBATCH_ACCOUNT}" in
  nemotron_edge_omni | nemotron_omni_vision) ;;
  *)
    echo "SBATCH_ACCOUNT must be nemotron_edge_omni or nemotron_omni_vision" >&2
    exit 2
    ;;
esac

mkdir -p "${NEMO_RL_RUN_ROOT}"

NEMO_RL_RUN_ID="${NEMO_RL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
NEMO_RL_RUN_DATE="${NEMO_RL_RUN_DATE:-$(date -u +%m%d)}"

export BASE_LOG_DIR="${NEMO_RL_RUN_ROOT}"
export CONTAINER
export GPUS_PER_NODE=8
export MOUNTS
export NEMO_RL_CHAT_TEMPLATE
export NEMO_RL_EXPECTED_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
export NEMO_RL_EXPECTED_GYM_COMMIT="$(
  git -C "${REPO_ROOT}" ls-tree HEAD 3rdparty/Gym-workspace/Gym | awk '{print $3}'
)"
export NEMO_RL_EXPECTED_MBRIDGE_COMMIT="$(
  git -C "${REPO_ROOT}" ls-tree HEAD 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge | awk '{print $3}'
)"
export NEMO_RL_MODEL
export NEMO_RL_PYTHON_VERSION="$(tr -d '[:space:]' < "${REPO_ROOT}/.python-version")"
export NEMO_RL_REPO="${REPO_ROOT}"
export NEMO_RL_RUN_ID
export NEMO_RL_RUN_DATE
export NEMO_RL_RUN_ROOT
export NEMO_RL_VIDEO_MEDIA_ROOT
export NEMO_RL_VIDEO_TRAIN_JSONL
export NEMO_RL_VIDEO_VAL_JSONL
export NRL_RUN_PREFIX="rl_main_vg16_async_video_nemotron_omni_model_tp4_unlimited"
export RAY_TMPDIR=/tmp/ray
export NEMO_RL_UV_VERSION="$(awk -F= '/^ARG UV_VERSION=/{print $2; exit}' "${REPO_ROOT}/docker/Dockerfile")"
export NEMO_RL_UV_DIR="${NEMO_RL_RUN_ROOT}/uv/${NEMO_RL_UV_VERSION}"
export NEMO_RL_MAIN_VENV="/tmp/nemorl-main-${NEMO_RL_EXPECTED_COMMIT:0:12}"
export NEMO_RL_UV_CACHE_DIR="/tmp/nemorl-uv-cache-${NEMO_RL_EXPECTED_COMMIT:0:12}"
export NEMO_RL_RAY_WRAPPER_DIR="${NEMO_RL_RUN_ROOT}/runtime/${NEMO_RL_EXPECTED_COMMIT:0:12}/bin"
export UV_PYTHON_INSTALL_DIR="${NEMO_RL_RUN_ROOT}/uv_python/${NEMO_RL_PYTHON_VERSION}"
export WANDB_API_KEY
export WANDB_ENTITY
export WANDB_PROJECT
mkdir -p "${NEMO_RL_RAY_WRAPPER_DIR}"
readonly RAY_WRAPPER="${NEMO_RL_RAY_WRAPPER_DIR}/ray"
readonly RAY_WRAPPER_TMP="${RAY_WRAPPER}.tmp.$$"
printf '%s\n' \
  '#!/bin/bash' \
  'set -euo pipefail' \
  'if [[ -x "${NEMO_RL_MAIN_VENV}/bin/ray" ]]; then' \
  '  exec "${NEMO_RL_MAIN_VENV}/bin/ray" "$@"' \
  'fi' \
  'exec /opt/nemo_rl_venv/bin/ray.nemorl-base "$@"' \
  > "${RAY_WRAPPER_TMP}"
chmod 755 "${RAY_WRAPPER_TMP}"
mv -f "${RAY_WRAPPER_TMP}" "${RAY_WRAPPER}"
export PATH="${NEMO_RL_RAY_WRAPPER_DIR}:${NEMO_RL_MAIN_VENV}/bin:${NEMO_RL_UV_DIR}:${PATH}"
export VIRTUAL_ENV="${NEMO_RL_MAIN_VENV}"
export UV_PROJECT_ENVIRONMENT="${NEMO_RL_MAIN_VENV}"

read -r -d '' SETUP_COMMAND <<'SETUP_EOF' || true
set -euo pipefail

source /opt/rl_main_vg_runtime.env
mkdir -p "${NEMO_RL_UV_DIR}"
flock "${NEMO_RL_UV_DIR}.lock" bash -c '
  set -euo pipefail
  if [[ ! -x "${NEMO_RL_UV_DIR}/uv" ]]; then
    curl --retry 3 --retry-delay 2 -LsSf \
      "https://astral.sh/uv/${NEMO_RL_UV_VERSION}/install.sh" | \
      env UV_INSTALL_DIR="${NEMO_RL_UV_DIR}" UV_NO_MODIFY_PATH=1 sh
  fi
'
export PATH="${NEMO_RL_UV_DIR}:/root/.local/bin:/opt/nemo_rl_venv/bin:${PATH}"
uv --version
mkdir -p "${UV_PYTHON_INSTALL_DIR}" "${NEMO_RL_UV_CACHE_DIR}"
flock "${UV_PYTHON_INSTALL_DIR}.lock" \
  uv python install "${NEMO_RL_PYTHON_VERSION}"
uv python find "${NEMO_RL_PYTHON_VERSION}"
export VIRTUAL_ENV="${NEMO_RL_MAIN_VENV}"
export UV_PROJECT_ENVIRONMENT="${NEMO_RL_MAIN_VENV}"
UV_CACHE_DIR="${NEMO_RL_UV_CACHE_DIR}" uv sync \
  --directory "${NEMO_RL_REPO}" \
  --locked \
  --no-install-project
readonly CONTAINER_RAY=/opt/nemo_rl_venv/bin/ray
readonly CONTAINER_RAY_BASE=/opt/nemo_rl_venv/bin/ray.nemorl-base
if [[ ! -x "${CONTAINER_RAY_BASE}" ]]; then
  cp -p "${CONTAINER_RAY}" "${CONTAINER_RAY_BASE}"
fi
install -m 755 \
  "${NEMO_RL_RAY_WRAPPER_DIR}/ray" \
  "${CONTAINER_RAY}.tmp.$$"
mv -f "${CONTAINER_RAY}.tmp.$$" "${CONTAINER_RAY}"
export PATH="${NEMO_RL_MAIN_VENV}/bin:${NEMO_RL_UV_DIR}:/root/.local/bin:${PATH}"
python -c \
  'import sys; assert sys.version_info[:3] == tuple(map(int, sys.argv[1].split("."))), sys.version' \
  "${NEMO_RL_PYTHON_VERSION}"
ray --version
readonly VLLM_OVERLAY="/tmp/nemorl-stock-vllm-0.25.1-${NEMO_RL_EXPECTED_COMMIT:0:12}"
readonly VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1-cp38-abi3-manylinux_2_28_x86_64.whl"
mkdir -p "${VLLM_OVERLAY}"
UV_CACHE_DIR="${NEMO_RL_UV_CACHE_DIR}" uv pip install \
  --target "${VLLM_OVERLAY}" \
  --reinstall \
  --no-deps \
  "vllm @ ${VLLM_WHEEL}"
SETUP_EOF
export SETUP_COMMAND

read -r -d '' COMMAND <<'COMMAND_EOF' || true
set -euo pipefail

source /opt/rl_main_vg_runtime.env
export NEMO_GYM_VENV_DIR="/tmp/nemorl-gym-${NEMO_RL_EXPECTED_COMMIT:0:12}-${NEMO_RL_PYTHON_VERSION}-${NEMO_RL_RUN_ID}"
export PATH="${NEMO_RL_MAIN_VENV}/bin:${NEMO_RL_UV_DIR}:/root/.local/bin:${PATH}"
export VIRTUAL_ENV="${NEMO_RL_MAIN_VENV}"
export UV_PROJECT_ENVIRONMENT="${NEMO_RL_MAIN_VENV}"
export HF_HOME="${NEMO_RL_RUN_ROOT}/hf_home"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export NRL_MEGATRON_CHECKPOINT_DIR="${HF_HOME}/nemo_rl-${NEMO_RL_EXPECTED_MBRIDGE_COMMIT:0:12}"
export NRL_FORCE_REBUILD_VENVS=true
export NRL_VIDEO_BACKEND=torchcodec
export NRL_VIDEO_SAMPLING_STYLE=nemotron_vl
export NRL_VIDEO_SFT_MAX_FRAMES=32
export NRL_VIDEO_SFT_MIN_FRAMES=32
export NRL_VIDEO_TEMPORAL_PATCH_SIZE=2
export TORCH_CUDA_ARCH_LIST=9.0
export VLLM_MAMBA_BACKEND=flashinfer
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="PYTHONPATH,NEMO_RL_VIDEO_MEDIA_ROOT,NRL_VIDEO_BACKEND,NRL_VIDEO_SAMPLING_STYLE,NRL_VIDEO_TEMPORAL_PATCH_SIZE"
export VLLM_VIDEO_LOADER_BACKEND=nemotron_vl
unset NRL_IGNORE_VERSION_MISMATCH

readonly VLLM_OVERLAY="/tmp/nemorl-stock-vllm-0.25.1-${NEMO_RL_EXPECTED_COMMIT:0:12}"
export NEMO_GYM_EXTRA_ROOTS="${NEMO_RL_REPO}/3rdparty/Gym-workspace/Gym"
export PYTHONPATH="${VLLM_OVERLAY}:${NEMO_RL_REPO}:${NEMO_GYM_EXTRA_ROOTS}:${NEMO_RL_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMO_RL_REPO}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${HF_MODULES_CACHE}${PYTHONPATH:+:${PYTHONPATH}}"

readonly RUN_NAME="${NRL_RUN_PREFIX}_${NEMO_RL_RUN_DATE}_${NEMO_RL_RUN_ID}"
readonly CHECKPOINT_DIR="${NEMO_RL_RUN_ROOT}/checkpoints/${RUN_NAME}"
mkdir -p "${CHECKPOINT_DIR}" "${HF_MODULES_CACHE}" "${NRL_MEGATRON_CHECKPOINT_DIR}"
cd "${NEMO_RL_REPO}"
test "$(git rev-parse HEAD)" = "${NEMO_RL_EXPECTED_COMMIT}"
test "$(git -C 3rdparty/Gym-workspace/Gym rev-parse HEAD)" = "${NEMO_RL_EXPECTED_GYM_COMMIT}"
test "$(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge rev-parse HEAD)" = "${NEMO_RL_EXPECTED_MBRIDGE_COMMIT}"
python -c \
  'import sys; assert sys.version_info[:3] == tuple(map(int, sys.argv[1].split("."))), sys.version' \
  "${NEMO_RL_PYTHON_VERSION}"

python \
  examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_nemotron_omni_30ba3b_video_async.yaml \
  policy.model_name="${NEMO_RL_MODEL}" \
  policy.tokenizer.name="${NEMO_RL_MODEL}" \
  policy.tokenizer.chat_template="${NEMO_RL_CHAT_TEMPLATE}" \
  policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template="${NEMO_RL_CHAT_TEMPLATE}" \
  grpo.max_num_steps=-1 \
  grpo.max_num_epochs=1000000 \
  grpo.seq_logprob_error_threshold=null \
  grpo.num_prompts_per_step=4 \
  grpo.num_generations_per_prompt=16 \
  grpo.async_grpo.enabled=true \
  grpo.async_grpo.max_trajectory_age_steps=1 \
  grpo.async_grpo.in_flight_weight_updates=true \
  policy.train_global_batch_size=64 \
  policy.max_total_sequence_length=32768 \
  policy.megatron_cfg.tensor_model_parallel_size=4 \
  policy.megatron_cfg.expert_model_parallel_size=4 \
  policy.megatron_cfg.moe_shared_expert_overlap=false \
  policy.megatron_cfg.optimizer.optimizer_cpu_offload=true \
  policy.megatron_cfg.optimizer.optimizer_offload_fraction=1.0 \
  policy.generation.max_new_tokens=16000 \
  policy.generation.vllm_cfg.tensor_parallel_size=4 \
  policy.generation.vllm_cfg.max_model_len=32768 \
  policy.generation.vllm_kwargs.max_num_batched_tokens=32768 \
  policy.generation.vllm_kwargs.max_num_seqs=1 \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=14 \
  cluster.num_nodes=16 \
  cluster.gpus_per_node=8 \
  checkpointing.enabled=true \
  checkpointing.checkpoint_dir="${CHECKPOINT_DIR}" \
  logger.log_dir="${NEMO_RL_RUN_ROOT}/${RUN_NAME}/training" \
  logger.wandb_enabled=true \
  logger.wandb.project="${WANDB_PROJECT}" \
  +logger.wandb.entity="${WANDB_ENTITY}" \
  logger.wandb.name="${RUN_NAME}"
COMMAND_EOF
export COMMAND

cd "${REPO_ROOT}"
sbatch \
  --account="${SBATCH_ACCOUNT}" \
  --partition="${SBATCH_PARTITION}" \
  --nodes=16 \
  --ntasks=16 \
  --ntasks-per-node=1 \
  --gpus-per-node=8 \
  --time="${SBATCH_TIME:-04:00:00}" \
  --job-name="rl_main_vg16_async_omni_video_${SBATCH_ACCOUNT}" \
  --output="${NEMO_RL_RUN_ROOT}/slurm-16n-async-%j.out" \
  ray.sub
