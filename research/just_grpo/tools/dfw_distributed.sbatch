#!/usr/bin/env bash
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
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
set -euo pipefail
: "${REPO_DIR:?Set the upstream-main worktree}"
: "${POLICY_PYTHON:?Set the policy interpreter}"
: "${MODEL:?Set the Nemotron checkpoint}"
: "${OUTPUT_DIR:?Set a fresh output directory}"
: "${CONFIG:?Set the Megatron recipe}"
: "${CONTAINER_IMAGE:?Set the container image}"
: "${CUDA_COMPAT_DIR:?Set the CUDA compatibility library directory}"
export EXPANDABLE_SEGMENTS=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 RAYON_NUM_THREADS=1
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/research/just_grpo${POLICY_EXTRA_PYTHONPATH:+:${POLICY_EXTRA_PYTHONPATH}}"
export XDG_CACHE_HOME="${OUTPUT_DIR}/cache" UV_CACHE_DIR="${OUTPUT_DIR}/uv_cache"
export VLLM_CACHE_ROOT="${OUTPUT_DIR}/vllm_cache" VLLM_CONFIG_ROOT="${OUTPUT_DIR}/vllm_config"
export VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
export MEGATRON_CONFIG_LOCK_DIR="${OUTPUT_DIR}/megatron_config_locks"
export HF_HOME="${OUTPUT_DIR}/hf_cache" HF_MODULES_CACHE="${OUTPUT_DIR}/hf_modules"
export MPLCONFIGDIR="${OUTPUT_DIR}/mpl"
# Compilation performs many small atomic renames; keep them off shared Lustre.
export TORCHINDUCTOR_CACHE_DIR="/tmp/just_grpo_${SLURM_JOB_ID}/inductor"
export TRITON_CACHE_DIR="/tmp/just_grpo_${SLURM_JOB_ID}/triton"
export WANDB_MODE=online WANDB_DIR="${OUTPUT_DIR}/wandb"
export WANDB_CACHE_DIR="${OUTPUT_DIR}/wandb_cache" WANDB_CONFIG_DIR="${OUTPUT_DIR}/wandb_config"
if [[ -z "${WANDB_API_KEY:-}" && -f /home/snorouzi/wandb_api_key.txt ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\n\r' < /home/snorouzi/wandb_api_key.txt)"
fi
# Native policy workers use the provisioned interpreter in this container.
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}"
if [[ "${SLURM_NNODES}" == 1 ]]; then
  srun --kill-on-bad-exit=1 \
    --output="${OUTPUT_DIR}/node-%t.log" --error="${OUTPUT_DIR}/node-%t.log" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/home/snorouzi:/home/snorouzi,/lustre:/lustre \
    --container-workdir="${REPO_DIR}" \
    bash research/just_grpo/tools/driver.sh
else
  export CONTAINER="${CONTAINER_IMAGE}"
  # ray.sub invokes the container's bare `ray`; bind the provisioned CLI
  # (whose shebang selects POLICY_PYTHON) over the NeMo container default.
  ray_cli="${POLICY_PYTHON%/*}/ray"
  test -x "${ray_cli}"
  export MOUNTS="/home/snorouzi:/home/snorouzi,/lustre:/lustre,${ray_cli}:/opt/nemo_rl_venv/bin/ray"
  export CONTAINER_CWD="${REPO_DIR}"
  export BASE_LOG_DIR="${OUTPUT_DIR}/ray"
  export GPUS_PER_NODE="${SLURM_GPUS_ON_NODE}" CPUS_PER_WORKER="${SLURM_CPUS_PER_TASK}"
  printf -v COMMAND 'cd %q && bash research/just_grpo/tools/driver.sh' "${REPO_DIR}"
  export COMMAND
  bash "${REPO_DIR}/ray.sub"
fi
