#!/usr/bin/env bash
set -euo pipefail

# PivotRL training for Nemotron Super Omni over demonstrated image-tool turns.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIVOT_DATA_DIR="${PIVOT_DATA_DIR:-${CODE_DIR}/3rdparty/Gym-workspace/Gym/resources_servers/image_tools_pivot/data}"
TRAIN_PATH="${TRAIN_PATH:-${PIVOT_DATA_DIR}/train.jsonl}"
VAL_PATH="${VAL_PATH:-${PIVOT_DATA_DIR}/validation.jsonl}"

for required in "${TRAIN_PATH}" "${VAL_PATH}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Error: missing or empty PivotRL dataset: ${required}" >&2
    echo "Generate it with tools/convert_sft_to_image_pivot.py or stage the donor dataset." >&2
    exit 1
  fi
done

python3 "${CODE_DIR}/tools/preflight_pivot_config.py" \
  "${CODE_DIR}/3rdparty/Gym-workspace/Gym/resources_servers/image_tools_pivot/configs/image_tools_pivot.yaml"

export EXP_NAME="${EXP_NAME:-grpo-super-omni-image-tools-pivot}"
export CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-image-tools-pivot.yaml}"
export ENTRYPOINT="${ENTRYPOINT:-examples/nemo_gym/run_grpo_nemo_gym.py}"
export TRAIN_PATH
export VAL_PATH

export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/smohsenitahe/workspace/output/sft_super_omni_toolcalls_32k_rep2_0719/checkpoints/tp_1_hf/iter_0000276/mcore_to_hf}"
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/smohsenitahe/images/nemo-rl-main-20260807-super-ultra-omni-prefetched.sqsh}"
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-latest.sqsh}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/llmservice/users/smohsenitahe/cache/rl-v2-image-tools}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_omni_vision}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch_long,batch}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# The JSONL files and their referenced media must exist at the same absolute
# paths inside the container. Keep the mounts narrow and overridable.
PIVOT_DATA_MOUNT="${PIVOT_DATA_DIR}:${PIVOT_DATA_DIR}"
PIVOT_MEDIA_ROOT="${PIVOT_MEDIA_ROOT:-/scratch/fsw/portfolios/llmservice/projects/llmservice_fm_vision/users/smohsenitahe/tool_calls_gen}"
PIVOT_MEDIA_MOUNT="${PIVOT_MEDIA_ROOT}:${PIVOT_MEDIA_ROOT}"
MODEL_MOUNT="${MODEL_PATH}:${MODEL_PATH}"
CACHE_MOUNT="${PERSISTENT_CACHE}:${PERSISTENT_CACHE}"
export EXTRA_MOUNTS="${PIVOT_DATA_MOUNT},${PIVOT_MEDIA_MOUNT},${MODEL_MOUNT},${CACHE_MOUNT}${EXTRA_MOUNTS:+,${EXTRA_MOUNTS}}"

exec "${SCRIPT_DIR}/super_omni_launch.sh" "$@"
