#!/usr/bin/env bash
set -euo pipefail

# PivotRL training for Nemotron Super Omni over demonstrated image-tool turns.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PIVOT_DATA_DIR="${PIVOT_DATA_DIR:-${CODE_DIR}/3rdparty/Gym-workspace/Gym/resources_servers/image_tools/data}"
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
  "${CODE_DIR}/3rdparty/Gym-workspace/Gym/resources_servers/image_tools/configs/image_tools_pivot.yaml"

export EXP_NAME="${EXP_NAME:-grpo-super-omni-image-tools-pivot}"
export CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-image-tools-pivot.yaml}"
export ENTRYPOINT="${ENTRYPOINT:-examples/nemo_gym/run_grpo_nemo_gym.py}"
export TRAIN_PATH
export VAL_PATH

# Every path below is a placeholder; export the real values (see
# super_omni_launch.sh, which this script wraps and which rejects
# unset placeholders before submitting).
export MODEL_PATH="${MODEL_PATH:-/path/to/nemotron-super-omni-sft-checkpoint}"
export CONTAINER="${CONTAINER:-/path/to/nemo-rl.sqsh}"
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/path/to/nemo-skills-sandbox.sqsh}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/path/to/cache/rl-v2-image-tools}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-your_slurm_account}"
export SLURM_PARTITION="${SLURM_PARTITION:-batch_long,batch}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# The JSONL files and their referenced media must exist at the same absolute
# paths inside the container. Keep the mounts narrow and overridable.
PIVOT_DATA_MOUNT="${PIVOT_DATA_DIR}:${PIVOT_DATA_DIR}"
PIVOT_MEDIA_ROOT="${PIVOT_MEDIA_ROOT:-/path/to/tool_calls_gen}"
PIVOT_MEDIA_MOUNT="${PIVOT_MEDIA_ROOT}:${PIVOT_MEDIA_ROOT}"
MODEL_MOUNT="${MODEL_PATH}:${MODEL_PATH}"
# A chat template kept outside the checkpoint (e.g. a fixed template under review)
# has to be mounted too; super_omni_launch.sh verifies this and aborts otherwise.
CHAT_TEMPLATE_MOUNT=""
if [[ -n "${CHAT_TEMPLATE:-}" && "${CHAT_TEMPLATE}" != "${MODEL_PATH}"* ]]; then
  CHAT_TEMPLATE_DIR="$(cd "$(dirname "${CHAT_TEMPLATE}")" && pwd)"
  CHAT_TEMPLATE_MOUNT=",${CHAT_TEMPLATE_DIR}:${CHAT_TEMPLATE_DIR}"
fi
CACHE_MOUNT="${PERSISTENT_CACHE}:${PERSISTENT_CACHE}"
export EXTRA_MOUNTS="${PIVOT_DATA_MOUNT},${PIVOT_MEDIA_MOUNT},${MODEL_MOUNT},${CACHE_MOUNT}${CHAT_TEMPLATE_MOUNT}${EXTRA_MOUNTS:+,${EXTRA_MOUNTS}}"

exec "${SCRIPT_DIR}/super_omni_launch.sh" "$@"
