#!/usr/bin/env bash
set -euo pipefail

# One-step end-to-end GRPO smoke run for the image-tools Gym agent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SAMPLE_DIR="${SAMPLE_DIR:-${CODE_DIR}/.tmp/image_tools_grpo_sample}"
python3 "${CODE_DIR}/tools/make_image_tools_grpo_sample.py" \
  --output-dir "${SAMPLE_DIR}" \
  --train-repeats "${SAMPLE_TRAIN_REPEATS:-16}" \
  --validation-repeats "${SAMPLE_VALIDATION_REPEATS:-1}"

export EXP_NAME="${EXP_NAME:-grpo-super-omni-image-tools-sample}"
export CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-image-tools-sample.yaml}"
export ENTRYPOINT="${ENTRYPOINT:-examples/nemo_gym/run_grpo_nemo_gym.py}"
export TRAIN_PATH="${TRAIN_PATH:-${SAMPLE_DIR}/train.jsonl}"
export VAL_PATH="${VAL_PATH:-${SAMPLE_DIR}/validation.jsonl}"

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

SAMPLE_MOUNT="${SAMPLE_DIR}:${SAMPLE_DIR}"
MODEL_MOUNT="${MODEL_PATH}:${MODEL_PATH}"
CACHE_MOUNT="${PERSISTENT_CACHE}:${PERSISTENT_CACHE}"
export EXTRA_MOUNTS="${SAMPLE_MOUNT},${MODEL_MOUNT},${CACHE_MOUNT}${EXTRA_MOUNTS:+,${EXTRA_MOUNTS}}"

exec "${SCRIPT_DIR}/super_omni_launch.sh" "$@"
