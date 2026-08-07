#!/usr/bin/env bash
set -euo pipefail

# Run the public Super Omni image-MOPD recipe through the reproducible Super
# launcher. Required environment variables are validated by
# super_omni_launch.sh:
#
#   MODEL_PATH=/path/to/super_omni_hf \
#   TRAIN_PATH=/path/to/circle_count_train.jsonl \
#   CONTAINER=/path/to/nemo-rl.sqsh \
#   PERSISTENT_CACHE=/path/to/cache \
#   SLURM_ACCOUNT=account \
#   WANDB_API_KEY=... \
#   ./examples/nemo_gym/nemotron-3-super-omni/run_mopd_circle_count.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_NAME="${EXP_NAME:-mopd-super-omni-circle-count}"
export CONFIG_PATH="${CONFIG_PATH:-examples/configs/recipes/vlm/mopd-nemotron-super-omni-120ba12b-10n8g-megatron-tp8ep16cp2-async-gym.v1.yaml}"
export ENTRYPOINT="${ENTRYPOINT:-examples/nemo_gym/run_grpo_nemo_gym.py}"
export WANDB_PROJ="${WANDB_PROJ:-mopd-nemotron-super-omni}"

exec "${SCRIPT_DIR}/super_omni_launch.sh"
