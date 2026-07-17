#!/bin/bash
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

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_training.sh TRAIN_DATA QUANT_CALIB_DATA OUTPUT_DIR [MODEL]

Arguments:
  TRAIN_DATA       NeMo-Gym JSONL training data on shared storage.
  QUANT_CALIB_DATA JSONL calibration data on shared storage.
  OUTPUT_DIR       Shared directory for checkpoints and logs.
  MODEL            Local model path or Hugging Face model ID.

Optional environment:
  NUM_NODES        Ray worker nodes (default: 8).
  GPUS_PER_NODE    GPUs per node (default: 4).
  MAX_NUM_STEPS    Training steps (default: 3000).
EOF
}

if (( $# < 3 || $# > 4 )); then
    usage >&2
    exit 2
fi

TRAIN_DATA=$1
QUANT_CALIB_DATA=$2
OUTPUT_DIR=$3
MODEL=${4:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}
NUM_NODES=${NUM_NODES:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MAX_NUM_STEPS=${MAX_NUM_STEPS:-3000}

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "Training data does not exist: $TRAIN_DATA" >&2
    exit 1
fi
if [[ ! -f "$QUANT_CALIB_DATA" ]]; then
    echo "Calibration data does not exist: $QUANT_CALIB_DATA" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

export DISABLE_MODELOPT_LAYER_SPEC=1
export ENABLE_BRIDGE_QUANT_MAPPING=1
export NRL_TQ_MOONCAKE_BATCH_SIZE_LIMIT=1
export NRL_TQ_MOONCAKE_MAX_WORKER_THREADS=1
export MC_TCP_ENABLE_CONNECTION_POOL=1

uv run --locked --extra modelopt --extra mcore --extra nemo-gym \
    python examples/nemo_gym/run_distillation_nemo_gym.py \
    --config examples/modelopt/qa_distillation_nano3_megatron_tq.yaml \
    "policy.model_name=$MODEL" \
    "policy.tokenizer.name=$MODEL" \
    "teacher.model_name=$MODEL" \
    "teacher.tokenizer.name=$MODEL" \
    "policy.quant_calib_data=$QUANT_CALIB_DATA" \
    "data.train.data_path=$TRAIN_DATA" \
    "distillation.max_num_steps=$MAX_NUM_STEPS" \
    "cluster.num_nodes=$NUM_NODES" \
    "cluster.gpus_per_node=$GPUS_PER_NODE" \
    "checkpointing.checkpoint_dir=$OUTPUT_DIR/checkpoints" \
    "logger.log_dir=$OUTPUT_DIR/logs" \
    "logger.tensorboard.log_dir=$OUTPUT_DIR/tensorboard"
