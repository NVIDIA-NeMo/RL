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
  submit_slurm.sh CONTAINER ACCOUNT PARTITION SHARED_ROOT MODEL \
      TRAIN_DATA QUANT_CALIB_DATA OUTPUT_DIR [NUM_NODES] [GPUS_PER_NODE]

CONTAINER may be an Enroot .sqsh file or a registry image accepted by Pyxis.
All local paths must be visible from every allocated node.
EOF
}

if (( $# < 8 || $# > 10 )); then
    usage >&2
    exit 2
fi

CONTAINER_IMAGE=$1
SLURM_ACCOUNT=$2
SLURM_PARTITION=$3
SHARED_ROOT=$4
MODEL=$5
TRAIN_DATA=$6
QUANT_CALIB_DATA=$7
OUTPUT_DIR=$8
NUM_NODES=${9:-8}
GPUS_PER_NODE=${10:-4}

REPO_ROOT=$(git rev-parse --show-toplevel)
mkdir -p "$OUTPUT_DIR"

printf -v TRAIN_COMMAND '%q ' \
    bash examples/modelopt/nano3_qaopd_tq/run_training.sh \
    "$TRAIN_DATA" "$QUANT_CALIB_DATA" "$OUTPUT_DIR" "$MODEL"
COMMAND="cd /opt/nemo-rl && $TRAIN_COMMAND"
MOUNTS="$SHARED_ROOT:$SHARED_ROOT,$REPO_ROOT:/opt/nemo-rl"

CONTAINER="$CONTAINER_IMAGE" \
MOUNTS="$MOUNTS" \
COMMAND="$COMMAND" \
BASE_LOG_DIR="$OUTPUT_DIR/ray" \
GPUS_PER_NODE="$GPUS_PER_NODE" \
sbatch \
    --nodes="$NUM_NODES" \
    --account="$SLURM_ACCOUNT" \
    --partition="$SLURM_PARTITION" \
    --time=04:00:00 \
    --exclusive \
    --dependency=singleton \
    --job-name=nano3-qaopd-tq \
    ray.sub
