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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
# ===== BEGIN CONFIG =====
# Keep the ordinary rollout snapshots, checkpoints and metrics separate.
PERF_RUN_SUFFIX=-eagle3-k1
NUM_NODES=32
GPUS_PER_NODE=4
SEGMENT_SIZE=16     # nodes per NVLink-domain segment; tools/launch passes it as sbatch --segment. Matches cluster.segment_size in the yaml.
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=100
# ===== END CONFIG =====

source "$SCRIPT_DIR/common.env"
# disable NVLS to avoid OOM issue
export NCCL_NVLS_ENABLE=0

run_grpo_if_needed "$@"
check_completed_grpo_run
