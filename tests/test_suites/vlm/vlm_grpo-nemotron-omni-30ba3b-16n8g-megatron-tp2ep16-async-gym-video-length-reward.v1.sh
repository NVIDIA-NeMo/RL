#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ===== BEGIN CONFIG =====
NUM_NODES=16
GPUS_PER_NODE=8
STEPS_PER_RUN=1
MAX_STEPS=1
NUM_RUNS=1
NUM_MINUTES=240
# ===== END CONFIG =====

# Reuse the video regression setup while resolving this script's production
# TP2/EP16 length-reward recipe by basename.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp4ep4-async-gym-video.v1.sh"
