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

export EXP_NAME="$(basename "$0" .sh)"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ===== BEGIN CONFIG =====
NUM_NODES=2
STEPS_PER_RUN=40
MAX_STEPS=40
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=40
# ===== END CONFIG =====

# Reuse the same one-policy-epoch checks as the streaming arms.
exec bash "$SCRIPT_DIR/ppo-qwen2.5-1.5b-gsm8k-2n8g-megatron-valuetp2sp-dynbatch-noncolocated-async-single-controller-streaming32.sh" "$@"
