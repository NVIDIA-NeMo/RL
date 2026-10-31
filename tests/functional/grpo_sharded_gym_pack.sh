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

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

NEMO_GYM_GRPO_CONFIG="$PROJECT_ROOT/examples/nemo_gym/grpo_sharded_gym_smoke.yaml" \
    bash "$SCRIPT_DIR/grpo_async_gym.sh" \
    +env.nemo_gym.placement_strategy=PACK \
    +policy.megatron_cfg.moe_per_layer_logging=false \
    policy.generation.vllm_cfg.http_server_serving_chat_kwargs.tool_parser=hermes \
    "$@"
