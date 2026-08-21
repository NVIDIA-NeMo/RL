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

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)

cd ${PROJECT_ROOT}

# L2 Megatron generation tests: nano-3.5 (30B-A3B mamba-hybrid MoE) async
# non-colocated generation with the Megatron inference backend, exercising extra
# features on top of the base non-colocated path. These are CLUSTER tests: they
# must run from /opt/nemo-rl on the HEAD of a 2-node `cog submit --launcher ray`
# allocation on GB200 / oci-hsg (1 gen node + 1 train node, 4 GPUs each -> EP=4).
# The model is pulled from the public HF release, so the nodes need hub access and
# a shared HF/Megatron checkpoint cache. See
# .claude/skills/run-nano35-megatron-inference-cog/SKILL.md.

# run_test [fast] <command...>
# - "run_test fast <cmd>" = always runs (both fast and full modes)
# - "run_test <cmd>"      = only runs in full mode; skipped when FAST=1
run_test() {
    if [[ "$1" == "fast" ]]; then
        shift
        time "$@"
    elif [[ "${FAST:-0}" == "1" ]]; then
        echo "FAST: Skipping: $*"
    else
        time "$@"
    fi
}


run_test      bash ./tests/functional/grpo_megatron_generation_async_prefix_caching.sh
run_test      bash ./tests/functional/grpo_megatron_generation_async_mxfp8.sh


cd ${PROJECT_ROOT}/tests
if compgen -G ".coverage*" > /dev/null; then
    coverage combine .coverage*
fi
