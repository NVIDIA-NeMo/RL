#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
# Run from an existing two-node Ray cluster with one available GPU per node.
# The same checkout and uv environments must be present on both nodes.
EXP_DIR=$(mktemp -d "$PROJECT_ROOT/tests/functional/xtoken-tq.XXXXXX")
RECIPE=examples/configs/recipes/llm/distillation-xtoken-qwen3-1.7b-to-llama3.2-1b-2n1g-dtensor2tp1-tq.yaml

# S1 + S2 frozen-payload check, with production student materialization/loss.
uv run --extra automodel python tests/functional/xtoken_tq_two_node.py

uv run python -m tools.x_token.minimal_projection_via_multitoken \
    --student-model meta-llama/Llama-3.2-1B \
    --teacher-model Qwen/Qwen3-1.7B \
    --top-k 4 --enable-special-token-mapping --enable-exact-match \
    --disable-reverse-pass --disable-scale-trick \
    --output-filename xtoken_tq_smoke --output-dir "$EXP_DIR/projection"
PROJ_PATH="$EXP_DIR/projection/xtoken_tq_smoke_special.pt"
test -f "$PROJ_PATH"

for steps in 3 10; do
    uv run python examples/run_xtoken_off_policy_distillation.py \
        --config "$RECIPE" \
        "teachers.0.projection_matrix_path=$PROJ_PATH" \
        "distillation.max_num_steps=$steps" \
        "logger.log_dir=$EXP_DIR/$steps" \
        "$@" 2>&1 | tee "$EXP_DIR/train-$steps.log"
    uv run python tests/functional/xtoken_tq_two_node.py \
        --check-log "$EXP_DIR/train-$steps.log" --expected-steps "$steps"
done
printf 'Acceptance artifacts: %s\n' "$EXP_DIR"
