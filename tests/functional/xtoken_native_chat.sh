#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
# Mark the current repo as safe, since wandb fetches metadata about the repo.
git config --global --add safe.directory "$PROJECT_ROOT"

EXP_NAME=$(basename "$0" .sh)
EXP_DIR="$SCRIPT_DIR/$EXP_NAME"
DATA_PATH="$PROJECT_ROOT/tests/functional/test_data/xtoken_native_chat.jsonl"
LOG_DIR="$EXP_DIR/logs"
PROJ_DIR="$EXP_DIR/projection"
PROJ_PATH="$PROJ_DIR/smol_qwen_special.pt"
JSON_METRICS="$EXP_DIR/metrics.json"
RUN_LOG="$EXP_DIR/run.log"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

rm -rf "$EXP_DIR"
mkdir -p "$LOG_DIR" "$PROJ_DIR"

cd "$PROJECT_ROOT"

# Step 1: build the projection for the distinct SmolLM2 student vocabulary.
# Qwen3-1.7B uses the same vocabulary as the pinned Qwen3-4B tokenizer in the
# chat config. SmolLM2 has its own, substantially smaller vocabulary.
uv run python -m tools.x_token.minimal_projection_via_multitoken \
    --student-model HuggingFaceTB/SmolLM2-135M-Instruct \
    --teacher-model Qwen/Qwen3-4B \
    --top-k 4 \
    --enable-special-token-mapping \
    --enable-exact-match \
    --disable-reverse-pass \
    --disable-scale-trick \
    --output-filename smol_qwen \
    --output-dir "$PROJ_DIR"

# Step 2: train for three steps on the checked-in multi-turn conversations.
# The ChatML history override preserves reasoning and tools on both sides,
# while different token boundaries exercise many-to-one and one-to-many alignment.
uv run coverage run -a --data-file="$PROJECT_ROOT/tests/.coverage" --source="$PROJECT_ROOT/nemo_rl" \
    examples/run_xtoken_off_policy_distillation.py \
    --config examples/configs/xtoken_native_chat.yaml \
    teachers.0.aligner.projection_matrix_path="$PROJ_PATH" \
    data.train.data_path="$DATA_PATH" \
    logger.log_dir="$LOG_DIR" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

# Step 3: verify that at least three training steps produced finite losses.
uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    'len(data["train/loss"]) >= 3' \
    'all_finite(data["train/loss"])'
