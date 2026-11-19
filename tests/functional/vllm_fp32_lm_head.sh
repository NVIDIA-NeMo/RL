#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory "$PROJECT_ROOT"

set -eou pipefail

EXP_NAME=$(basename "$0" .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf "$EXP_DIR"
mkdir -p "$EXP_DIR"

assert_grep() {
    local pattern=$1
    local file=$2
    grep -Eq "$pattern" "$file" || {
        echo "[FAIL] expected '$pattern' in $file"
        exit 1
    }
}

cd "$PROJECT_ROOT"
# A tied-embedding dense model and a hybrid Mamba model (the old Nemotron-H-only path).
for MODEL in Qwen/Qwen3-0.6B nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16; do
    uv run --extra vllm coverage run -a --data-file="$PROJECT_ROOT/tests/.coverage" --source="$PROJECT_ROOT/nemo_rl" \
        "$PROJECT_ROOT/tests/functional/vllm_fp32_lm_head.py" \
        --model "$MODEL" \
        "$@" \
        2>&1 | tee -a "$RUN_LOG"
done

assert_grep "\\[PASS\\] Qwen/Qwen3-0.6B fp32 lm_head" "$RUN_LOG"
assert_grep "\\[PASS\\] nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 fp32 lm_head" "$RUN_LOG"
