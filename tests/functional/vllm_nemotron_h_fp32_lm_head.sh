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
uv run --extra vllm coverage run -a --data-file="$PROJECT_ROOT/tests/.coverage" --source="$PROJECT_ROOT/nemo_rl" \
    "$PROJECT_ROOT/tests/functional/vllm_nemotron_h_fp32_lm_head.py" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

assert_grep "Resolved architecture: NemotronHForCausalLM" "$RUN_LOG"
assert_grep "\\[fp32_lm_head\\] NemotronH vLLM lm_head.forward casts input and weight to fp32" "$RUN_LOG"
assert_grep "\\[PASS\\] Nemotron-H fp32 lm_head generated text" "$RUN_LOG"
