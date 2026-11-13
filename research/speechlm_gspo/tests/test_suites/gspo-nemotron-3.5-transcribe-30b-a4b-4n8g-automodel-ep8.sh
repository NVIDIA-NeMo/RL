#!/bin/bash
set -euo pipefail

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=8
STEPS_PER_RUN=22
MAX_STEPS=22
NUM_RUNS=1
NUM_MINUTES=240
# ===== END CONFIG =====

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
CONFIG_PATH="$PROJECT_ROOT/configs/recipes/$(basename "$0" .sh).yaml"
test -f "$CONFIG_PATH"
if [[ -n "${TEST_DRYRUN:-}" ]]; then
  exit 0
fi

: "${SPEECHLM_GSPO_MODEL:?Set the native Speech checkpoint (HF ID or directory)}"
: "${SPEECHLM_GSPO_DATA:?Set the ASR consumer Parquet path}"
: "${SPEECHLM_GSPO_OUTPUT_DIR:?Set a dedicated writable output root}"
mkdir -p "$SPEECHLM_GSPO_OUTPUT_DIR"

cd "$PROJECT_ROOT"
uv run --locked run_speechlm_gspo.py --config "$CONFIG_PATH" grpo.max_num_steps=20 "$@"
# Require a complete save, including optimizer state, before restarting Python.
uv run --locked tools/verify_resume.py --output-dir "$SPEECHLM_GSPO_OUTPUT_DIR" --initial-only

# The checkpoint manager restores the latest save. Keep the second log so the
# check can prove exactly two resumed updates.
uv run --locked run_speechlm_gspo.py --config "$CONFIG_PATH" grpo.max_num_steps=22 "$@" \
  2>&1 | tee "$SPEECHLM_GSPO_OUTPUT_DIR/resume2.log"
uv run --locked tools/verify_resume.py --output-dir "$SPEECHLM_GSPO_OUTPUT_DIR"
