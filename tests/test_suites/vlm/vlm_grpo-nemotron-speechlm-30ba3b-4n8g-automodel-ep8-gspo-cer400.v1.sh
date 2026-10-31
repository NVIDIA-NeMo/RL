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
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../../..")
CONFIG_PATH="$PROJECT_ROOT/examples/configs/recipes/vlm/vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.yaml"
test -f "$CONFIG_PATH"
if [[ -n "${TEST_DRYRUN:-}" ]]; then
  exit 0
fi

: "${SPEECHLM_GSPO_MODEL:?Set the full Speech seed HF directory}"
: "${SPEECHLM_GSPO_DATA:?Set the training consumer Parquet path}"
: "${SPEECHLM_GSPO_OUTPUT_DIR:?Set a dedicated writable output root}"
mkdir -p "$SPEECHLM_GSPO_OUTPUT_DIR"

# Native policy actors inherit the prepared driver interpreter; generation
# keeps the normal separate vLLM actor environment. See the SpeechLM guide.
: "${SPEECHLM_GSPO_PYTHON:?Set the prepared policy/driver interpreter}"
cd "$PROJECT_ROOT"
uv run --no-project "$SPEECHLM_GSPO_PYTHON" examples/run_vlm_grpo.py \
  --config "$CONFIG_PATH" grpo.max_num_steps=20 "$@"
# Require a complete save, including optimizer state, before restarting Python.
uv run --no-project "$SPEECHLM_GSPO_PYTHON" tools/verify_speechlm_resume.py \
  --output-dir "$SPEECHLM_GSPO_OUTPUT_DIR" --initial-only

# The existing checkpoint manager restores the latest native save. Retain the
# second invocation's log so the test can prove exactly two resumed updates.
uv run --no-project "$SPEECHLM_GSPO_PYTHON" examples/run_vlm_grpo.py \
  --config "$CONFIG_PATH" grpo.max_num_steps=22 "$@" \
  2>&1 | tee "$SPEECHLM_GSPO_OUTPUT_DIR/resume2.log"
uv run --no-project "$SPEECHLM_GSPO_PYTHON" tools/verify_speechlm_resume.py \
  --output-dir "$SPEECHLM_GSPO_OUTPUT_DIR"
