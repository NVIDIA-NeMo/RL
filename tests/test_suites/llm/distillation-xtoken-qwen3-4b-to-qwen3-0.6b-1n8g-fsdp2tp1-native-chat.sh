#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

NUM_NODES=1
MAX_STEPS=20
NUM_MINUTES=20
exit_if_max_steps_reached
cd "$PROJECT_ROOT"

# This pair shares a vocabulary. An explicit projection exercises the native
# cross-tokenizer alignment path instead of the same-tokenizer bypass.
PROJ_DIR="$EXP_DIR/projection"
mkdir -p "$PROJ_DIR"
PROJ_PATH="$PROJ_DIR/native_chat_special.pt"
if [[ ! -f "$PROJ_PATH" ]]; then
  uv run python -m tools.x_token.minimal_projection_via_multitoken \
    --student-model Qwen/Qwen3-4B \
    --teacher-model Qwen/Qwen3-4B \
    --top-k 4 --enable-special-token-mapping --enable-exact-match \
    --disable-reverse-pass --disable-scale-trick \
    --output-filename native_chat --output-dir "$PROJ_DIR"
fi

uv run examples/run_xtoken_off_policy_distillation.py \
  --config "$CONFIG_PATH" \
  distillation.max_num_steps="$MAX_STEPS" \
  teachers.0.aligner.projection_matrix_path="$PROJ_PATH" \
  logger.log_dir="$LOG_DIR" \
  logger.tensorboard_enabled=True \
  checkpointing.enabled=False \
  "$@" 2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
# This is a boundary/masking smoke test, with no unmeasured convergence target.
uv run tests/check_metrics.py "$JSON_METRICS" \
  'len(data["train/loss"]) >= 20' \
  'all_finite(data["train/loss"])'
