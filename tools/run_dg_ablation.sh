#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${1:-$ROOT_DIR/results/dg_ablation_${STAMP}}"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"
export PYTHONUNBUFFERED=1
export NEMO_RL_PY_EXECUTABLES_SYSTEM="${NEMO_RL_PY_EXECUTABLES_SYSTEM:-1}"
UV_RUN=(uv run)
if [[ "${UV_NO_SYNC:-0}" == "1" ]]; then
  UV_RUN+=(--no-sync)
fi
UV_RUN+=(python)

MAX_STEPS="${MAX_STEPS:-30}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-8}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-8}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-64}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-4}"
LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-4}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
VAL_PERIOD="${VAL_PERIOD:-10}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
SAVE_PERIOD="${SAVE_PERIOD:-10}"
SPLIT_VALIDATION_SIZE="${SPLIT_VALIDATION_SIZE:-0.01}"
MATH_NUM_WORKERS="${MATH_NUM_WORKERS:-4}"
SEED="${SEED:-42}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
VAL_DATA_PATH="${VAL_DATA_PATH:-}"
INPUT_KEY="${INPUT_KEY:-input}"
OUTPUT_KEY="${OUTPUT_KEY:-output}"
DTENSOR_V2="${DTENSOR_V2:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_NAME}"
PROMPT_FILE="${PROMPT_FILE:-examples/prompts/cot.txt}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-}"
VAL_DATASET_NAME="${VAL_DATASET_NAME:-}"
VAL_SPLIT="${VAL_SPLIT:-}"

mkdir -p "$OUT_ROOT" "$HF_HOME"

COMMON_OVERRIDES=(
  "grpo.max_num_steps=${MAX_STEPS}"
  "grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP}"
  "grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT}"
  "grpo.val_period=${VAL_PERIOD}"
  "grpo.val_at_end=true"
  "grpo.max_val_samples=${MAX_VAL_SAMPLES}"
  "grpo.val_batch_size=${VAL_BATCH_SIZE}"
  "grpo.seed=${SEED}"
  "policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE}"
  "policy.train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE}"
  "policy.logprob_batch_size=${LOGPROB_BATCH_SIZE}"
  "policy.max_total_sequence_length=${MAX_TOTAL_SEQUENCE_LENGTH}"
  "policy.model_name=${MODEL_NAME}"
  "policy.tokenizer.name=${TOKENIZER_NAME}"
  "policy.generation.max_new_tokens=${MAX_NEW_TOKENS}"
  "policy.generation.vllm_cfg.enable_vllm_metrics_logger=false"
  "checkpointing.enabled=false"
  "checkpointing.keep_top_k=1"
  "checkpointing.save_period=${SAVE_PERIOD}"
  "data.train.split_validation_size=${SPLIT_VALIDATION_SIZE}"
  "data.num_workers=1"
  "env.math.num_workers=${MATH_NUM_WORKERS}"
  "logger.tensorboard_enabled=true"
  "logger.wandb_enabled=false"
  "logger.mlflow_enabled=false"
  "logger.swanlab_enabled=false"
  "logger.monitor_gpus=true"
  "logger.num_val_samples_to_print=0"
  "data.default.prompt_file=${PROMPT_FILE}"
)

if [[ "$DTENSOR_V2" == "0" ]]; then
  COMMON_OVERRIDES+=(
    "policy.dtensor_cfg._v2=false"
    "checkpointing.model_save_format=null"
  )
fi

if [[ -n "$TRAIN_DATASET_NAME" ]]; then
  COMMON_OVERRIDES+=(
    "data.train.dataset_name=${TRAIN_DATASET_NAME}"
  )
  if [[ -n "$TRAIN_SPLIT" ]]; then
    COMMON_OVERRIDES+=(
      "+data.train.split=${TRAIN_SPLIT}"
    )
  fi
elif [[ -n "$TRAIN_DATA_PATH" ]]; then
  COMMON_OVERRIDES+=(
    "data.train.dataset_name=ResponseDataset"
    "+data.train.data_path=${TRAIN_DATA_PATH}"
    "data.train.split_validation_size=0"
    "+data.default.dataset_name=ResponseDataset"
    "+data.default.input_key=${INPUT_KEY}"
    "+data.default.output_key=${OUTPUT_KEY}"
  )
fi

if [[ -n "$VAL_DATASET_NAME" ]]; then
  COMMON_OVERRIDES+=(
    "++data.validation.dataset_name=${VAL_DATASET_NAME}"
  )
  if [[ -n "$VAL_SPLIT" ]]; then
    COMMON_OVERRIDES+=(
      "++data.validation.split=${VAL_SPLIT}"
    )
  fi
elif [[ -n "$VAL_DATA_PATH" ]]; then
  COMMON_OVERRIDES+=(
    "++data.validation.dataset_name=ResponseDataset"
    "++data.validation.data_path=${VAL_DATA_PATH}"
  )
elif [[ -n "$TRAIN_DATASET_NAME" || -n "$TRAIN_DATA_PATH" ]]; then
  COMMON_OVERRIDES+=(
    "data.train.split_validation_size=${SPLIT_VALIDATION_SIZE}"
  )
fi

run_one() {
  local label="$1"
  shift
  local run_dir="${OUT_ROOT}/${label}"
  local stdout_log="${run_dir}/stdout.log"
  mkdir -p "$run_dir"

  local cmd=(
    "${UV_RUN[@]}" examples/run_grpo.py
    "logger.log_dir=${run_dir}/logs"
    "checkpointing.checkpoint_dir=${run_dir}/checkpoints"
  )
  cmd+=("${COMMON_OVERRIDES[@]}")
  cmd+=("$@")

  printf '%s\n' "${cmd[@]}" > "${run_dir}/command.txt"
  (
    cd "$ROOT_DIR"
    "${cmd[@]}" 2>&1 | tee "$stdout_log"
  )
}

run_one "baseline" "loss_fn.dg_enabled=false"
run_one "dg_eta1p0" "loss_fn.dg_enabled=true" "loss_fn.dg_eta=1.0"
run_one "dg_eta0p5" "loss_fn.dg_enabled=true" "loss_fn.dg_eta=0.5"

(
  cd "$ROOT_DIR"
  "${UV_RUN[@]}" tools/plot_grpo_ablation.py \
    --run "baseline=${OUT_ROOT}/baseline" \
    --run "dg_eta1p0=${OUT_ROOT}/dg_eta1p0" \
    --run "dg_eta0p5=${OUT_ROOT}/dg_eta0p5" \
    --metric "validation/accuracy" \
    --metric "train/reward_mean" \
    --metric "train/reward_std" \
    --metric "train/loss" \
    --metric "train/advantages/mean" \
    --metric "train/baseline_reward/pct_1" \
    --metric "train/dg_gate_mean" \
    --metric "train/dg_gate_positive_mean" \
    --metric "train/dg_gate_negative_mean" \
    --metric "train/dg_gate_spread" \
    --metric "train/dg_surprisal_mean" \
    --metric "train/dg_surprisal_max" \
    --metric "timing/train/total_step_time" \
    --out-dir "${OUT_ROOT}/plots"
)

printf 'Artifacts written to %s\n' "$OUT_ROOT"
