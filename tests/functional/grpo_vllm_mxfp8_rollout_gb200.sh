#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory "$PROJECT_ROOT"

set -eou pipefail

EXP_NAME=$(basename "$0" .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf "$EXP_DIR" "$LOG_DIR"
mkdir -p "$EXP_DIR" "$LOG_DIR"

assert_grep() {
    local pattern=$1
    local file=$2
    grep -Eq "$pattern" "$file" || {
        echo "[FAIL] expected '$pattern' in $file"
        exit 1
    }
}

cd "$PROJECT_ROOT"
uv run coverage run -a --data-file="$PROJECT_ROOT/tests/.coverage" --source="$PROJECT_ROOT/nemo_rl" \
    "$PROJECT_ROOT/examples/run_grpo.py" \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    grpo.max_num_steps=2 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=1 \
    policy.max_total_sequence_length=256 \
    policy.generation.max_new_tokens=128 \
    policy.generation.vllm_cfg.precision=fp8 \
    ++policy.generation.vllm_cfg.is_mx=true \
    policy.generation.vllm_cfg.kv_cache_dtype=auto \
    policy.generation.vllm_cfg.max_model_len=256 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.5 \
    policy.generation.vllm_cfg.enforce_eager=true \
    policy.generation.vllm_cfg.use_tqdm=false \
    '++policy.generation.vllm_cfg.quantization_ignored_layer_kws=[q_proj,k_proj,v_proj,o_proj]' \
    loss_fn.use_importance_sampling_correction=true \
    cluster.gpus_per_node=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

uv run tests/check_metrics.py "$JSON_METRICS" \
    'len(data["train/loss"]) == 2' \
    'max(data["train/gen_kl_error"]) < 0.05' \
    'max(data["train/token_mult_prob_error"]) < 2.0'

assert_grep 'quantization=modelopt_mxfp8|quant_algo.*MXFP8' "$RUN_LOG"
assert_grep 'ignored_layers' "$RUN_LOG"

echo "[PASS] GB200 dense Qwen GRPO vLLM MXFP8 rollout functional test"
