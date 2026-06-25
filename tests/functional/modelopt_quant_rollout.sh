#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

set -euo pipefail

EXP_NAME=$(basename "$0" .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf "$EXP_DIR"
mkdir -p "$EXP_DIR"

assert_grep() {
    local pattern=$1
    local file=$2
    grep -q "$pattern" "$file" || {
        echo "[FAIL] expected '$pattern' in $file"
        exit 1
    }
}

assert_not_grep() {
    local pattern=$1
    local file=$2
    if grep -q "$pattern" "$file"; then
        echo "[FAIL] did not expect '$pattern' in $file"
        exit 1
    fi
}

run_quant_rollout_case() {
    local case_name=$1
    local quant_cfg=$2
    local real_quant=$3
    local gen_kl_error_step1_max=$4
    local token_mult_prob_error_max=$5
    local model_name=$6
    local log_dir="$EXP_DIR/$case_name/logs"
    local metrics_json="$EXP_DIR/$case_name/metrics.json"
    local run_log="$EXP_DIR/$case_name/run.log"
    local megatron_cache_root="${NRL_MEGATRON_CHECKPOINT_DIR:-$PROJECT_ROOT/.cache/modelopt_quant_rollout/megatron}"
    local quant_cfg_hash
    if [[ -f "$PROJECT_ROOT/$quant_cfg" ]]; then
        quant_cfg_hash=$(sha256sum "$PROJECT_ROOT/$quant_cfg" | cut -c1-12)
    else
        quant_cfg_hash=$(printf '%s' "$quant_cfg" | sha256sum | cut -c1-12)
    fi
    local megatron_cache_dir="$megatron_cache_root/$case_name-$quant_cfg_hash"
    local real_quant_override=()
    shift 6

    if [[ "$real_quant" == "true" ]]; then
        real_quant_override+=(++policy.generation.real_quant=true)
    fi

    rm -rf "$EXP_DIR/$case_name"
    mkdir -p "$log_dir" "$megatron_cache_dir"

    cd "$PROJECT_ROOT"
    NRL_MEGATRON_CHECKPOINT_DIR="$megatron_cache_dir" \
    uv run --extra modelopt --group test \
        coverage run -a --data-file="$PROJECT_ROOT/tests/.coverage" --source="$PROJECT_ROOT/nemo_rl" \
        "$PROJECT_ROOT/examples/run_grpo.py" \
        --config "$PROJECT_ROOT/examples/modelopt/qa_grpo_math_megatron.yaml" \
        policy.model_name=$model_name \
        policy.tokenizer.name=$model_name \
        policy.quant_cfg=$quant_cfg \
        policy.generation.quant_cfg=$quant_cfg \
        policy.quant_calib_size=4 \
        policy.quant_batch_size=1 \
        policy.quant_sequence_length=128 \
        policy.max_total_sequence_length=256 \
        policy.train_global_batch_size=4 \
        policy.train_micro_batch_size=1 \
        policy.logprob_batch_size=4 \
        policy.generation.max_new_tokens=128 \
        policy.generation.vllm_cfg.max_model_len=256 \
        policy.generation.vllm_cfg.gpu_memory_utilization=0.5 \
        policy.generation.vllm_cfg.enforce_eager=true \
        grpo.num_prompts_per_step=2 \
        grpo.num_generations_per_prompt=2 \
        grpo.max_num_steps=1 \
        grpo.val_period=1 \
        grpo.max_val_samples=8 \
        grpo.val_batch_size=8 \
        env.math.num_workers=2 \
        cluster.gpus_per_node=2 \
        logger.tensorboard_enabled=true \
        logger.log_dir="$log_dir" \
        logger.wandb_enabled=false \
        logger.monitor_gpus=true \
        checkpointing.enabled=false \
        "${real_quant_override[@]}" \
        "$@" \
        2>&1 | tee "$run_log"

    uv run --extra modelopt --group test tests/json_dump_tb_logs.py "$log_dir" --output_path "$metrics_json"

    uv run --extra modelopt --group test tests/check_metrics.py "$metrics_json" \
        "data[\"train/gen_kl_error\"][\"1\"] < $gen_kl_error_step1_max" \
        "max(data[\"train/token_mult_prob_error\"]) < $token_mult_prob_error_max"

    assert_grep "VllmQuantInternalWorkerExtension" "$run_log"
    if [[ "$real_quant" == "true" ]]; then
        assert_grep "Detected ModelOpt NVFP4 checkpoint" "$run_log"
        assert_not_grep "FakeQuantWorker" "$run_log"
        assert_not_grep "VLLM_QUANT_CFG" "$run_log"
    else
        assert_grep "FakeQuantWorker" "$run_log"
        assert_grep "VLLM_QUANT_CFG" "$run_log"
        assert_not_grep "Detected ModelOpt NVFP4 checkpoint" "$run_log"
    fi
}

run_quant_rollout_case w4a16_real_quant examples/modelopt/quant_configs/nvfp4_a16_mlp_only.yaml true 0.003 1.05 Qwen/Qwen2.5-0.5B "$@"
run_quant_rollout_case w4a8_fake_quant examples/modelopt/quant_configs/nvfp4_w4a8_fp8.yaml false 0.006 1.06 Qwen/Qwen2.5-0.5B "$@"

echo "[PASS] ModelOpt W4A16 real-quant and W4A8 fake-quant rollout functional test"
