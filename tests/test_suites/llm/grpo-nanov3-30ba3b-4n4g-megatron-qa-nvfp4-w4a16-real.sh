#!/bin/bash
# One-step Nano3 W4A16 real-quant rollout check. This validates that Megatron
# exports ModelOpt NVFP4 packed tensors, vLLM loads them through the real
# ModelOpt kernel path, and generation/policy logprobs stay aligned.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=4
STEPS_PER_RUN=1
MAX_STEPS=1
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=180
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${PYTHONPATH:-}"

uv run --no-sync examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    cluster.num_nodes=$NUM_NODES \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run --no-sync tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

grep -q "VllmQuantInternalWorkerExtension" "$RUN_LOG"
grep -q "Detected ModelOpt NVFP4 checkpoint" "$RUN_LOG"
grep -q "quantization=modelopt" "$RUN_LOG"
! grep -q "FakeQuantWorker" "$RUN_LOG"
! grep -q "VLLM_QUANT_CFG" "$RUN_LOG"

if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' "$JSON_METRICS") -ge $MAX_STEPS ]]; then
    uv run --no-sync tests/check_metrics.py "$JSON_METRICS" \
        'data["train/gen_kl_error"]["1"] < 0.003' \
        'max(data["train/token_mult_prob_error"]) < 1.05' \
        'data["train/loss"]["1"] > 0.0' \
        'data["train/num_valid_samples"]["1"] > 0'
fi
