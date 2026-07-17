#!/bin/bash
# One-step W4A4 real-quant smoke for the public 300-step Super 120B-A12B recipe.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=16
GPUS_PER_NODE=4
SEGMENT_SIZE=16
STEPS_PER_RUN=1
MAX_STEPS=1
NUM_RUNS=1
NUM_MINUTES=240
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
uv run --no-sync examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    cluster.num_nodes=$NUM_NODES \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    grpo.val_at_start=false \
    grpo.val_at_end=true \
    grpo.max_val_samples=16 \
    grpo.val_batch_size=16 \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=false \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run --no-sync tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

# Real-quant rollout must go through the ModelOpt NVFP4 vLLM kernel path.
grep -q "VllmQuantInternalWorkerExtension" "$RUN_LOG"
grep -q "Detected ModelOpt NVFP4 checkpoint" "$RUN_LOG"
grep -q "quantization=modelopt" "$RUN_LOG"
grep -q "examples/modelopt/quant_configs/nvfp4_experts.yaml" "$RUN_LOG"
! grep -q "FakeQuantWorker" "$RUN_LOG"
! grep -q "VLLM_QUANT_CFG" "$RUN_LOG"
! grep -q "Using NvFp4LinearBackend.MARLIN" "$RUN_LOG"

uv run --no-sync tests/check_metrics.py "$JSON_METRICS" \
    'data["train/num_valid_samples"]["1"] >= 64' \
    'data["train/reward"]["1"] >= 0.4' \
    'data["train/gen_kl_error"]["1"] < 0.1' \
    'data["train/token_mult_prob_error"]["1"] < 1.15' \
    'data["validation/accuracy"]["1"] >= 0.5'
