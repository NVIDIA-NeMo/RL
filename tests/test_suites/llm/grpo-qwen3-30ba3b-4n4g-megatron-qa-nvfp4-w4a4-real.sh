#!/bin/bash
# Two-step GB200 smoke test for Megatron -> vLLM W4A4 real-quant refits.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=4
STEPS_PER_RUN=2
MAX_STEPS=2
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=180
SNAPSHOT_MEGATRON_BRIDGE=1
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
MEGATRON_BRIDGE_ROOT=${MEGATRON_BRIDGE_ROOT:-$HOME/modelopt/Megatron-Bridge}
MEGATRON_LM_ROOT=${MEGATRON_LM_ROOT:-$PROJECT_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM}
if [[ ! -f "$MEGATRON_BRIDGE_ROOT/src/megatron/bridge/models/conversion/modelopt_utils.py" ]]; then
    echo "[ERROR] Invalid Megatron-Bridge source root: $MEGATRON_BRIDGE_ROOT"
    exit 1
fi
if ! grep -q "modelopt_finalize_ep_weight" \
    "$MEGATRON_BRIDGE_ROOT/src/megatron/bridge/models/conversion/auto_bridge.py"; then
    echo "[ERROR] Megatron-Bridge lacks grouped-MoE ModelOpt export support: $MEGATRON_BRIDGE_ROOT"
    exit 1
fi
if [[ ! -f "$MEGATRON_LM_ROOT/megatron/core/__init__.py" ]]; then
    echo "[ERROR] Invalid Megatron-LM source root: $MEGATRON_LM_ROOT"
    exit 1
fi
BRIDGE_REVISION=$(git -C "$MEGATRON_BRIDGE_ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
if ! git -C "$MEGATRON_BRIDGE_ROOT" diff-index --quiet HEAD --; then
    BRIDGE_REVISION="${BRIDGE_REVISION}-dirty"
fi
echo "[INFO] Megatron-Bridge source: $MEGATRON_BRIDGE_ROOT ($BRIDGE_REVISION)"
# Actor-specific binary dependencies come from each Ray actor's virtualenv.
# Only source roots belong on the shared PYTHONPATH.
export PYTHONPATH="$MEGATRON_BRIDGE_ROOT/src:$MEGATRON_LM_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"

uv run --no-sync examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    grpo.val_at_start=true \
    grpo.val_at_end=true \
    grpo.max_val_samples=32 \
    grpo.val_batch_size=32 \
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
! grep -q "Using NvFp4LinearBackend.MARLIN" "$RUN_LOG"

MAX_RECORDED_STEP=$(jq -r 'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' "$JSON_METRICS")
if [[ $MAX_RECORDED_STEP -lt $MAX_STEPS ]]; then
    echo "[ERROR] Expected train/loss through step $MAX_STEPS, found step $MAX_RECORDED_STEP"
    exit 1
fi

uv run --no-sync tests/check_metrics.py "$JSON_METRICS" \
    "data[\"train/reward\"][\"$MAX_STEPS\"] >= 0.25" \
    "data[\"validation/accuracy\"][\"$MAX_STEPS\"] >= 0.4" \
    "data[\"train/gen_kl_error\"][\"$MAX_STEPS\"] < 0.03" \
    "data[\"train/js_divergence_error\"][\"$MAX_STEPS\"] < 0.007" \
    "data[\"train/approx_entropy\"][\"$MAX_STEPS\"] < 0.35"

mapfile -t TRAIN_DATA_FILES < <(
    find "$LOG_DIR" -type f -name 'train_data_step*.jsonl' -print | sort -V
)
if [[ ${#TRAIN_DATA_FILES[@]} -ne $MAX_STEPS ]]; then
    echo "[ERROR] Expected $MAX_STEPS rollout JSONL files, found ${#TRAIN_DATA_FILES[@]}"
    exit 1
fi
