#!/bin/bash
# Two-step GB200 smoke for Nemotron 3 Super non-gated expert W4A4 refits.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=32
GPUS_PER_NODE=4
SEGMENT_SIZE=16
STEPS_PER_RUN=2
MAX_STEPS=2
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=72
SNAPSHOT_MEGATRON_BRIDGE=1
# ===== END CONFIG =====

if [[ -n "${RUN_ROOT:-}" ]]; then
    EXP_DIR=$RUN_ROOT
    LOG_DIR=$RUN_ROOT/logs
    CKPT_DIR=$RUN_ROOT/ckpts
    JSON_METRICS=$RUN_ROOT/metrics.json
    RUN_LOG=$RUN_ROOT/run.log
    mkdir -p "$LOG_DIR" "$CKPT_DIR" "$RUN_ROOT/tb_logs" "$RUN_ROOT/wandb"
fi

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
SUPER_MODEL_ROOT=${SUPER_MODEL_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model/nvidia_nvidia-nemotron-3-super-120b-a12b-bf16/hf/hf-d51eab0_orig}
if ! jq -e '.architectures | index("NemotronHForCausalLM")' \
    "$SUPER_MODEL_ROOT/config.json" >/dev/null; then
    echo "[ERROR] Invalid Nemotron 3 Super model root: $SUPER_MODEL_ROOT"
    exit 1
fi

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
if ! grep -Fq 'weight_name.endswith(".experts.up_proj")' \
    "$MEGATRON_BRIDGE_ROOT/src/megatron/bridge/models/conversion/modelopt_utils.py"; then
    echo "[ERROR] Megatron-Bridge lacks non-gated expert W4A4 export support: $MEGATRON_BRIDGE_ROOT"
    exit 1
fi
if grep -Fq "no supported input-scale transport" \
    "$MEGATRON_BRIDGE_ROOT/src/megatron/bridge/models/conversion/modelopt_utils.py"; then
    echo "[ERROR] Megatron-Bridge disables non-gated expert W4A4 export: $MEGATRON_BRIDGE_ROOT"
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

MODE_CACHE_ROOT=${MODE_CACHE_ROOT:-$EXP_DIR/model_cache}
mkdir -p "$MODE_CACHE_ROOT/megatron" "$MODE_CACHE_ROOT/modelopt" "$MODE_CACHE_ROOT/vllm"
export NRL_MEGATRON_CHECKPOINT_DIR=$MODE_CACHE_ROOT/megatron
export NRL_MODELOPT_CHECKPOINT_DIR=$MODE_CACHE_ROOT/modelopt
export VLLM_CACHE_ROOT=$MODE_CACHE_ROOT/vllm
export WANDB_DIR=$EXP_DIR/wandb
export UV_NO_SYNC=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$MEGATRON_BRIDGE_ROOT/src:$MEGATRON_LM_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"

WANDB_RUN_ID=${WANDB_RUN_ID:-${EXP_NAME}-${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)-$$}}
uv run --no-sync examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    policy.model_name="$SUPER_MODEL_ROOT" \
    policy.tokenizer.name="$SUPER_MODEL_ROOT" \
    cluster.num_nodes=$NUM_NODES \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    cluster.segment_size=$SEGMENT_SIZE \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl-qarl \
    logger.wandb.name="$EXP_NAME" \
    ++logger.wandb.id="$WANDB_RUN_ID" \
    ++logger.wandb.resume=never \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    logger.tensorboard.log_dir="$EXP_DIR/tb_logs" \
    checkpointing.enabled=False \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run --no-sync tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

grep -q "VllmQuantInternalWorkerExtension" "$RUN_LOG"
grep -q "Detected ModelOpt NVFP4 checkpoint" "$RUN_LOG"
grep -q "quantization=modelopt" "$RUN_LOG"
grep -q "Skipping FlashInfer autotune because it is disabled" "$RUN_LOG"
! grep -q "FakeQuantWorker" "$RUN_LOG"
! grep -q "VLLM_QUANT_CFG" "$RUN_LOG"
! grep -q "Using NvFp4LinearBackend.MARLIN" "$RUN_LOG"
! grep -q "Autotuning process ends" "$RUN_LOG"
! grep -q "RayChannelTimeoutError" "$RUN_LOG"

MAX_RECORDED_STEP=$(jq -r 'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' "$JSON_METRICS")
if [[ $MAX_RECORDED_STEP -lt $MAX_STEPS ]]; then
    echo "[ERROR] Expected train/loss through step $MAX_STEPS, found step $MAX_RECORDED_STEP"
    exit 1
fi

uv run --no-sync tests/check_metrics.py "$JSON_METRICS" \
    "data[\"train/global_valid_toks\"][\"$MAX_STEPS\"] > 0"

mapfile -t TRAIN_DATA_FILES < <(
    find "$LOG_DIR" -type f -name 'train_data_step*.jsonl' -print | sort -V
)
if [[ ${#TRAIN_DATA_FILES[@]} -ne $MAX_STEPS ]]; then
    echo "[ERROR] Expected $MAX_STEPS rollout JSONL files, found ${#TRAIN_DATA_FILES[@]}"
    exit 1
fi

if [[ "${WANDB_MODE:-online}" == "offline" ]]; then
    OFFLINE_RUN_DIR=$(find "$LOG_DIR" -type d -name 'offline-run-*' -print -quit)
    echo "[INFO] W&B offline run: ${OFFLINE_RUN_DIR:-not found under $LOG_DIR}"
else
    echo "[INFO] W&B: https://wandb.ai/nvidia/nemo-rl-qarl/runs/$WANDB_RUN_ID"
fi
