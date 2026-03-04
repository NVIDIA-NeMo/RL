#!/usr/bin/env bash
set -euo pipefail
cd /home/scratch.shaunakj_other/Development/RL

# Run inside an 8-GPU allocation.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES is empty. Run this inside the 8-GPU allocation." >&2
  exit 1
fi

export HOME=/home/scratch.shaunakj_other
export PATH=/home/scratch.shaunakj_other/Development/RL/.venv/bin:/home/scratch.shaunakj_other/.local/bin:$PATH
export VIRTUAL_ENV=/home/scratch.shaunakj_other/Development/RL/.venv

export TMPDIR=/home/scratch.shaunakj_other/t
export TMP=/home/scratch.shaunakj_other/t
export TEMP=/home/scratch.shaunakj_other/t

export HF_HOME=/home/scratch.shaunakj_other/.cache/hf_user
export HF_HUB_CACHE=/home/scratch.shaunakj_other/.cache/hf_user/hub
export HUGGINGFACE_HUB_CACHE=/home/scratch.shaunakj_other/.cache/hf_user/hub
export TRANSFORMERS_CACHE=/home/scratch.shaunakj_other/.cache/hf_user/hub

export UV=/home/scratch.shaunakj_other/.local/bin/uv
export UV_CACHE_DIR=/home/scratch.shaunakj_other/.cache/uv

export RAY_TMPDIR=/home/scratch.shaunakj_other/t
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_CLIENT_MODE=0
export PYTHONUNBUFFERED=1

unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK NODE_RANK
unset AVAILABLE_ADDR_LIST AVAILABLE_PORT_LIST

RAY_BIN=/home/scratch.shaunakj_other/Development/RL/.venv/bin/ray
PYTHON_BIN=/home/scratch.shaunakj_other/Development/RL/.venv/bin/python

RUN_STEPS="${RUN_STEPS:-60}"
MAX_EPOCHS="${MAX_EPOCHS:-2}"
TRAIN_LR="${TRAIN_LR:-1.5e-5}"
RATIO_CLIP="${RATIO_CLIP:-0.30}"
REF_KL="${REF_KL:-0.005}"
SAVE_PERIOD="${SAVE_PERIOD:-40}"
SEED="${SEED:-124}"
BASE_MODEL="${BASE_MODEL:-/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"

RUN_TS="$(date +%F-%H%M%S)"
TAG="grpo-32b-eagle3-temp0p6-s4-seed${SEED}-train-reasonable-sp${SAVE_PERIOD}-${RUN_TS}"
RUN_ROOT="${RUN_ROOT:-/home/scratch.shaunakj_other/runs/${TAG}}"
LOG_DIR="${RUN_ROOT}/logs"
CKPT_DIR="${RUN_ROOT}/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "[launch] TAG=$TAG"
echo "[launch] RUN_ROOT=$RUN_ROOT"
echo "[launch] LOG_DIR=$LOG_DIR"
echo "[launch] CKPT_DIR=$CKPT_DIR"

echo "[launch] cfg RUN_STEPS=$RUN_STEPS MAX_EPOCHS=$MAX_EPOCHS LR=$TRAIN_LR CLIP=$RATIO_CLIP REF_KL=$REF_KL SAVE_PERIOD=$SAVE_PERIOD"

run_once() {
  "$PYTHON_BIN" examples/run_grpo.py \
    --config=/home/scratch.shaunakj_other/tmp/grpo-qwen3-32b-spec-decode-lowbatch-1n8g.2026-02-25-235814.seed124.normal100.local.yaml \
    ++grpo.max_num_steps="$RUN_STEPS" \
    ++grpo.max_num_epochs="$MAX_EPOCHS" \
    ++grpo.val_period=0 \
    ++grpo.val_at_start=false \
    ++grpo.val_at_end=false \
    ++grpo.seed="$SEED" \
    ++grpo.num_prompts_per_step=2 \
    ++grpo.num_generations_per_prompt=4 \
    ++policy.dtensor_cfg.activation_checkpointing=true \
    ++policy.model_name="$BASE_MODEL" \
    ++policy.tokenizer.name="$BASE_MODEL" \
    ++policy.max_total_sequence_length=5120 \
    ++policy.generation.max_new_tokens=4096 \
    ++policy.generation.temperature=0.6 \
    ++policy.generation.top_p=1.0 \
    ++policy.generation.top_k=null \
    ++policy.generation.vllm_cfg.load_format=auto \
    ++policy.generation.vllm_cfg.max_model_len=5120 \
    ++policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN \
    ++policy.generation.vllm_kwargs.speculative_config.model=/home/scratch.shaunakj_other/.cache/hf_user/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/e5756763c9b3bef3cc260cab70b76008fb42a81b \
    ++policy.generation.vllm_kwargs.speculative_config.method=eagle3 \
    ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=4 \
    ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 \
    ++data.train.dataset_name=ResponseDataset \
    ++data.train.data_path=/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl \
    ++data.train.split_validation_size=0.0 \
    ++data.validation=null \
    ++data.default.dataset_name=ResponseDataset \
    ++data.default.input_key=input \
    ++data.default.output_key=output \
    ++env.math.num_workers=1 \
    ++logger.log_dir="$LOG_DIR" \
    ++checkpointing.enabled=true \
    ++checkpointing.save_period="$SAVE_PERIOD" \
    ++checkpointing.keep_top_k=6 \
    ++checkpointing.checkpoint_dir="$CKPT_DIR" \
    ++policy.optimizer.kwargs.lr="$TRAIN_LR" \
    ++loss_fn.ratio_clip_min="$RATIO_CLIP" \
    ++loss_fn.ratio_clip_max="$RATIO_CLIP" \
    ++loss_fn.reference_policy_kl_penalty="$REF_KL" \
    ++policy.max_grad_norm=1.0
}

max_attempts=3
attempt=1
while (( attempt <= max_attempts )); do
  echo "[launch] attempt ${attempt}/${max_attempts} on $(hostname) at $(date '+%F %T %Z')"
  "$RAY_BIN" stop --force >/dev/null 2>&1 || true
  sleep 2

  if run_once; then
    exit 0
  fi

  rc=$?
  echo "[launch] attempt ${attempt} failed with exit code ${rc}" >&2
  if (( attempt == max_attempts )); then
    exit "$rc"
  fi

  "$RAY_BIN" stop --force >/dev/null 2>&1 || true
  pkill -f 'VllmAsyncGenerationWorker|EngineCore_DP0|run_grpo.py' >/dev/null 2>&1 || true
  sleep 10
  ((attempt += 1))
done
