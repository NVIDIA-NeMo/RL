#!/bin/bash
# Submit MATH + MMLU evals for step_1000 checkpoints from:
#   1. Off-policy distillation (forward-kl-cosine) Llama 1B
#   2. SFT arrow Llama 1B
#
# Each job first consolidates the sharded checkpoint to HF format (if needed),
# then runs the eval.
#
# Usage:
#   bash submit_eval_step1000.sh

set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL"
PROJECT_ROOT="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha"

DISTILL_CKPT="${PROJECT_ROOT}/nemo_rl/RL/checkpoints/distillation-forward-kl-cosine-meta-llama/Llama-3.2-1B/step_1000/policy/weights"
DISTILL_HF="${PROJECT_ROOT}/nemo_rl/RL/checkpoints/distillation-forward-kl-cosine-meta-llama/Llama-3.2-1B/step_1000_hf"

SFT_CKPT="${PROJECT_ROOT}/nemo_rl/RL/results/sft-arrow-eval/step_1000/policy/weights"
SFT_HF="${PROJECT_ROOT}/nemo_rl/RL/results/sft-arrow-eval/step_1000_hf"

MODEL_NAME="meta-llama/Llama-3.2-1B"
MATH_CONFIG="${REPO_ROOT}/examples/configs/evals/llama_math_eval.yaml"
MMLU_CONFIG="${REPO_ROOT}/examples/configs/evals/llama_mmlu_eval.yaml"
CONSOLIDATE_SCRIPT="${REPO_ROOT}/examples/converters/consolidate_checkpoint.py"

MY_CONTAINER="${PROJECT_ROOT}/nemo_rl/nemo-rl.sqsh"
MOUNTS="${PROJECT_ROOT}:${PROJECT_ROOT},/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl:/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl,/lustre/fsw/portfolios/llmservice/users/sdiao/data:/lustre/fsw/portfolios/llmservice/users/sdiao/data"

submit_eval_job() {
  local CKPT_SHARDED="$1"
  local CKPT_HF="$2"
  local EVAL_CONFIG="$3"
  local EVAL_TYPE="$4"
  local MODEL_TAG="$5"

  local EXP_NAME="Eval-${EVAL_TYPE}-${MODEL_TAG}-step1000"
  local LOG_DIR_BASE="${REPO_ROOT}/llama-eval"

  read -r -d '' CMD <<CMDEOF || true
export WANDB_API_KEY=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
export HF_TOKEN=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

export HF_HOME=${PROJECT_ROOT}/hf
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_CACHE=${PROJECT_ROOT}/hf_datasets_cache

# Consolidate checkpoint if not already done
if [ ! -f "${CKPT_HF}/config.json" ]; then
  echo "Consolidating checkpoint from ${CKPT_SHARDED} to ${CKPT_HF}..."
  uv run python ${CONSOLIDATE_SCRIPT} \
    --input ${CKPT_SHARDED} \
    --output ${CKPT_HF} \
    --model-name ${MODEL_NAME}
  echo "Consolidation complete."
else
  echo "HF checkpoint already exists at ${CKPT_HF}, skipping consolidation."
fi

# Run evaluation
uv run ${REPO_ROOT}/examples/run_eval.py \
  --config ${EVAL_CONFIG} \
  generation.model_name=${CKPT_HF} \
  tokenizer.name=${MODEL_NAME}
CMDEOF

  export COMMAND="$CMD"
  export BASE_LOG_DIR="$LOG_DIR_BASE"
  export CONTAINER="${MY_CONTAINER}"
  export MOUNTS="${MOUNTS}"

  echo "Submitting: ${EXP_NAME}"
  sbatch \
    --nodes=1 \
    --account=coreai_dlalgo_genai \
    --job-name="nemo-rl.${EXP_NAME}" \
    --partition=batch \
    --time=1:0:0 \
    --gres=gpu:8 \
    "${REPO_ROOT}/ray.sub"
}

echo "=== Submitting evals for step_1000 checkpoints ==="
echo ""

# 1. Distillation - MATH
submit_eval_job "$DISTILL_CKPT" "$DISTILL_HF" "$MATH_CONFIG" "math" "distill-fwd-kl-cosine"

# 2. Distillation - MMLU
submit_eval_job "$DISTILL_CKPT" "$DISTILL_HF" "$MMLU_CONFIG" "mmlu" "distill-fwd-kl-cosine"

# 3. SFT - MATH
submit_eval_job "$SFT_CKPT" "$SFT_HF" "$MATH_CONFIG" "math" "sft-arrow"

# 4. SFT - MMLU
submit_eval_job "$SFT_CKPT" "$SFT_HF" "$MMLU_CONFIG" "mmlu" "sft-arrow"

echo ""
echo "=== All 4 eval jobs submitted ==="
echo "Check logs at: ${REPO_ROOT}/llama-eval/"
