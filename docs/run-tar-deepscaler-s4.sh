#!/usr/bin/env bash
set -euo pipefail

cd /home/scratch.shaunakj_other/Development/RL

MODE="${1:-train}"   # train | frozen | both

case "$MODE" in
  train|frozen|both) ;;
  *)
    echo "Usage: $0 [train|frozen|both]"
    exit 2
    ;;
esac

# Models
export TARGET="${TARGET:-/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
export DRAFT="${DRAFT:-/home/scratch.shaunakj_other/.cache/huggingface/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"

# Environment / cache paths
export HOME="${HOME:-/home/scratch.shaunakj_other}"
export TMPDIR="${TMPDIR:-/home/scratch.shaunakj_other/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/scratch.shaunakj_other/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/home/scratch.shaunakj_other/.cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/home/scratch.shaunakj_other/.cache/vllm}"
export HF_HOME="${HF_HOME:-/home/scratch.shaunakj_other/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/scratch.shaunakj_other/.cache/hf_json_cache}"

# Some hosts have a non-writable shared HF hub cache. Fall back to TMPDIR for dataset cache.
if [[ ! -w "${HF_HOME}" || ( -d "${HF_HOME}/hub" && ! -w "${HF_HOME}/hub" ) ]]; then
  export HF_HOME="${TMPDIR}/hf-home"
  export HF_DATASETS_CACHE="${TMPDIR}/hf-json-cache"
  echo "WARN: HF cache is not writable, using HF_HOME=${HF_HOME} HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
fi

export RAY_TMPDIR=/tmp/ray
export VLLM_LOG_STATS_INTERVAL=1
export PYTHONUNBUFFERED=1
export NRL_FORCE_LOCAL_RAY=true
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export VLLM_DISABLE_COMPILE_CACHE=1

# Budget/profile
export RUN_STEPS="${RUN_STEPS:-135}"
export ARM_TIMEOUT="${ARM_TIMEOUT:-230m}"
export SPEC_TOKENS="${SPEC_TOKENS:-4}"
export SEED="${SEED:-124}"

export TRAIN_LR="${TRAIN_LR:-1e-5}"
export FROZEN_LR="${FROZEN_LR:-0.0}"
export MAX_EPOCHS="${MAX_EPOCHS:-2}"
export PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-2}"
export GENERATIONS_PER_PROMPT="${GENERATIONS_PER_PROMPT:-4}"
export TEMP="${TEMP:-0.6}"

export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NUM_NODES="${NUM_NODES:-1}"

export EXP_TS="$(date +%F-%H%M%S)"
export EXP_ROOT="/home/scratch.shaunakj_other/logs/tar-drift-deepscaler-s${SPEC_TOKENS}-seed${SEED}-steps${RUN_STEPS}-${EXP_TS}"
export EXP_RESULTS_ROOT="/home/scratch.shaunakj_other/results/tar-drift-deepscaler-s${SPEC_TOKENS}-seed${SEED}-steps${RUN_STEPS}-${EXP_TS}"

mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" \
  "$HF_DATASETS_CACHE" "$RAY_TMPDIR" "$EXP_ROOT" "$EXP_RESULTS_ROOT"

export LOCAL_CFG="/home/scratch.shaunakj_other/tmp/grpo-qwen3-32b-spec-decode-lowbatch-1n8g.${EXP_TS}.deepscaler.local.yaml"
cp examples/configs/recipes/llm/grpo-qwen3-32b-spec-decode-lowbatch-1n8g.yaml "$LOCAL_CFG"
sed -i 's#^defaults: /opt/nemo-rl/examples/configs/grpo_math_1B.yaml#defaults: /home/scratch.shaunakj_other/Development/RL/examples/configs/grpo_math_1B.yaml#' "$LOCAL_CFG"

STATUS="$EXP_ROOT/status.log"
{
  echo "$(date '+%F %T %Z') START mode=${MODE} seed=${SEED} steps=${RUN_STEPS} spec_tokens=${SPEC_TOKENS} timeout=${ARM_TIMEOUT}"
  echo "$(date '+%F %T %Z') dataset=DeepScaler lr(train)=${TRAIN_LR} lr(frozen)=${FROZEN_LR}"
  echo "$(date '+%F %T %Z') max_epochs=${MAX_EPOCHS} prompts_per_step=${PROMPTS_PER_STEP} gens_per_prompt=${GENERATIONS_PER_PROMPT} temp=${TEMP}"
  echo "exp_root=$EXP_ROOT"
  echo "exp_results_root=$EXP_RESULTS_ROOT"
  echo "local_cfg=$LOCAL_CFG"
} | tee "$STATUS"

run_arm() {
  local arm="$1"
  local lr="$2"

  local run_tag="grpo-32b-deepscaler-spec0p6b-temp0p6-s${SPEC_TOKENS}-seed${SEED}-${arm}-steps${RUN_STEPS}-${EXP_TS}"
  local logroot="/home/scratch.shaunakj_other/logs/${run_tag}"
  local resroot="/home/scratch.shaunakj_other/results/${run_tag}"
  local run_log="${logroot}/run-${RUN_STEPS}steps-seed${SEED}-${arm}-deepscaler.log"

  mkdir -p "$logroot" "$resroot"
  echo "$(date '+%F %T %Z') seed=${SEED} arm=${arm} lr=${lr} logroot=${logroot}" | tee -a "$STATUS"

  .venv/bin/ray stop --force || true

  timeout "$ARM_TIMEOUT" stdbuf -oL -eL .venv/bin/python examples/run_grpo.py \
    --config "$LOCAL_CFG" \
    "++grpo.max_num_steps=${RUN_STEPS}" \
    "++grpo.max_num_epochs=${MAX_EPOCHS}" \
    "++grpo.val_period=0" \
    "++grpo.val_at_start=false" \
    "++grpo.val_at_end=false" \
    "++grpo.seed=${SEED}" \
    "++grpo.num_prompts_per_step=${PROMPTS_PER_STEP}" \
    "++grpo.num_generations_per_prompt=${GENERATIONS_PER_PROMPT}" \
    "++policy.dtensor_cfg.activation_checkpointing=true" \
    "++policy.model_name=${TARGET}" \
    "++policy.tokenizer.name=${TARGET}" \
    "++policy.max_total_sequence_length=5120" \
    "++policy.generation.max_new_tokens=4096" \
    "++policy.generation.temperature=${TEMP}" \
    "++policy.generation.top_p=1.0" \
    "++policy.generation.top_k=null" \
    "++policy.generation.vllm_cfg.load_format=auto" \
    "++policy.generation.vllm_cfg.max_model_len=5120" \
    "++policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN" \
    "++policy.generation.vllm_kwargs.speculative_config.model=${DRAFT}" \
    "++policy.generation.vllm_kwargs.speculative_config.method=draft_model" \
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${SPEC_TOKENS}" \
    "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1" \
    "++policy.optimizer.kwargs.lr=${lr}" \
    "++policy.max_grad_norm=1.0" \
    "++loss_fn.ratio_clip_min=0.2" \
    "++loss_fn.ratio_clip_max=0.2" \
    "++loss_fn.reference_policy_kl_penalty=0.01" \
    "++cluster.gpus_per_node=${GPUS_PER_NODE}" \
    "++cluster.num_nodes=${NUM_NODES}" \
    "++env.math.num_workers=1" \
    "++data.train.dataset_name=DeepScaler" \
    "++data.validation=null" \
    "++data.default.processor=math_hf_data_processor" \
    "++data.default.env_name=math" \
    "++data.default.prompt_file=examples/prompts/cot.txt" \
    "++logger.log_dir=${logroot}" \
    "++checkpointing.checkpoint_dir=${resroot}" \
    2>&1 | tee "$run_log"

  local rc=${PIPESTATUS[0]}
  echo "$(date '+%F %T %Z') seed=${SEED} arm=${arm} rc=${rc} run_log=${run_log}" | tee -a "$STATUS"
  return "$rc"
}

case "$MODE" in
  train)
    run_arm train "$TRAIN_LR"
    ;;
  frozen)
    run_arm frozen "$FROZEN_LR"
    ;;
  both)
    run_arm train "$TRAIN_LR"
    run_arm frozen "$FROZEN_LR"
    ;;
esac

echo "$(date '+%F %T %Z') DONE mode=${MODE} exp_root=${EXP_ROOT}" | tee -a "$STATUS"
