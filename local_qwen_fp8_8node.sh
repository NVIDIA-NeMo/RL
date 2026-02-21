source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh
export HF_HOME=/lustre/fsw/portfolios/coreai/users/zhiyul/hf

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export FP8=${FP8:-false}
export FORCE_CLEAR_FP8_CACHES=${FORCE_CLEAR_FP8_CACHES:-false}
export EXPANDABLE=${EXPANDABLE:-false}

if [ "$FP8" = true ]; then
  FLAG="policy.generation.vllm_cfg.precision=fp8 policy.megatron_cfg.fp8_cfg.enabled=true policy.generation.vllm_cfg.gpu_memory_utilization=0.5"
  EXPNAME="_fp8"
  if [ "$FORCE_CLEAR_FP8_CACHES" = true ]; then
    FLAG="${FLAG} ++policy.megatron_cfg.fp8_cfg.force_clear_fp8_caches=true"
    EXPNAME="${EXPNAME}_forceClearFP8Caches"
  fi
  if [ "$EXPANDABLE" = true ]; then
    FLAG="${FLAG} policy.megatron_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'"
    EXPNAME="${EXPNAME}_expandableTrue"
  else
    FLAG="${FLAG} policy.megatron_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False'"
  fi
else
  FLAG="policy.generation.vllm_cfg.gpu_memory_utilization=0.7"  # vllm need more memory to load bf16 weights
  EXPNAME="_bf16"
fi

uv run examples/run_grpo_math.py \
  --config examples/configs/grpo_math_qwen30ba3b_base.yaml \
  grpo.max_num_steps=5 \
  logger.wandb_enabled=true \
  logger.wandb.project=grpo-dev-zhiyul \
  logger.wandb.name=memory_debug_grpo_math_qwen30ba3b_8node${EXPNAME} \
  checkpointing.enabled=false \
  policy.max_total_sequence_length=6144 \
  cluster.num_nodes=8 \
  ${FLAG} \