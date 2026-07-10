#!/bin/bash
# Driver script for 16-node TRT-LLM GRPO SWE 30B training.
# Run via attach script: COMMAND='bash /path/to/this' bash 12878975-attach.sh
# Output redirected to log file by the caller.
set -euo pipefail

source /lustre/fsw/portfolios/coreai/users/erinh/env.sh

cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/erinh/RL

REPO_ROOT="$(pwd)"
# WAR: Force this repo to win `import nemo_rl` over the container's baked /opt/nemo-rl copy
# (entrypoint lives in examples/nemo_gym so sys.path[0] != REPO_ROOT; without this the
# baked code shadows the mounted rebased tree). Propagates to Ray actors via env copy.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CONFIG_FILE="${REPO_ROOT}/examples/swe_bench/grpo_qwen3_30b_async_swe.yaml"
CHECKPOINT_ROOT="${REPO_ROOT}/results"
TRAIN_DATA_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl"
VAL_DATA_PATH="${TRAIN_DATA_PATH}"
DEFAULT_MODEL_PATH="/lustre/fsw/portfolios/coreai/users/bihu/repos/nemo-rl-async-swe/results/qwen3-30b-thinking-swe1-async-age1-pps64-gpp8-gbs512-lr1e-06/step_230_hf"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"

SKIP_TRAINING="${SKIP_TRAINING:-0}"
NUM_ACTOR_NODES=16
NUM_GENERATION_NODES=8
NUM_GPU=8

TP=4; EP=8; CP=4; PP=2
TRTLLM_TP=2
MIN_PAD=1
if [ ${CP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * CP * 2)); fi
if [ ${TP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * TP)); fi
MAKE_SEQ_DIVISIBLE_BY=${MIN_PAD}

SEQLEN=131072
SEQUENCE_PACKING=True

ASYNC_GRPO_ENABLED=True
MAX_TRAJECTORY_AGE_STEPS=1
FORCE_ON_POLICY_RATIO=True
INFLIGHT_WEIGHT_UPDATE=True
RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES=False
SEQ_LOGPROB_ERROR_THRESHOLD=null
COLOCATED_ENABLED=False
TRTLLM_GPU_UTIL="${TRTLLM_GPU_UTIL:-0.6}"   # was 0.8; lowered for refit NCCL-comm VRAM headroom. TODO: investigate TRTLLM mem regression.
OVERLAP_GRAD_REDUCE=False
ADVANTAGE_CLIP_LOW=-100
ADVANTAGE_CLIP_HIGH=100
TIS_THRESHOLD=5

PPS=8; GPP=8; GBS=64
NORMALIZE_REWARDS=True; OVERLONG_FILTERING=True

KL=0; CLIP_MIN=0.2; CLIP_MAX=0.28
USE_ON_POLICY_KL_APPROXIMATION=True; IMPORTANCE_SAMPLING_CORRECTION=True
SEQ_LEVEL_IS=False; TOKEN_LEVEL_LOSS=True

LR="1e-06"

MOE_FREEZE_ROUTER=True; MOE_PERMUTE_FUSION=True; MOE_ENABLE_DEEPEP=False
MOE_TOKEN_DISPATCHER_TYPE="alltoall"; MOE_AUX_LOSS_COEFF=0
MOE_ROUTER_LOAD_BALANCING_TYPE="none"; MOE_ROUTER_BIAS_UPDATE_RATE="1e-3"

TEMPERATURE=1.0
SAVE_PERIOD=5; VAL_PERIOD=1000; KEEP_TOP_K=2
AGENT_MAX_TURNS=200; AGENT_TIMEOUT=1800

WANDB_PROJ="swe-benchmark-30b-erinh"
WANDB_GROUP="${WANDB_GROUP:-swe-repro-trtllm}"

SYNC_MODE="async-age${MAX_TRAJECTORY_AGE_STEPS}"
EXP_SUFFIX="${EXP_SUFFIX:-repro-baseline-swe2-trtllm-${SYNC_MODE}-pps${PPS}-gpp${GPP}-gbs${GBS}-lr${LR}-tp${TP}-eos-fix}"
WANDB_NAME="${EXP_SUFFIX}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${EXP_SUFFIX}"
mkdir -p "${CHECKPOINT_DIR}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export UV_CACHE_DIR=/tmp/uv_cache
export UV_LOCK_TIMEOUT=3600
export RAY_DEDUP_LOGS=1

PERSISTENT_CACHE="${HOME}/.cache/qwen3_30b_thinking_swe_repro_trtllm"
export LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
export LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
export INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
export TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
export CACHE_SYNC_FREQUENCY=120

NRL_WG_USE_RAY_REF=1 \
  WANDB_API_KEY=${WANDB_API_KEY} \
  HF_HOME=${HF_HOME} \
  HF_DATASETS_CACHE=${HF_DATASETS_CACHE} \
  UV_CACHE_DIR=${UV_CACHE_DIR} \
  NRL_FORCE_REBUILD_VENVS=false \
  NRL_IGNORE_VERSION_MISMATCH=1 \
  RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
  UV_HTTP_TIMEOUT=3600 \
  UV_LOCK_TIMEOUT=900 \
  TORCH_CUDA_ARCH_LIST='9.0 10.0' \
  NEMO_GYM_SKIP_VENV_IF_PRESENT=1 \
  NEMO_GYM_VENV_DIR="${REPO_ROOT}/gym_venvs" \
  OMPI_MCA_plm=^slurm \
  OMPI_MCA_btl=^openib \
  uv run --frozen --no-sync --extra mcore ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config=${CONFIG_FILE} \
  cluster.num_nodes=${NUM_ACTOR_NODES} \
  cluster.gpus_per_node=${NUM_GPU} \
  ++data.train.data_path=${TRAIN_DATA_PATH} \
  ++data.validation.data_path=${VAL_DATA_PATH} \
  grpo.num_prompts_per_step=${PPS} \
  grpo.num_generations_per_prompt=${GPP} \
  grpo.val_at_start=False \
  grpo.normalize_rewards=${NORMALIZE_REWARDS} \
  grpo.overlong_filtering=${OVERLONG_FILTERING} \
  grpo.val_period=${VAL_PERIOD} \
  grpo.seq_logprob_error_threshold=${SEQ_LOGPROB_ERROR_THRESHOLD} \
  grpo.async_grpo.enabled=${ASYNC_GRPO_ENABLED} \
  grpo.async_grpo.in_flight_weight_updates=${INFLIGHT_WEIGHT_UPDATE} \
  grpo.async_grpo.recompute_kv_cache_after_weight_updates=${RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES} \
  grpo.async_grpo.max_trajectory_age_steps=${MAX_TRAJECTORY_AGE_STEPS} \
  env.should_log_nemo_gym_responses=true \
  policy.generation.colocated.enabled=${COLOCATED_ENABLED} \
  policy.model_name=${MODEL_PATH} \
  policy.max_total_sequence_length=${SEQLEN} \
  policy.dynamic_batching.enabled=False \
  policy.train_global_batch_size=${GBS} \
  policy.make_sequence_length_divisible_by=${MAKE_SEQ_DIVISIBLE_BY} \
  policy.offload_optimizer_for_logprob=true \
  policy.sequence_packing.enabled=${SEQUENCE_PACKING} \
  policy.megatron_cfg.tensor_model_parallel_size=${TP} \
  policy.megatron_cfg.expert_model_parallel_size=${EP} \
  policy.megatron_cfg.context_parallel_size=${CP} \
  policy.megatron_cfg.pipeline_model_parallel_size=${PP} \
  policy.megatron_cfg.sequence_parallel=True \
  policy.megatron_cfg.bias_activation_fusion=False \
  ++policy.megatron_cfg.use_fused_weighted_squared_relu=False \
  policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=${OVERLAP_GRAD_REDUCE} \
  policy.megatron_cfg.moe_permute_fusion=${MOE_PERMUTE_FUSION} \
  policy.megatron_cfg.moe_enable_deepep=${MOE_ENABLE_DEEPEP} \
  policy.megatron_cfg.moe_token_dispatcher_type=${MOE_TOKEN_DISPATCHER_TYPE} \
  policy.megatron_cfg.moe_aux_loss_coeff=${MOE_AUX_LOSS_COEFF} \
  policy.megatron_cfg.moe_router_load_balancing_type=${MOE_ROUTER_LOAD_BALANCING_TYPE} \
  policy.megatron_cfg.moe_router_bias_update_rate=${MOE_ROUTER_BIAS_UPDATE_RATE} \
  policy.megatron_cfg.freeze_moe_router=${MOE_FREEZE_ROUTER} \
  policy.megatron_cfg.optimizer.lr=${LR} \
  policy.megatron_cfg.optimizer.min_lr=${LR} \
  policy.megatron_cfg.optimizer.weight_decay=0 \
  policy.megatron_cfg.empty_unused_memory_level=2 \
  policy.megatron_cfg.activation_checkpointing=True \
  policy.generation.temperature=${TEMPERATURE} \
  policy.generation.backend=trtllm \
  policy.generation.trtllm_cfg.tensor_parallel_size=${TRTLLM_TP} \
  policy.generation.trtllm_cfg.gpu_memory_utilization=${TRTLLM_GPU_UTIL} \
  policy.generation.trtllm_cfg.max_model_len=${SEQLEN} \
  ++policy.generation.trtllm_cfg.max_num_tokens=${SEQLEN} \
  policy.generation.trtllm_cfg.async_engine=true \
  policy.generation.trtllm_cfg.in_flight_weight_updates=${INFLIGHT_WEIGHT_UPDATE} \
  policy.generation.trtllm_cfg.recompute_kv_cache_after_weight_updates=${RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES} \
  loss_fn.reference_policy_kl_penalty=${KL} \
  loss_fn.ratio_clip_min=${CLIP_MIN} \
  loss_fn.ratio_clip_max=${CLIP_MAX} \
  loss_fn.use_on_policy_kl_approximation=${USE_ON_POLICY_KL_APPROXIMATION} \
  loss_fn.use_importance_sampling_correction=${IMPORTANCE_SAMPLING_CORRECTION} \
  loss_fn.sequence_level_importance_ratios=${SEQ_LEVEL_IS} \
  loss_fn.token_level_loss=${TOKEN_LEVEL_LOSS} \
  loss_fn.force_on_policy_ratio=${FORCE_ON_POLICY_RATIO} \
  checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
  checkpointing.save_period=${SAVE_PERIOD} \
  checkpointing.keep_top_k=${KEEP_TOP_K} \
  ++checkpointing.metric_name=train:total_reward/mean \
  ++checkpointing.checkpoint_must_save_by=00:03:35:00 \
  logger.wandb_enabled=True \
  logger.wandb.name=${WANDB_NAME} \
  logger.wandb.project=${WANDB_PROJ} \
  ++logger.wandb.group=${WANDB_GROUP} \
  policy.generation.colocated.resources.num_nodes=${NUM_GENERATION_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${NUM_GPU} \
  grpo.advantage_clip_low=${ADVANTAGE_CLIP_LOW} \
  grpo.advantage_clip_high=${ADVANTAGE_CLIP_HIGH} \
  loss_fn.truncated_importance_sampling_ratio=${TIS_THRESHOLD} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT}
