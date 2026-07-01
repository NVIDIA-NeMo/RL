#!/bin/bash
# ============================================================================
# NeMo-RL Async GRPO SWE2 — Qwen3-30B-A3B-Thinking — PURE RUN (GB200, 8 nodes)
#
# Placement: 8 GB200 nodes (4 GPU/node) = 4 training + 4 rollout, non-colocated async.
#   training one replica = TP*CP*PP = 4*2*2 = 16 GPU = 4 nodes (DP=1)
#   rollout = 4 nodes / 16 GPU, vLLM TP=2 -> 8 replicas
#
# This script ONLY runs the training driver. It assumes the Ray cluster + the
# NeMo-RL container are ALREADY up and that per-node env / apptainer / cache
# seeding were done by node_init_script_nemorl.sh. Bring the cluster up with:
#
#   g8nv                      # = start_container.sh --fw nemorl --engine vllm --nodes 8
#   bash <run-dir>/<jobid>-attach.sh        # attach into the Ray head container
#   bash run_grpo_qwen3_30b_thinking_swe2_gb200_8node.sh
#
# Overrides: MODEL_PATH=/path/to/swe1_hf  bash run_grpo_...gb200_8node.sh
# ============================================================================
set -e

REPO_ROOT="/lustre/fsw/coreai_comparch_trtllm/erinh/RL"
CONFIG_FILE="${REPO_ROOT}/grpo_qwen3_30b_async_swe.yaml"
CHECKPOINT_ROOT="${REPO_ROOT}/results"

# ================ env ================
# Run this from INSIDE the attached enroot container. This script does NOT set env —
# it assumes your container session already has it. node_init_script_nemorl.sh sets it on
# every node at cluster bringup (g8nv); if your interactive shell is missing tokens/flags,
# source them yourself, e.g.:
#   source /lustre/fsw/coreai_comparch_trtllm/erinh/env.sh   # WANDB / HF / GIT tokens
# Driver needs at least: WANDB_API_KEY, HF token + HF_HOME, NRL_IGNORE_VERSION_MISMATCH=1,
# NEMO_GYM_SKIP_VENV_IF_PRESENT=1, RAY_ENABLE_UV_RUN_RUNTIME_ENV=0.

# ================ Scaling 核心参数 ================
PPS=2          # GB200 8-node smoke: small steps
GPP=4
GBS=8
LR="1e-06"
AGENT_MAX_TURNS=50         # smoke: shorter trajectories
AGENT_TIMEOUT=600

# ================ Sync/Async ================
ASYNC_GRPO_ENABLED=True
MAX_TRAJECTORY_AGE_STEPS=1
NUM_ACTOR_NODES=${NUM_NODES:-8}            # TOTAL ray-cluster nodes
FORCE_ON_POLICY_RATIO=True
INFLIGHT_WEIGHT_UPDATE=True
RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES=False
SEQ_LOGPROB_ERROR_THRESHOLD=null

if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  COLOCATED_ENABLED=False
  VLLM_GPU_UTIL=0.8
  NUM_GENERATION_NODES=${NUM_GEN_NODES:-4}  # rollout pool = 4 nodes / 16 GPU
  OVERLAP_GRAD_REDUCE=False
  ADVANTAGE_CLIP_LOW=-100
  ADVANTAGE_CLIP_HIGH=100
  TIS_THRESHOLD=5
else
  COLOCATED_ENABLED=True
  VLLM_GPU_UTIL=0.5
  OVERLAP_GRAD_REDUCE=True
fi

# ================ 固定参数 ================
SEQLEN=131072
NUM_GPU=4                                  # GB200 = 4 GPU/node

TP=4
EP=8
CP=2           # GB200: one replica TP*CP*PP = 4*2*2 = 16 GPU = 4 train nodes
PP=2
VLLM_TP=2

SEQUENCE_PACKING=True
TOKEN_LEVEL_LOSS=True

# make_sequence_length_divisible_by = (cp*2)*tp when CP>1 and TP>1+SP
MIN_PAD=1
if [ ${CP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * CP * 2)); fi
if [ ${TP} -gt 1 ]; then MIN_PAD=$((MIN_PAD * TP)); fi
MAKE_SEQ_DIVISIBLE_BY=${MIN_PAD}
SEQ_LEVEL_IS=False
NORMALIZE_REWARDS=True
OVERLONG_FILTERING=True

USE_ON_POLICY_KL_APPROXIMATION=True
IMPORTANCE_SAMPLING_CORRECTION=True
KL=0
CLIP_MIN=0.2
CLIP_MAX=0.28
TEMPERATURE=1.0

SAVE_PERIOD=5
VAL_PERIOD=1000
KEEP_TOP_K=2

MOE_FREEZE_ROUTER=True
MOE_PERMUTE_FUSION=True
MOE_ENABLE_DEEPEP=False
MOE_TOKEN_DISPATCHER_TYPE="alltoall"
MOE_AUX_LOSS_COEFF=0
MOE_ROUTER_LOAD_BALANCING_TYPE="none"
MOE_ROUTER_BIAS_UPDATE_RATE="1e-3"

# ================ 数据/模型路径 (SET THESE for a real run) ================
TRAIN_DATA_PATH="/lustre/fsw/coreai_comparch_trtllm/erinh/datasets/r2e_gym_train_with_sifs.jsonl"  # R2E-Gym subset, filtered to instances with validated arm64 sifs (see datasets/make_r2e_subset.sh)
VAL_DATA_PATH="${TRAIN_DATA_PATH}"
# Smoke default = base model. Override with MODEL_PATH=/path/to/swe1_hf for a real SWE2 run.
MODEL_PATH=${MODEL_PATH:-"/lustre/fsw/coreai_comparch_trtllm/erinh/llm-models/Qwen/Qwen3-30B-A3B-Thinking-2507"}

# ================ 实验命名 ================
WANDB_PROJ="nemo-rl-agentic-erinh"
if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then SYNC_MODE="async-age${MAX_TRAJECTORY_AGE_STEPS}"; else SYNC_MODE="sync"; fi
EXP_SUFFIX="gb200-8node-vllm-qwen3-30b-a3b-thinking-swe2-${SYNC_MODE}-pps${PPS}-gpp${GPP}-gbs${GBS}-lr${LR}"
WANDB_NAME="${EXP_SUFFIX}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${EXP_SUFFIX}"
mkdir -p "${CHECKPOINT_DIR}"

# ================ 训练命令 (no env prefix — env comes from node_init above) ================
COMMAND="uv run --frozen ./examples/nemo_gym/run_grpo_nemo_gym.py \
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
  policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
  policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_UTIL} \
  policy.generation.vllm_cfg.skip_tokenizer_init=False \
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
  logger.wandb_enabled=True \
  logger.wandb.name=${WANDB_NAME} \
  logger.wandb.project=${WANDB_PROJ}"

if [ "${ASYNC_GRPO_ENABLED}" = "True" ]; then
  COMMAND="${COMMAND} \
  policy.generation.colocated.resources.num_nodes=${NUM_GENERATION_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${NUM_GPU} \
  grpo.advantage_clip_low=${ADVANTAGE_CLIP_LOW} \
  grpo.advantage_clip_high=${ADVANTAGE_CLIP_HIGH} \
  loss_fn.truncated_importance_sampling_ratio=${TIS_THRESHOLD} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT}"
fi

# ================ RUN (Ray cluster already up; just connect + drive) ================
cd "${REPO_ROOT}"
echo "=========================================="
echo "Pure-run against existing Ray cluster"
echo "  nodes=${NUM_ACTOR_NODES} (train $((NUM_ACTOR_NODES - NUM_GENERATION_NODES)) + rollout ${NUM_GENERATION_NODES}), GPUs/node=${NUM_GPU}"
echo "  TP=${TP} CP=${CP} PP=${PP} EP=${EP} | vLLM_TP=${VLLM_TP} | seqlen=${SEQLEN}"
echo "  PPS=${PPS} GPP=${GPP} GBS=${GBS} | agent: turns=${AGENT_MAX_TURNS} timeout=${AGENT_TIMEOUT}s"
echo "  model=${MODEL_PATH}"
echo "  checkpoint=${CHECKPOINT_DIR}"
echo "=========================================="
eval "${COMMAND}"
