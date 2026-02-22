#!/bin/bash
set -euo pipefail

# =============================================================================
# Example: 21-node Super V3 GRPO launch with all cache optimizations enabled.
#
# This script demonstrates persistent caching, CUDA graph tuning, and
# venv-skip optimizations that significantly reduce initialization time.
#
# See README.md for documentation on each environment variable.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAIN_REPO_ROOT="${MAIN_REPO_ROOT:-$(git -C "${WORKTREE_ROOT}" worktree list --porcelain | awk '/^worktree /{print $2}' | grep -v '/.worktrees/' | head -n1)}"

if [[ -z "${MAIN_REPO_ROOT}" || ! -d "${MAIN_REPO_ROOT}" ]]; then
  echo "Could not resolve MAIN_REPO_ROOT; set MAIN_REPO_ROOT explicitly."
  exit 1
fi

if [[ ! -f "${MAIN_REPO_ROOT}/3rdparty/vllm/nemo-rl.env" ]]; then
  echo "Missing main vLLM env file: ${MAIN_REPO_ROOT}/3rdparty/vllm/nemo-rl.env"
  exit 1
fi

# Ensure required worktree submodules are present before submitting job.
MISSING=0
for p in \
  "${WORKTREE_ROOT}/3rdparty/Gym-workspace/Gym/nemo_gym/cli.py" \
  "${WORKTREE_ROOT}/3rdparty/Megatron-LM-workspace/Megatron-LM" \
  "${WORKTREE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" \
  "${WORKTREE_ROOT}/3rdparty/Automodel-workspace/Automodel"
do
  if [[ ! -e "${p}" ]]; then
    echo "Missing required worktree path: ${p}"
    MISSING=1
  fi
done
if [[ "${MISSING}" -ne 0 ]]; then
  echo "Initialize submodules on login node first:"
  echo "  git -C ${WORKTREE_ROOT} submodule update --init --recursive"
  exit 1
fi

#### User-configurable credentials (override via env) ####
WANDB_PROJ="${WANDB_PROJ:-}"          # set via env or edit here
WANDB_NAME="${WANDB_NAME:-}"          # set via env or edit here
export WANDB_API_KEY="${WANDB_API_KEY:-}"   # set via env or edit here
export HF_HOME="${HF_HOME:-}"               # set via env or edit here
export HF_HUB_CACHE="${HF_HUB_CACHE:-}"     # set via env or edit here
SLURM_ACCOUNT="${SLURM_ACCOUNT:-llmservice_nemotron_super}"

# ---------- Precision recipe ----------
PRECISION_RECIPE=${1:-"bf16"} # choices: bf16, mxfp8-e2e, mxfp8-rollout, mxfp8-train
DISABLE_FP8_LINEAR=0 # choices: 0, 1

MXFP8_GEN_EXTRA_ARGS="policy.generation.vllm_cfg.precision=\"fp8\" \
+policy.generation.vllm_cfg.fp8_cfg.is_mx=true \
+policy.generation.vllm_cfg.fp8_cfg.dynamic_weight_quant=false"

if [ "$DISABLE_FP8_LINEAR" == "1" ]; then
MXFP8_GEN_EXTRA_ARGS="$MXFP8_GEN_EXTRA_ARGS +policy.generation.vllm_cfg.quantization_ignored_layer_kws=[\"conv1d\",\"in_proj\",\"out_proj\",\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"fc1_latent_proj\",\"fc2_latent_proj\",\"shared_experts\",\"mtp\"]"
else
MXFP8_GEN_EXTRA_ARGS="$MXFP8_GEN_EXTRA_ARGS +policy.generation.vllm_cfg.quantization_ignored_layer_kws=[\"conv1d\",\"mtp\"]"
fi

MXFP8_TRAIN_EXTRA_ARGS="+policy.megatron_cfg.fp8_cfg.enabled=true \
+policy.megatron_cfg.fp8_cfg.fp8=\"e4m3\" \
+policy.megatron_cfg.fp8_cfg.fp8_recipe=\"mxfp8\" \
+policy.megatron_cfg.fp8_cfg.fp8_param=false \
policy.megatron_cfg.moe_router_dtype=fp32"

if [ "$PRECISION_RECIPE" == "mxfp8-rollout" ]; then
EXTRA_ARGS="$MXFP8_GEN_EXTRA_ARGS"
elif [ "$PRECISION_RECIPE" == "mxfp8-train" ]; then
EXTRA_ARGS="$MXFP8_TRAIN_EXTRA_ARGS"
elif [ "$PRECISION_RECIPE" == "mxfp8-e2e" ]; then
EXTRA_ARGS="$MXFP8_GEN_EXTRA_ARGS $MXFP8_TRAIN_EXTRA_ARGS"
elif [ "$PRECISION_RECIPE" == "bf16" ]; then
EXTRA_ARGS=""
else
    echo "Invalid recipe: $PRECISION_RECIPE"
    exit 1
fi

# ---------- Model parallelism ----------
MAX_LENGTH="${MAX_LENGTH:-49152}"
CP="${CP:-8}"
TP="${TP:-4}"
EP="${EP:-16}"
PP="${PP:-1}"
VLLM_TP="${VLLM_TP:-4}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.6}"
LBS="${LBS:-1}"

# ---------- GRPO ----------
PPS="${PPS:-128}"
GPP="${GPP:-8}"
GBS="${GBS:-1024}"
VAL_PERIOD="${VAL_PERIOD:-10}"
SAVE_PERIOD="${SAVE_PERIOD:-10}"

# ---------- Async ----------
ASYNC_GRPO="${ASYNC_GRPO:-True}"
MAX_TRAJECTORY_AGE_STEPS="${MAX_TRAJECTORY_AGE_STEPS:-1}"
IN_FLIGHT_WEIGHT_UPDATES="${IN_FLIGHT_WEIGHT_UPDATES:-True}"
RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES="${RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES:-False}"
COLOCATED_INFERENCE="${COLOCATED_INFERENCE:-False}"
INFERENCE_NUM_NODES="${INFERENCE_NUM_NODES:-8}"

# ---------- Node allocation ----------
NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-16}"
GENRM_MODEL_PATH="${GENRM_MODEL_PATH:-/scratch/fsw/portfolios/llmservice/users/jiaqiz/models/qwen235b_principle_comparison_genrm_step1230}"
GENRM_ROUTER_DP_SIZE="${GENRM_ROUTER_DP_SIZE:-2}"
NL2BASH_JUDGE_ROUTER_DP_SIZE="${NL2BASH_JUDGE_ROUTER_DP_SIZE:-2}"
GENRM_TP="${GENRM_TP:-8}"

NUM_GENRM_NODES="${NUM_GENRM_NODES:-2}"
NUM_LLMJUDGE_NODES="${NUM_LLMJUDGE_NODES:-3}"
NUM_JUDGE_NODES=$((NUM_GENRM_NODES + NUM_LLMJUDGE_NODES))
NUM_TOTAL_NODES=$((NUM_ACTOR_NODES + NUM_JUDGE_NODES))

# ---------- Loss ----------
TIS_RATIO="${TIS_RATIO:-5.0}"

# ---------- Paths ----------
TRAIN_PATH="${TRAIN_PATH:-/scratch/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v24_succinct-swift.train.jsonl}"
VAL_PATH="${VAL_PATH:-/scratch/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v24_succinct-swift.val.jsonl}"
MODEL_PATH="${MODEL_PATH:-/scratch/fsw/portfolios/llmservice/projects/llmservice_nemotron_super/users/liding/nemo/workspace/experiments/super_v3/agent-fixed-repeated-mtp-reinit-embed/iter_0000700}"

EXP_SUFFIX="${EXP_SUFFIX:-super-v3-grpo-cache-optimized}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/${EXP_SUFFIX}}"
mkdir -p "${CHECKPOINT_DIR}"

# ==========================================================================
# Persistent cache directories on shared filesystem
# ==========================================================================
PERSISTENT_CACHE="/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/ykarnati/.cache"
VLLM_CACHE_DIR="${PERSISTENT_CACHE}/vllm_compile_cache"
FLASHINFER_CUBIN_CACHE="${PERSISTENT_CACHE}/flashinfer_cubins"
FLASHINFER_WS_BASE="${PERSISTENT_CACHE}/flashinfer_workspace"
mkdir -p "${VLLM_CACHE_DIR}" "${FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}"

export OMP_NUM_THREADS=16
export RAY_DEDUP_LOGS=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0-cp38-abi3-manylinux_2_31_x86_64.whl}"

echo "Worktree root: ${WORKTREE_ROOT}"
echo "Main repo vLLM path: ${MAIN_REPO_ROOT}/3rdparty/vllm"
echo "Nodes: ${NUM_TOTAL_NODES} (actor=${NUM_ACTOR_NODES}, inference=${INFERENCE_NUM_NODES}, judge=${NUM_JUDGE_NODES})"
echo "Precision recipe: ${PRECISION_RECIPE}"
echo "Persistent cache root: ${PERSISTENT_CACHE}"
echo "VLLM_CACHE_ROOT: ${VLLM_CACHE_DIR}"
echo "FLASHINFER_CUBIN_DIR: ${FLASHINFER_CUBIN_CACHE}"
echo "FLASHINFER_WORKSPACE_BASE: ${FLASHINFER_WS_BASE}"

export COMMAND="cd ${WORKTREE_ROOT} && date ; \
source ${MAIN_REPO_ROOT}/3rdparty/vllm/nemo-rl.env && \
UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py && \
NRL_VLLM_USE_V1=1 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
VLLM_CACHE_ROOT=${VLLM_CACHE_DIR} \
DG_JIT_CACHE_DIR=${VLLM_CACHE_DIR}/deep_gemm \
UV_CACHE_DIR=${PERSISTENT_CACHE}/uv \
NEMO_GYM_SKIP_VENV_IF_PRESENT=1 \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
UV_HTTP_TIMEOUT=10 \
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION} \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_CACHE} \
FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WS_BASE} \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config examples/configs/grpo_superv3.yaml \
policy.model_name=${MODEL_PATH} \
cluster.gpus_per_node=8 \
cluster.num_nodes=${NUM_TOTAL_NODES} \
grpo.val_period=${VAL_PERIOD} \
checkpointing.save_period=${SAVE_PERIOD} \
grpo.num_prompts_per_step=${PPS} \
grpo.num_generations_per_prompt=${GPP} \
grpo.async_grpo.enabled=${ASYNC_GRPO} \
grpo.async_grpo.max_trajectory_age_steps=${MAX_TRAJECTORY_AGE_STEPS} \
grpo.async_grpo.in_flight_weight_updates=${IN_FLIGHT_WEIGHT_UPDATES} \
grpo.async_grpo.recompute_kv_cache_after_weight_updates=${RECOMPUTE_KV_CACHE_AFTER_WEIGHT_UPDATES} \
loss_fn.truncated_importance_sampling_ratio=${TIS_RATIO} \
policy.generation.colocated.enabled=${COLOCATED_INFERENCE} \
policy.generation.colocated.resources.num_nodes=${INFERENCE_NUM_NODES} \
policy.generation.colocated.resources.gpus_per_node=8 \
env.nemo_gym.num_gpu_nodes=${NUM_JUDGE_NODES} \
env.nemo_gym.genrm_model.responses_api_models.vllm_model.model=${GENRM_MODEL_PATH} \
env.nemo_gym.genrm_model.responses_api_models.vllm_model.router_dp_size=${GENRM_ROUTER_DP_SIZE} \
env.nemo_gym.nl2bash_judge_model.responses_api_models.vllm_model.router_dp_size=${NL2BASH_JUDGE_ROUTER_DP_SIZE} \
++env.nemo_gym.safety_judge_model.responses_api_models.vllm_model.server_args.max_num_seqs=16 \
++env.nemo_gym.safety_judge_model.responses_api_models.vllm_model.server_args.compilation_config.cudagraph_capture_sizes='[1,2,4,8,16]' \
++env.nemo_gym.genrm_model.responses_api_models.vllm_model.server_args.max_num_seqs=16 \
++env.nemo_gym.genrm_model.responses_api_models.vllm_model.server_args.compilation_config.cudagraph_capture_sizes='[1,2,4,8,16]' \
++env.nemo_gym.nl2bash_judge_model.responses_api_models.vllm_model.server_args.max_num_seqs=16 \
++env.nemo_gym.nl2bash_judge_model.responses_api_models.vllm_model.server_args.compilation_config.cudagraph_capture_sizes='[1,2,4,8,16]' \
env.nemo_gym.genrm_model.responses_api_models.vllm_model.server_args.tensor_parallel_size=${GENRM_TP} \
env.nemo_gym.genrm_compare.resources_servers.genrm_compare.group_reasoning_length_penalty_coeff=0 \
env.nemo_gym.genrm_compare.resources_servers.genrm_compare.group_answer_length_penalty_coeff=0 \
policy.megatron_cfg.sequence_parallel=True \
policy.megatron_cfg.context_parallel_size=${CP} \
policy.megatron_cfg.tensor_model_parallel_size=${TP} \
data.train_jsonl_fpath=${TRAIN_PATH} \
data.validation_jsonl_fpath=${VAL_PATH} \
policy.megatron_cfg.expert_tensor_parallel_size=1 \
policy.megatron_cfg.expert_model_parallel_size=${EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${PP} \
policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_UTIL} \
++policy.generation.vllm_kwargs.max_num_seqs=16 \
++policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes='[1,2,4,8,16]' \
policy.sequence_packing.enabled=True \
++policy.generation.vllm_cfg.skip_tokenizer_init=False \
policy.megatron_cfg.bias_activation_fusion=False \
policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=False \
logger.wandb_enabled=True \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ} \
++policy.generation.vllm_kwargs.mamba_ssm_cache_dtype=float32 \
policy.megatron_cfg.moe_permute_fusion=True \
policy.megatron_cfg.defer_fp32_logits=True \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
logger.log_dir=${CHECKPOINT_DIR}/logs \
policy.logprob_batch_size=${LBS} \
policy.train_global_batch_size=${GBS} \
policy.megatron_cfg.activation_checkpointing=True \
policy.max_total_sequence_length=${MAX_LENGTH} \
${EXTRA_ARGS}"

export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/guyueh/container_images/superv3-260130.sqsh}"
export MOUNTS="/scratch:/scratch,\
/lustre:/lustre:ro,\
${WORKTREE_ROOT}:${WORKTREE_ROOT},\
${WORKTREE_ROOT}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym,\
${WORKTREE_ROOT}/3rdparty/Megatron-LM-workspace/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM,\
${MAIN_REPO_ROOT}/3rdparty/vllm:/opt/nemo-rl/3rdparty/vllm"

sbatch \
  --nodes="${NUM_TOTAL_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${WANDB_NAME}" \
  --partition=batch \
  --time=4:0:0 \
  --gres=gpu:8 \
  --exclusive \
  --dependency=singleton \
  "${WORKTREE_ROOT}/ray.sub"
