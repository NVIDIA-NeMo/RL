#!/bin/bash
# Launch step-level REINFORCE on the step_arithmetic NeMo-Gym env.
#
# Required env vars: NRL_ROOT, ENV_DIR, BASE_IMAGE, SLURM_ACCOUNT, HF_HOME.
# Optional: NUM_NODES, GPUS_PER_NODE, TP, VLLM_TP, VLLM_GPU_MEM, PPS, MODEL_NAME,
#           MAX_STEPS, WALLTIME, PARTITION, EXCLUDE_NODES, RUN_NAME, EXTRA_ARGS.
#
# Usage:
#   bash examples/research/launch_step_reinforce.sh
#   NUM_NODES=1 GPUS_PER_NODE=1 MAX_STEPS=20 bash examples/research/launch_step_reinforce.sh
set -euo pipefail

NRL_SOURCE="${NRL_ROOT:?set NRL_ROOT=/path/to/your/NeMo-RL/clone}"
ENV_DIR="${ENV_DIR:?set ENV_DIR=/path/to/nemo-rl-env-mcore}"
BASE_IMAGE="${BASE_IMAGE:?set BASE_IMAGE=<your container image>}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:?set SLURM_ACCOUNT=<your slurm account>}"
: "${HF_HOME:?set HF_HOME=/path/to/hf_cache (on a shared fs)}"

NUM_NODES=${NUM_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
PARTITION=${PARTITION:-batch}
WALLTIME=${WALLTIME:-1:59:59}
EXCLUDE_NODES=${EXCLUDE_NODES:-}

TP=${TP:-1}
VLLM_TP=${VLLM_TP:-${TP}}
VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.5}

PPS=${PPS:-8}
GBS=${GBS:-${PPS}}

CONFIG_FILE=${CONFIG_FILE:-examples/research/configs/step_reinforce_step_arithmetic.yaml}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}
MAX_STEPS=${MAX_STEPS:-100}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_NAME=${RUN_NAME:-step-reinforce-${TIMESTAMP}}
RUN_DIR=${RUN_DIR:-${NRL_SOURCE}/results/${RUN_NAME}}
LOG_DIR=${RUN_DIR}/logs
CKPT_DIR=${RUN_DIR}/checkpoints
mkdir -p "$LOG_DIR" "$CKPT_DIR"

JOB_NAME="${SLURM_ACCOUNT}-step_reinforce.${RUN_NAME}"

[ -f "${NRL_SOURCE}/.env" ] && source "${NRL_SOURCE}/.env"
: "${WANDB_API_KEY:=}"
: "${HF_TOKEN:=}"

MOUNTS="/lustre:/lustre"
MOUNTS="${MOUNTS},${ENV_DIR}:/opt/nemo-rl-env"
MOUNTS="${MOUNTS},${ENV_DIR}/venv:/opt/nemo_rl_venv"
MOUNTS="${MOUNTS},${ENV_DIR}/bin/uv:/usr/local/bin/uv"
MOUNTS="${MOUNTS},${NRL_SOURCE}:${NRL_SOURCE}"
MOUNTS="${MOUNTS},${HF_HOME}:/root/.cache/huggingface"
MOUNTS="${MOUNTS},${RUN_DIR}:${RUN_DIR}"
MOUNTS="${MOUNTS},${NRL_SOURCE}/3rdparty/Gym-workspace/Gym:${NRL_SOURCE}/3rdparty/Gym-workspace/Gym"

export CONTAINER="$BASE_IMAGE"
export MOUNTS
export GPUS_PER_NODE
export WANDB_API_KEY
export HF_TOKEN
export NEMO_RL_VENV_DIR=/opt/nemo-rl-env/ray_venvs
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

export COMMAND="cd ${NRL_SOURCE} && rm -rf .venv 2>/dev/null; \
export FLASHINFER_DISABLE_VERSION_CHECK=1 && \
export VLLM_USE_STANDALONE_COMPILE=0 && \
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv && \
export UV_NO_SYNC=1 && \
export UV_CACHE_DIR=/tmp/uv_cache_\$\$ && \
export PATH=/opt/nemo_rl_venv/bin:/opt/nemo-rl-env/bin:\$PATH && \
export PYTHONPATH=${NRL_SOURCE}:${NRL_SOURCE}/3rdparty/Megatron-LM-workspace/Megatron-LM:${NRL_SOURCE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NRL_SOURCE}/3rdparty/Gym-workspace/Gym:\${PYTHONPATH:-} && \
NRL_FORCE_REBUILD_VENVS=false RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run ./examples/research/run_step_reinforce.py --config ${CONFIG_FILE} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  cluster.num_nodes=${NUM_NODES} \
  grpo.num_prompts_per_step=${PPS} \
  grpo.num_generations_per_prompt=1 \
  grpo.max_num_steps=${MAX_STEPS} \
  policy.train_global_batch_size=${GBS} \
  policy.model_name=${MODEL_NAME} \
  policy.dtensor_cfg.enabled=false \
  policy.optimizer=null \
  policy.scheduler=null \
  policy.megatron_cfg.enabled=true \
  policy.megatron_cfg.tensor_model_parallel_size=${TP} \
  policy.megatron_cfg.pipeline_model_parallel_size=${PP:-1} \
  policy.megatron_cfg.expert_model_parallel_size=${EP:-1} \
  policy.megatron_cfg.context_parallel_size=1 \
  policy.generation.backend=vllm \
  policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
  policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_MEM} \
  policy.generation.colocated.enabled=true \
  checkpointing.checkpoint_dir=${CKPT_DIR} \
  checkpointing.enabled=false \
  logger.log_dir=${LOG_DIR} \
  logger.wandb_enabled=false \
  ${EXTRA_ARGS:-}"

echo "[step-reinforce] launching:"
echo "  run_dir   = ${RUN_DIR}"
echo "  nodes     = ${NUM_NODES}, gpus/node = ${GPUS_PER_NODE}, TP = ${TP}"
echo "  model     = ${MODEL_NAME}"
echo "  pps       = ${PPS}, max_steps = ${MAX_STEPS}"

sbatch \
  --nodes=${NUM_NODES} \
  --exclusive \
  --account=${SLURM_ACCOUNT} \
  --partition=${PARTITION} \
  --job-name=${JOB_NAME} \
  ${EXCLUDE_NODES:+--exclude=${EXCLUDE_NODES}} \
  --time=${WALLTIME} \
  --output="${LOG_DIR}/slurm-%j.out" \
  "${NRL_SOURCE}/ray.sub"
