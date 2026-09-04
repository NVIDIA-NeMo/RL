#!/bin/bash
# Run from the root of the NeMo RL repo.
#
# Qwen3-32B debug / A/B perf script (4-node 4-GPU-per-node).
# Derived from scripts/run_deepseek.sh for quick iteration.
#
# Usage:
#   bash ./scripts/run_qwen3.sh                  # default: 4n4g, 3 steps, no wandb
#   MAX_STEPS=8 ENABLE_WANDB=1 bash ./scripts/run_qwen3.sh

set -euo pipefail

# Site-specific root: one path, visible on every node, holding the container
# image, the HF cache and the mcore checkpoint cache. No default on purpose --
# a wrong shared path is a multi-node failure that only shows up at init.
WORK_DIR=${WORK_DIR:?set WORK_DIR to a shared path visible on every node}

SLURM_ACCOUNT=${SLURM_ACCOUNT:?set SLURM_ACCOUNT to your Slurm account}
NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
test_case=${test_case:-grpo-qwen3-32b-4n4g}

# ray.sub pins `#SBATCH --dependency=singleton`, so jobs sharing a name run one
# at a time. Override JOB_NAME to run submissions in parallel, e.g.
#   JOB_NAME=qwen3-nsys NSYS=1 bash ./scripts/run_qwen3.sh
JOB_NAME=${JOB_NAME:-${SLURM_ACCOUNT}-qwen3}

MAX_STEPS=${MAX_STEPS:-3}
ENABLE_WANDB=${ENABLE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-async-grpo-perfscript-test}

# Per-DP rollout batch/load stats for perf analysis (see VllmGeneration._log_dp_batch_stats).
# Set LOG_DP_BATCH_STATS=1 to emit "[DP-BATCH-STATS]" lines per generate() call.
LOG_DP_BATCH_STATS=${LOG_DP_BATCH_STATS:-0}

# Live per-DP in-flight batch timeline (batch size + per-seq context length over
# time), written to JSONL for offline plotting. See vllm/inflight_profiler.py.
# PROFILE_INFLIGHT=1 to enable; INTERVAL is the sampling cadence in seconds.
PROFILE_INFLIGHT=${PROFILE_INFLIGHT:-0}
PROFILE_INFLIGHT_INTERVAL=${PROFILE_INFLIGHT_INTERVAL:-0.5}
PROFILE_INFLIGHT_DIR=${PROFILE_INFLIGHT_DIR:-dp_inflight_profiles}

# Use vLLM async engine (AsyncLLM + per-sample streaming rollouts). The
# in-flight profiler also works under async (front-end streaming source).
# ASYNC=1 only flips the engine; set async GRPO separately if desired.
ASYNC=${ASYNC:-0}
if [ "${ASYNC}" = "1" ]; then
    ASYNC_OVERRIDES="policy.generation.vllm_cfg.async_engine=true"
else
    ASYNC_OVERRIDES=""
fi

# Nsight Systems (nsys) GPU profiling (see docs/nsys-profiling.md).
# NSYS=1 enables it. NSYS_PATTERNS are fnmatch globs over worker names
# (vllm_generation_worker / megatron_policy_worker); NSYS_STEP_RANGE is
# start:stop (1-indexed, stop exclusive). To also profile the Megatron policy,
# set NSYS_PATTERNS='*policy*,*vllm*' (the Megatron worker additionally needs
# LD_LIBRARY_PATH per the docs). .nsys-rep files land under $JOB-logs/ray/...
# and require RAY_LOG_SYNC_FREQUENCY to be synced off the container tmpfs.
NSYS=${NSYS:-0}
NSYS_PATTERNS=${NSYS_PATTERNS:-'*vllm*'}
NSYS_STEP_RANGE=${NSYS_STEP_RANGE:-2:3}
RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-30}

if [ "${NSYS}" = "1" ]; then
    NSYS_ENV="NRL_NSYS_WORKER_PATTERNS='${NSYS_PATTERNS}' NRL_NSYS_PROFILE_STEP_RANGE='${NSYS_STEP_RANGE}' "
    _ray_log_sync="${RAY_LOG_SYNC_FREQUENCY}"
else
    NSYS_ENV=""
    # Empty disables the ray.sub log-sync sidecar (its default-off behavior).
    _ray_log_sync=""
fi

QWEN3_32B=${QWEN3_32B:-${WORK_DIR}/hf_home/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}

CONTAINER=${CONTAINER:-${WORK_DIR}/sqsh/nemo_rl.v0.6.0.sqsh}
HF_HOME=${HF_HOME:-${WORK_DIR}/hf_home}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${WORK_DIR}/hf_home/cache}

# Pin the one-time HF->mcore conversion cache to a path mounted on
# all nodes. get_megatron_checkpoint_dir() otherwise falls back to $HF_HOME/nemo_rl,
# and a leaked HF_HOME pointing at a node-local path can resolve
# to a directory not mounted on every node, which fails multi-node mcore init.
NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR:-${WORK_DIR}/hf_home/nemo_rl}

if [ ${NUM_ACTOR_NODES} -le 16 ]; then
    SEGMENT=${SEGMENT:-${NUM_ACTOR_NODES}}
else
    SEGMENT=${SEGMENT:-16}
fi

if [ ! -f "${QWEN3_32B}/config.json" ]; then
    echo "Qwen3-32B checkpoint is missing config.json: ${QWEN3_32B}" >&2
    exit 1
fi

wandb_log_name=${WANDB_NAME:-OCI-${test_case}-steps${MAX_STEPS}}

if [ "${ENABLE_WANDB}" = "1" ]; then
    _wandb_key="${WANDB_API_KEY:-}"
    if [ -z "${_wandb_key}" ]; then
        echo "ENABLE_WANDB=1 requires WANDB_API_KEY to be set." >&2
        exit 1
    fi
    WANDB_OVERRIDES="logger.wandb_enabled=true \
logger.wandb.name=${wandb_log_name} \
logger.wandb.project=${WANDB_PROJECT}"
elif [ "${ENABLE_WANDB}" = "0" ]; then
    WANDB_OVERRIDES="logger.wandb_enabled=false"
else
    echo "ENABLE_WANDB must be 0 or 1, got: ${ENABLE_WANDB}" >&2
    exit 1
fi

COMMAND="${NSYS_ENV}HF_HOME=${HF_HOME} HF_DATASETS_CACHE=${HF_DATASETS_CACHE} NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR} NRL_LOG_DP_BATCH_STATS=${LOG_DP_BATCH_STATS} NRL_PROFILE_INFLIGHT=${PROFILE_INFLIGHT} NRL_PROFILE_INFLIGHT_INTERVAL=${PROFILE_INFLIGHT_INTERVAL} NRL_PROFILE_INFLIGHT_DIR=${PROFILE_INFLIGHT_DIR} uv run ./examples/run_grpo.py \
--config examples/configs/recipes/llm/performance/${test_case}.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
logger.wandb_enabled=false \
${WANDB_OVERRIDES} \
policy.model_name=${QWEN3_32B} \
${ASYNC_OVERRIDES} \
grpo.max_num_steps=${MAX_STEPS}"

echo "Submitting ${test_case}"
echo "  policy.model_name=${QWEN3_32B}"
echo "  max steps=${MAX_STEPS}"
echo "  dp batch stats=${LOG_DP_BATCH_STATS}"
echo "  async engine=${ASYNC}"
echo "  inflight profile=${PROFILE_INFLIGHT} (interval=${PROFILE_INFLIGHT_INTERVAL}s, dir=${PROFILE_INFLIGHT_DIR})"
if [ "${NSYS}" = "1" ]; then
    echo "  nsys=ON patterns='${NSYS_PATTERNS}' steps=${NSYS_STEP_RANGE} (ray_log_sync=${_ray_log_sync}s)"
else
    echo "  nsys=OFF"
fi
echo "  wandb enabled=${ENABLE_WANDB}"
if [ "${ENABLE_WANDB}" = "1" ]; then
    echo "  wandb=${WANDB_PROJECT}/${wandb_log_name}"
fi

COMMAND="${COMMAND}" \
CONTAINER="${CONTAINER}" \
HF_HOME="${HF_HOME}" \
HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
GPUS_PER_NODE="${GPUS_PER_NODE}" \
WANDB_API_KEY="${WANDB_API_KEY:-}" \
HF_TOKEN="${HF_TOKEN:-}" \
RAY_LOG_SYNC_FREQUENCY="${_ray_log_sync}" \
MOUNTS="${MOUNTS:-${WORK_DIR}:${WORK_DIR}}" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${SLURM_ACCOUNT} \
    --job-name=${JOB_NAME} \
    --partition=${SLURM_PARTITION:-batch} \
    --time=${WALLTIME:-00:30:00} \
    --segment=${SEGMENT} \
    --gres=gpu:${GPUS_PER_NODE} \
    ray.sub
