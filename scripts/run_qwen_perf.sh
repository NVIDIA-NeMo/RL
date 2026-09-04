#!/bin/bash
# Run from the root of the NeMo RL repo.
#
# Generic GRPO perf-profiling launcher for the "Qwen series / on-policy / GB200"
# rows of the perf tracker. Parameterized over recipe + node layout + model so a
# thin per-config wrapper (see scripts/perf_qwen*.sh) just sets a few vars.
# The in-flight rollout profiler is ON by default and writes a meaningfully-named
# JSONL under dp_inflight_profiles/${RUN_TAG}/.
#
# Required (usually set by the wrapper):
#   RECIPE   recipe yaml stem under examples/configs/recipes/llm/performance/
#   RUN_TAG  short slug used for JSONL dir, job name, wandb name
# Optional:
#   MODEL              local snapshot path OR HF repo id; empty => use recipe default
#   NUM_ACTOR_NODES, GPUS_PER_NODE, MAX_STEPS, WALLTIME
#   PROFILE_INFLIGHT (default 1), PROFILE_INFLIGHT_INTERVAL, ASYNC, NSYS, ENABLE_WANDB

set -euo pipefail

# Site-specific root: one path, visible on every node, holding the container
# image, the HF cache and the mcore checkpoint cache. No default on purpose --
# a wrong shared path is a multi-node failure that only shows up at init.
WORK_DIR=${WORK_DIR:?set WORK_DIR to a shared path visible on every node}

RECIPE=${RECIPE:?set RECIPE to a recipe yaml stem (e.g. grpo-qwen3-32b-4n4g)}
RUN_TAG=${RUN_TAG:?set RUN_TAG to a short slug for naming outputs}

SLURM_ACCOUNT=${SLURM_ACCOUNT:?set SLURM_ACCOUNT to your Slurm account}
NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MAX_STEPS=${MAX_STEPS:-3}

# ray.sub pins `#SBATCH --dependency=singleton`, so distinct names run in parallel.
JOB_NAME=${JOB_NAME:-${SLURM_ACCOUNT}-${RUN_TAG}}

ENABLE_WANDB=${ENABLE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-qwen-gb200-perf}

# In-flight rollout profiler (per-DP batch size + per-seq context length over
# time). ON by default for these perf scripts. JSONL is named by RUN_TAG.
PROFILE_INFLIGHT=${PROFILE_INFLIGHT:-1}
PROFILE_INFLIGHT_INTERVAL=${PROFILE_INFLIGHT_INTERVAL:-0.5}
PROFILE_INFLIGHT_DIR=${PROFILE_INFLIGHT_DIR:-dp_inflight_profiles/${RUN_TAG}}
# Resolve to an absolute path so it is unambiguous inside the container (the
# driver's cwd is the repo root, but pin it explicitly).
case "${PROFILE_INFLIGHT_DIR}" in
    /*) ;;
    *) PROFILE_INFLIGHT_DIR="$(pwd)/${PROFILE_INFLIGHT_DIR}" ;;
esac
mkdir -p "${PROFILE_INFLIGHT_DIR}"

LOG_DP_BATCH_STATS=${LOG_DP_BATCH_STATS:-1}

# vLLM async engine (AsyncLLM + per-sample streaming rollouts).
ASYNC=${ASYNC:-0}
if [ "${ASYNC}" = "1" ]; then
    ASYNC_OVERRIDES="policy.generation.vllm_cfg.async_engine=true"
else
    ASYNC_OVERRIDES=""
fi

# Nsight Systems profiling (see docs/nsys-profiling.md).
NSYS=${NSYS:-0}
NSYS_PATTERNS=${NSYS_PATTERNS:-'*vllm*'}
NSYS_STEP_RANGE=${NSYS_STEP_RANGE:-2:3}
RAY_LOG_SYNC_FREQUENCY=${RAY_LOG_SYNC_FREQUENCY:-30}
if [ "${NSYS}" = "1" ]; then
    NSYS_ENV="NRL_NSYS_WORKER_PATTERNS='${NSYS_PATTERNS}' NRL_NSYS_PROFILE_STEP_RANGE='${NSYS_STEP_RANGE}' "
    _ray_log_sync="${RAY_LOG_SYNC_FREQUENCY}"
else
    NSYS_ENV=""
    _ray_log_sync=""
fi

CONTAINER=${CONTAINER:-${WORK_DIR}/sqsh/nemo_rl.v0.6.0.sqsh}
HF_HOME=${HF_HOME:-${WORK_DIR}/hf_home}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${WORK_DIR}/hf_home/cache}
# Pin the HF->mcore conversion cache under WORK_DIR (visible on every node);
# avoids a leaked HF_HOME on a node-local path breaking multi-node mcore init.
NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR:-${WORK_DIR}/hf_home/nemo_rl}

if [ ${NUM_ACTOR_NODES} -le 16 ]; then
    SEGMENT=${SEGMENT:-${NUM_ACTOR_NODES}}
else
    SEGMENT=${SEGMENT:-16}
fi

# Only override policy.model_name if MODEL is given; empty => recipe default.
# A local path (starts with /) must exist; a HF repo id is resolved/downloaded.
MODEL=${MODEL:-}
MODEL_OVERRIDE=""
if [ -n "${MODEL}" ]; then
    case "${MODEL}" in
        /*)
            if [ ! -f "${MODEL}/config.json" ]; then
                echo "MODEL local path missing config.json: ${MODEL}" >&2
                exit 1
            fi
            ;;
    esac
    MODEL_OVERRIDE="policy.model_name=${MODEL}"
fi

wandb_log_name=${WANDB_NAME:-${RUN_TAG}-steps${MAX_STEPS}}
if [ "${ENABLE_WANDB}" = "1" ]; then
    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "ENABLE_WANDB=1 requires WANDB_API_KEY to be set." >&2
        exit 1
    fi
    WANDB_OVERRIDES="logger.wandb_enabled=true logger.wandb.name=${wandb_log_name} logger.wandb.project=${WANDB_PROJECT}"
else
    WANDB_OVERRIDES="logger.wandb_enabled=false"
fi

COMMAND="${NSYS_ENV}HF_HOME=${HF_HOME} HF_DATASETS_CACHE=${HF_DATASETS_CACHE} NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR} NRL_LOG_DP_BATCH_STATS=${LOG_DP_BATCH_STATS} NRL_PROFILE_INFLIGHT=${PROFILE_INFLIGHT} NRL_PROFILE_INFLIGHT_INTERVAL=${PROFILE_INFLIGHT_INTERVAL} NRL_PROFILE_INFLIGHT_DIR=${PROFILE_INFLIGHT_DIR} uv run ./examples/run_grpo.py \
--config examples/configs/recipes/llm/performance/${RECIPE}.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
${WANDB_OVERRIDES} \
${MODEL_OVERRIDE} \
${ASYNC_OVERRIDES} \
grpo.max_num_steps=${MAX_STEPS}"

echo "Submitting ${RUN_TAG}"
echo "  recipe=${RECIPE}  nodes=${NUM_ACTOR_NODES}x${GPUS_PER_NODE}gpu"
echo "  model=${MODEL:-<recipe default>}"
echo "  max steps=${MAX_STEPS}  walltime=${WALLTIME:-00:30:00}"
echo "  async engine=${ASYNC}  nsys=${NSYS}  dp batch stats=${LOG_DP_BATCH_STATS}"
echo "  inflight profile=${PROFILE_INFLIGHT} -> ${PROFILE_INFLIGHT_DIR}/inflight_timeline.jsonl"

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
