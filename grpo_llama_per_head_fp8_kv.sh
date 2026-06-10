#!/bin/bash
set -euo pipefail

NUM_NODES=${NUM_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
MAX_STEPS=${MAX_STEPS:-4000}
NUM_PROMPTS_PER_STEP=${NUM_PROMPTS_PER_STEP:-16}
NUM_GENERATIONS_PER_PROMPT=${NUM_GENERATIONS_PER_PROMPT:-32}
TRAIN_GLOBAL_BATCH_SIZE=${TRAIN_GLOBAL_BATCH_SIZE:-512}
MAX_TOTAL_SEQUENCE_LENGTH=${MAX_TOTAL_SEQUENCE_LENGTH:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}
WANDB_PROJECT=${WANDB_PROJECT:-guyueh-nemo-rl-fp8-kv}
MEGATRON_FP8_KV_HOOK=${MEGATRON_FP8_KV_HOOK:-true}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_per_token_head}
DRIVER_SRUN_ARGS=${DRIVER_SRUN_ARGS:---mem=64G}
INTERACTIVE=${INTERACTIVE:-0}

case "${MEGATRON_FP8_KV_HOOK}" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
        MEGATRON_FP8_KV_HOOK=true
        HOOK_VARIANT=megatron-hook
        ;;
    0|false|FALSE|False|no|NO|No|off|OFF|Off)
        MEGATRON_FP8_KV_HOOK=false
        HOOK_VARIANT=no-megatron-hook
        ;;
    *)
        echo "MEGATRON_FP8_KV_HOOK must be true or false; got ${MEGATRON_FP8_KV_HOOK}" >&2
        exit 1
        ;;
esac

case "${KV_CACHE_DTYPE}" in
    fp8_per_token_head)
        KV_CACHE_VARIANT=per-head-fp8-kv
        ;;
    auto)
        KV_CACHE_VARIANT=bf16-kv-cache
        ;;
    *)
        echo "KV_CACHE_DTYPE must be fp8_per_token_head or auto; got ${KV_CACHE_DTYPE}" >&2
        exit 1
        ;;
esac

EXPERIMENT_VARIANT=${KV_CACHE_VARIANT}-${HOOK_VARIANT}
if [ "${KV_CACHE_DTYPE}" = "auto" ] && [ "${MEGATRON_FP8_KV_HOOK}" = "false" ]; then
    EXPERIMENT_VARIANT=bf16-baseline
fi

JOB_NAME=${JOB_NAME:-grpo-llama3-1-8b-${EXPERIMENT_VARIANT}}
WANDB_NAME=${WANDB_NAME:-${JOB_NAME}-importance-sampling}

TRAIN_CMD="\
uv run examples/run_grpo.py \
--config examples/configs/grpo_math_8B_megatron.yaml \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
grpo.max_num_steps=${MAX_STEPS} \
grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT} \
policy.model_name=meta-llama/Llama-3.1-8B-Instruct \
policy.tokenizer.name=meta-llama/Llama-3.1-8B-Instruct \
policy.precision=bfloat16 \
policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
policy.max_total_sequence_length=${MAX_TOTAL_SEQUENCE_LENGTH} \
policy.megatron_cfg.tensor_model_parallel_size=${TP_SIZE} \
policy.megatron_cfg.pipeline_model_parallel_size=${PP_SIZE} \
policy.megatron_cfg.pipeline_dtype=bfloat16 \
policy.megatron_cfg.fp8_per_token_head_kv_cache_hook=${MEGATRON_FP8_KV_HOOK} \
policy.generation.max_new_tokens=${MAX_TOTAL_SEQUENCE_LENGTH} \
policy.generation.vllm_cfg.precision=bfloat16 \
policy.generation.vllm_cfg.tensor_parallel_size=${TP_SIZE} \
policy.generation.vllm_cfg.pipeline_parallel_size=1 \
policy.generation.vllm_cfg.max_model_len=${MAX_TOTAL_SEQUENCE_LENGTH} \
policy.generation.vllm_cfg.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
policy.generation.vllm_cfg.async_engine=false \
policy.generation.vllm_cfg.kv_cache_dtype=${KV_CACHE_DTYPE} \
data.max_input_seq_length=${MAX_TOTAL_SEQUENCE_LENGTH} \
loss_fn.use_importance_sampling_correction=true \
checkpointing.enabled=false \
logger.wandb_enabled=true \
logger.wandb.project=${WANDB_PROJECT} \
logger.wandb.name=${WANDB_NAME} \
${*}
"

echo "NUM_NODES: ${NUM_NODES}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "TP_SIZE: ${TP_SIZE}"
echo "PP_SIZE: ${PP_SIZE}"
echo "MAX_STEPS: ${MAX_STEPS}"
echo "MAX_TOTAL_SEQUENCE_LENGTH: ${MAX_TOTAL_SEQUENCE_LENGTH}"
echo "MEGATRON_FP8_KV_HOOK: ${MEGATRON_FP8_KV_HOOK}"
echo "KV_CACHE_DTYPE: ${KV_CACHE_DTYPE}"
echo "DRIVER_SRUN_ARGS: ${DRIVER_SRUN_ARGS}"
echo "JOB_NAME: ${JOB_NAME}"
echo "WANDB_NAME: ${WANDB_NAME}"

if [ "${INTERACTIVE}" -eq 1 ]; then
    eval "${TRAIN_CMD}"
else
    export COMMAND=${TRAIN_CMD}
    export GPUS_PER_NODE
    export DRIVER_SRUN_ARGS
    DEFAULT_CONTAINER=/lustre/fsw/portfolios/coreai/users/guyueh
    DEFAULT_CONTAINER=${DEFAULT_CONTAINER}/container_images/nvidian+nemo-rl+nightly.sqsh
    export CONTAINER=${CONTAINER:-${DEFAULT_CONTAINER}}
    REPO_DIR=${PWD}
    PHYSICAL_REPO_DIR=$(pwd -P)
    DEFAULT_MOUNTS="${REPO_DIR}:/opt/nemo-rl"
    HSG_FSW_USER_DIR=${HSG_FSW_USER_DIR:-/lustre/fsw/portfolios/coreai/users/guyueh}
    DEFAULT_MOUNTS="${DEFAULT_MOUNTS},${HSG_FSW_USER_DIR}:${HSG_FSW_USER_DIR}"
    if [ "${PHYSICAL_REPO_DIR}" != "${REPO_DIR}" ]; then
        DEFAULT_MOUNTS="${DEFAULT_MOUNTS},${PHYSICAL_REPO_DIR}:${PHYSICAL_REPO_DIR}"
    fi
    DEFAULT_MOUNTS="${DEFAULT_MOUNTS},/home/guyueh/:/home/guyueh/"
    export MOUNTS=${MOUNTS:-${DEFAULT_MOUNTS}}

    sbatch \
        --nodes=${NUM_NODES} \
        --segment=${NUM_NODES} \
        --account=${SLURM_ACCOUNT:-nemotron_n4_post} \
        --job-name=${JOB_NAME} \
        --partition=${PARTITION:-batch} \
        --gres=gpu:${GPUS_PER_NODE} \
        --mem=0 \
        --time=${TIME_LIMIT:-04:00:00} \
        ray.sub
fi
