#!/bin/bash
# Run from the root of the NeMo RL repo.
#
# DeepSeek-V3 perf A/B:
#   ENABLE_MTP=0 bash ./run-ds.sh   # baseline
#   ENABLE_MTP=1 bash ./run-ds.sh   # vLLM-side MTP speculative decoding

set -euo pipefail

# Site-specific root: one path, visible on every node, holding the container
# image, the HF cache and the mcore checkpoint cache. No default on purpose --
# a wrong shared path is a multi-node failure that only shows up at init.
WORK_DIR=${WORK_DIR:?set WORK_DIR to a shared path visible on every node}

SLURM_ACCOUNT=${SLURM_ACCOUNT:?set SLURM_ACCOUNT to your Slurm account}
NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
test_case=${test_case:-grpo-deepseek-v3-32n4g}

ENABLE_MTP=${ENABLE_MTP:-0}
MAX_STEPS=${MAX_STEPS:-8}
ENABLE_WANDB=${ENABLE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-async-grpo-perfscript-test}

DS_BF16=${DS_BF16:-${WORK_DIR}/hf_home/Deepseek-V3-BF16}
DS_BF16_MTP=${DS_BF16_MTP:-${WORK_DIR}/.cache/huggingface/nemo_rl/Deepseek-V3-BF16-mtp}

CONTAINER=${CONTAINER:-${WORK_DIR}/sqsh/nemo_rl.v0.6.0.sqsh}
HF_HOME=${HF_HOME:-${WORK_DIR}/hf_home}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${WORK_DIR}/hf_home/cache}

if [ ${NUM_ACTOR_NODES} -le 16 ]; then
    SEGMENT=${SEGMENT:-${NUM_ACTOR_NODES}}
else
    SEGMENT=${SEGMENT:-16}
fi

if [ ! -f "${DS_BF16}/config.json" ] || [ ! -f "${DS_BF16}/model.safetensors.index.json" ]; then
    echo "DeepSeek BF16 checkpoint is missing required files: ${DS_BF16}" >&2
    exit 1
fi

if ! grep -q "model.layers.61.eh_proj.weight" "${DS_BF16}/model.safetensors.index.json"; then
    echo "DeepSeek BF16 checkpoint does not appear to include the layer-61 MTP weights: ${DS_BF16}" >&2
    exit 1
fi

MTP_OVERRIDES=""
MTP_TAG="nomtp"
POLICY_MODEL="${DS_BF16}"

if [ "${ENABLE_MTP}" = "1" ]; then
    mkdir -p "${DS_BF16_MTP}"
    for f in "${DS_BF16}"/*; do
        ln -sfn "$f" "${DS_BF16_MTP}/$(basename "$f")"
    done
    rm -f "${DS_BF16_MTP}/config.json"
    jq '.num_nextn_predict_layers = 1' "${DS_BF16}/config.json" > "${DS_BF16_MTP}/config.json"

    POLICY_MODEL="${DS_BF16_MTP}"
    MTP_TAG="mtp"
    MTP_OVERRIDES="++policy.generation.vllm_kwargs.speculative_config.method=mtp \
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=1"
elif [ "${ENABLE_MTP}" != "0" ]; then
    echo "ENABLE_MTP must be 0 or 1, got: ${ENABLE_MTP}" >&2
    exit 1
fi

wandb_log_name=${WANDB_NAME:-OCI-${test_case}-${MTP_TAG}-steps${MAX_STEPS}}

if [ "${ENABLE_WANDB}" = "1" ]; then
    if [ -z "${WANDB_API_KEY:-}" ]; then
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

COMMAND="HF_HOME=${HF_HOME} HF_DATASETS_CACHE=${HF_DATASETS_CACHE} uv run ./examples/run_grpo.py \
--config examples/configs/recipes/llm/performance/${test_case}.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
${WANDB_OVERRIDES} \
policy.model_name=${POLICY_MODEL} \
grpo.max_num_steps=${MAX_STEPS} \
${MTP_OVERRIDES}"

echo "Submitting ${test_case}"
echo "  ENABLE_MTP=${ENABLE_MTP}"
echo "  policy.model_name=${POLICY_MODEL}"
echo "  max steps=${MAX_STEPS}"
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
MOUNTS="${MOUNTS:-${WORK_DIR}:${WORK_DIR}}" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${SLURM_ACCOUNT} \
    --job-name=${SLURM_ACCOUNT}-ds-${MTP_TAG} \
    --partition=${SLURM_PARTITION:-batch} \
    --time=${WALLTIME:-01:00:00} \
    --segment=${SEGMENT} \
    --gres=gpu:${GPUS_PER_NODE} \
    ray.sub
