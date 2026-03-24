#!/bin/bash

set -eou pipefail

SCRIPT_LOC="$(readlink -f $0)"
SCRIPT_PWD="$(pwd -P)"
date
echo "INFO: Script location = ${SCRIPT_LOC}"
echo "INFO: Current workdir = ${SCRIPT_PWD}"

SLURM_RUN_NAME="convert_nano_v3_ckpt_mcore_to_hf"

# change these
# ACCOUNT="${ACCOUNT:-llmservice_modelalignment_sft}"
ACCOUNT="${ACCOUNT:-llmservice_modelalignment_ppo}"
# ACCOUNT="${ACCOUNT:-llmservice_nemotron_nano}"
# ACCOUNT="${ACCOUNT:-llmservice_nemotron_super}"
# ACCOUNT="${ACCOUNT:-llmservice_fm_vision}"
PARTITION="${PARTITION:-interactive}"
# PARTITION="${PARTITION:-batch_block1}"
# PARTITION="${PARTITION:-batch_short}"
# PARTITION="${PARTITION:-batch}"
NUM_NODES="${NUM_NODES:-1}"
TIME_LIMIT="${TIME_LIMIT:-2:00:00}"

echo "INFO: Slurm account   = ${ACCOUNT}"
echo "INFO: Slurm partition = ${PARTITION}"
echo "INFO: Slurm num nodes = ${NUM_NODES}"
echo "INFO: Slurm run name  = ${SLURM_RUN_NAME}"

MOUNTS="/lustre/fsw:/lustre/fsw,${SCRIPT_PWD}:${SCRIPT_PWD}"
if [ -d "/scratch/fsw" ]; then
    MOUNTS="/scratch/fsw:/scratch/fsw,${MOUNTS}"
fi
echo "INFO: Container mount paths = ${MOUNTS}"

CONTAINER="/lustre/fsw/portfolios/llmservice/users/pjin/containers/nemo-rl:vllm-0_11_2-nanov3-20251205_1.squashfs"
echo "INFO: Container image path = ${CONTAINER}"

CKPT_PATH="$1"
echo "INFO: Checkpoint path = ${CKPT_PATH}"

srun \
    --nodes="${NUM_NODES}" \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    --job-name="${SLURM_RUN_NAME}" \
    --exclusive \
    --gres=gpu:8 \
    --container-image="${CONTAINER}" \
    --container-mounts="${MOUNTS}" \
    --no-container-mount-home \
    bash -c "
        cd ${SCRIPT_PWD}
        uv pip install setuptools_scm cmake
        uv run --extra mcore examples/converters/convert_megatron_to_hf.py \
            --config="${CKPT_PATH}/config.yaml" \
            --megatron-ckpt-path="${CKPT_PATH}/policy/weights/iter_0000000" \
            --hf-ckpt-path="${CKPT_PATH}/policy/weights/hf"
    "
