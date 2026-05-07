#!/usr/bin/env bash
# Run the Nemotron-3 Nano GRPO training command inside the Ray head-node shell.
# Prereq: launch_nano3_grpo.sh has been submitted and you have attached via
#   bash <JOBID>-attach.sh
set -euo pipefail

EOS_ROOT=/lustre/fsw/general_sa/xiyu
PROJECT_ROOT=${EOS_ROOT}/260505_prime_verifiers
DATA_DIR=${EOS_ROOT}/data/dapo17k_aime24/data
TRAIN_FILE=${DATA_DIR}/dapo17k_bytedtsinghua_train.jsonl
VAL_FILE=${DATA_DIR}/aime24_bytedtsinghua_validation.jsonl
RESULT_DIR=${PROJECT_ROOT}/results/nemotron_3_nano_grpo

POLICY_MODEL_DIR=${EOS_ROOT}/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
TOKENIZER_DIR=${POLICY_MODEL_DIR}

NUM_NODES=6
GPUS_PER_NODE=8
DATASET_NAME=dapo17k
RUN_TAG=$(date -u +%y%m%d-%H%M)

export HF_HOME=/opt/hf_home
export NRL_IGNORE_VERSION_MISMATCH=1

uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_nanov3_48xH100.yaml \
  cluster.num_nodes=${NUM_NODES} \
  cluster.gpus_per_node=${GPUS_PER_NODE} \
  data.train.data_path=${TRAIN_FILE} \
  data.validation.data_path=${VAL_FILE} \
  policy.model_name=${POLICY_MODEL_DIR} \
  policy.tokenizer.name=${TOKENIZER_DIR} \
  logger.log_dir=${RESULT_DIR}/logs/${SLURM_JOB_ID}-logs/training \
  checkpointing.checkpoint_dir=${RESULT_DIR}/checkpoints \
  logger.wandb_enabled=True \
  logger.wandb.project=nemo-rl-prime-verifiers \
  logger.wandb.name=nanov3-grpo-${DATASET_NAME}-${NUM_NODES}n-${RUN_TAG}
