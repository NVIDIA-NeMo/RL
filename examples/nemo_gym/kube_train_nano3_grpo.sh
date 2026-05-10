#!/usr/bin/env bash
# Run the Nemotron-3 Nano GRPO training command on a Kubernetes cluster.
# Mirrors slurm_train_nano3_grpo.sh — only the cluster-specific paths differ.
set -euo pipefail

PROJECT_ROOT=/workspace/260509_nemorl_prime_verifiers
DATA_DIR=/workspace/data/prime_intellect/acereason
TRAIN_FILE=${DATA_DIR}/acereason-math-mock-train.jsonl
VAL_FILE=${DATA_DIR}/acereason-math-mock-val.jsonl
RESULT_DIR=${PROJECT_ROOT}/results/nemotron_3_nano_grpo

POLICY_MODEL_DIR=/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
TOKENIZER_DIR=${POLICY_MODEL_DIR}

DATASET_NAME=acereason
RUN_TAG=$(date -u +%y%m%d-%H%M)
JOB_ID=${RAY_JOB_SUBMISSION_ID:-${NRL_K8S_RUN_ID:-${RUN_TAG}}}

export NRL_IGNORE_VERSION_MISMATCH=1

uv run run_grpo_nemo_gym.py \
  --config grpo_nanov3_24xH100.yaml \
  data.train.data_path=${TRAIN_FILE} \
  data.validation.data_path=${VAL_FILE} \
  policy.model_name=${POLICY_MODEL_DIR} \
  policy.tokenizer.name=${TOKENIZER_DIR} \
  logger.log_dir=${RESULT_DIR}/logs/${JOB_ID}-logs/training \
  checkpointing.checkpoint_dir=${RESULT_DIR}/checkpoints \
  logger.wandb_enabled=True \
  logger.wandb.project=nemo-rl-prime-verifiers \
  logger.wandb.name=nanov3-grpo-${DATASET_NAME}-kube-${RUN_TAG}
