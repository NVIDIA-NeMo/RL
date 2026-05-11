#!/usr/bin/env bash
# Run the Nemotron-3 Nano GRPO training command on a Kubernetes cluster.
# Mirrors slurm_train_nano3_grpo.sh — only the cluster-specific paths differ.
set -euo pipefail

PROJECT_ROOT=/workspace/260509_nemorl_prime_verifiers
RESULT_DIR=${PROJECT_ROOT}/results/nemotron_3_nano_grpo
POLICY_MODEL_DIR=/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
TOKENIZER_DIR=${POLICY_MODEL_DIR}
RUN_TAG=$(date -u +%y%m%d-%H%M)
JOB_ID=${RAY_JOB_SUBMISSION_ID:-${NRL_K8S_RUN_ID:-${RUN_TAG}}}
export NRL_IGNORE_VERSION_MISMATCH=1

# Export NVIDIA_API_KEY (and anything else in /workspace/.env) so the
# verifiers_agent judge + embedding clients can authenticate to the
# NVIDIA inference hub. set -a turns on allexport, so every assignment
# below is auto-exported to child processes (uv run, ng_run, etc.).
set -a
source /workspace/.env
set +a

# Dataset-specific knobs derived from measure_prompt_len.py:
#   MAX_NEW_TOKENS = MAX_TOTAL_SEQ_LENGTH - prompt_max - 32 (buffer) - 1 (boundary)
# Re-measure with examples/nemo_gym/measure_prompt_len.py whenever the dataset,
# chat template, or MAX_TOTAL_SEQ_LENGTH changes. Bumping MAX_TOTAL_SEQ_LENGTH
# costs activation memory — verify it fits the parallelism in the recipe (PP,
# TP, EP) before raising it.

# For Acereason (single-turn math; prompt_max=1577 on acereason-math-mock-train, n=1000)
# DATASET_NAME=acereason
# DATA_DIR=/workspace/data/prime_intellect/acereason
# TRAIN_FILE=${DATA_DIR}/acereason-math-mock-train.jsonl
# VAL_FILE=${DATA_DIR}/acereason-math-mock-val.jsonl
# AGENT_CONFIG=responses_api_agents/verifiers_agent/configs/acereason-math.yaml
# MAX_TOTAL_SEQ_LENGTH=8192
# MAX_NEW_TOKENS=6582   # 8192 - 1577 - 32 - 1

# For wiki-search (multi-turn RAG; prompt_max=65 on wiki-search-mock-train, n=300)
DATASET_NAME=wiki-search
DATA_DIR=/workspace/data/prime_intellect/wiki-search
TRAIN_FILE=${DATA_DIR}/wiki-search-mock-train.jsonl
VAL_FILE=${DATA_DIR}/wiki-search-mock-val.jsonl
AGENT_CONFIG=responses_api_agents/verifiers_agent/configs/wiki-search.yaml
MAX_TOTAL_SEQ_LENGTH=8192
MAX_NEW_TOKENS=8094     # 8192 - 65 - 32 - 1

# MODEL_CONFIG is required for any agent, and it needs to be vllm_model_for_training.yaml
MODEL_CONFIG=responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
NEMO_GYM_CONFIG_PATHS="[${MODEL_CONFIG},${AGENT_CONFIG}]"


uv run run_grpo_nemo_gym.py \
  --config grpo_nanov3_24xH100.yaml \
  data.train.data_path=${TRAIN_FILE} \
  data.validation.data_path=${VAL_FILE} \
  policy.model_name=${POLICY_MODEL_DIR} \
  policy.tokenizer.name=${TOKENIZER_DIR} \
  policy.max_total_sequence_length=${MAX_TOTAL_SEQ_LENGTH} \
  policy.generation.max_new_tokens=${MAX_NEW_TOKENS} \
  env.nemo_gym.config_paths="${NEMO_GYM_CONFIG_PATHS}" \
  logger.log_dir=${RESULT_DIR}/logs/${JOB_ID}-logs/training \
  checkpointing.checkpoint_dir=${RESULT_DIR}/checkpoints \
  logger.wandb_enabled=True \
  logger.wandb.project=nemo-rl-prime-verifiers \
  logger.wandb.name=nanov3-grpo-${DATASET_NAME}-kube-${RUN_TAG}
