#!/bin/bash
# Launch Qwen3.5-397B TRT-LLM skip-training on an existing four-node Ray
# cluster. Run this inside the enroot shell on the Ray head node.

set -euo pipefail

ENV_FILE="${ENV_FILE:-/lustre/fsw/coreai_comparch_trtllm/erinh/env.sh}"
source "${ENV_FILE}"

# Do not propagate the attach step's distributed-launch environment into Ray
# actors or TRT-LLM's internal GPU workers.
for v in $(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print $1}'); do
    unset "$v"
done

export NEMO_RL_PY_EXECUTABLES_TRTLLM="${NEMO_RL_PY_EXECUTABLES_TRTLLM:-/opt/nemo_rl_venv/bin/python}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-1}"

export SEED="${SEED:-42}"
export DGXNNODES="${DGXNNODES:-4}"
export DGXNGPU="${DGXNGPU:-4}"
export TRAIN_NODES="${TRAIN_NODES:-2}"
export GEN_NODES="${GEN_NODES:-2}"
export SKIP_TRAINING=1
export MAX_STEPS="${MAX_STEPS:-2}"

export GEN_BACKEND=trtllm
export TRTLLM_TP="${TRTLLM_TP:-4}"
export TRTLLM_MEP="${TRTLLM_MEP:-4}"
export TRTLLM_GPU_UTIL="${TRTLLM_GPU_UTIL:-0.8}"
export TRTLLM_MAX_TOKENS="${TRTLLM_MAX_TOKENS:-2048}"
export TRTLLM_MODEL_PATH="${TRTLLM_MODEL_PATH:-/lustre/share/coreai_dlalgo_ci/artifacts/model/qwen_qwen3.5-397b-a17b-fp8/hf/hf-9f1f3de_orig}"

export RECIPE="${RECIPE:-/workspace/llm/conf/grpo_qwen35_397b_skip_train_4node_trtllm.yaml}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
export EXP_NAME="${EXP_NAME:-trtllm-397b-skip-4n-${MAX_STEPS}step-erinh}"
export MLPERF_MLLOG_FILE="${MLPERF_MLLOG_FILE:-/logs/mllog-${RUN_TAG}.log}"
export _experiment_index="${_experiment_index:-skip-${MAX_STEPS}step-${RUN_TAG}}"
# Avoid a potentially long 256-sample final validation: this smoke test should
# terminate immediately after the requested training-loop steps.
export EXTRA_ARGS="${EXTRA_ARGS:-++grpo.val_at_end=false}"

export CONTAINER_HF_CKPT_PATH="${CONTAINER_HF_CKPT_PATH:-/lustre/share/coreai_dlalgo_ci/artifacts/model/qwen_qwen3.5-397b-a17b/hf/hf-98d1a50_orig}"
export CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR="${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR:-/lustre/share/coreai_mlperf_training/data/qwen35_397b_grpo}"
export CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH:-/lustre/share/coreai_dlalgo_ci/artifacts/dataset/r2e-gym/r2e-gym/easy-subset-hf-e4f5af9_orig/benchmark_r2e_gym_easy_train.jsonl}"
export CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH:-/lustre/share/coreai_dlalgo_ci/artifacts/dataset/r2e-gym/r2e-gym/easy-subset-hf-e4f5af9_orig/benchmark_r2e_gym_easy_val.jsonl}"
export CONTAINER_NEMO_GYM_SWE_SIF_DIR="${CONTAINER_NEMO_GYM_SWE_SIF_DIR:-/lustre/share/coreai_dlalgo_ci/artifacts/dataset/r2e-gym/r2e-gym/easy-sif}"

echo "Launching ${EXP_NAME}: MAX_STEPS=${MAX_STEPS}, TRTLLM_TP=${TRTLLM_TP}"
exec bash /workspace/llm/run_and_time.sh
