#!/bin/bash

WORK_DIR=/lustre/fsw/portfolios/coreai/users/mingyyang/xtoken_nemorl_upstream
NUM_ACTOR_NODES=1
EXP_NAME=CrossTokenizer-Distillation-Llama1B-Phi4

export CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/nemo-rl.sqsh
export MOUNTS="/lustre:/lustre"
export BASE_LOG_DIR="${WORK_DIR}/x_token"

read -r -d '' COMMAND <<EOF
export HF_HOME=/lustre/fsw/portfolios/coreai/users/mingyyang/hf_cache
export HF_TOKEN=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
export WANDB_API_KEY=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
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=1

cd ${WORK_DIR}

uv run ${WORK_DIR}/examples/run_off_policy_distillation_arrow_with_eval.py \
  --config ${WORK_DIR}/examples/configs/cross_tokenizer_off_policy_arrow.yaml \
  cluster.num_nodes=${NUM_ACTOR_NODES} \
  teacher.model_name=microsoft/phi-4 \
  teacher.tokenizer.name=microsoft/phi-4 \
  distillation.num_prompts_per_step=8 \
  policy.train_global_batch_size=8 \
  teacher.train_global_batch_size=8 \
  distillation.max_num_steps=10 \
  logger.wandb.name=${EXP_NAME} \
  logger.log_dir=logs/${EXP_NAME} \
  checkpointing.checkpoint_dir=checkpoints/${EXP_NAME}
EOF
export COMMAND

bash ray.sub