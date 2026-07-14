#!/bin/bash
# Run from the root of NeMo RL repo

SLURM_ACCOUNT=coreai_dlalgo_nemorl

test_case="grpo-qwen3-30ba3b-4n8g-async-1off"
TOTAL_GPUS="${TOTAL_GPUS:-32}"
GPUS_PER_NODE=8
JOB_TIME="${JOB_TIME:-00:40:00}"
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"

if ! [[ "$TOTAL_GPUS" =~ ^[0-9]+$ ]]; then
  echo "TOTAL_GPUS must be a positive integer, got: $TOTAL_GPUS"
  exit 1
fi

if ((TOTAL_GPUS < 32 || TOTAL_GPUS % (2 * GPUS_PER_NODE) != 0)); then
  echo "TOTAL_GPUS must be at least 32 and divisible by $((2 * GPUS_PER_NODE)), got: $TOTAL_GPUS"
  exit 1
fi

NUM_ACTOR_NODES=$((TOTAL_GPUS / GPUS_PER_NODE))
INFERENCE_NODES=$((NUM_ACTOR_NODES / 2))
gpp=4
pps=$((TOTAL_GPUS / gpp))
seq_len=512
training_batch_size=$((pps * gpp))

echo "Total GPUs: $TOTAL_GPUS"
echo "Actor nodes: $NUM_ACTOR_NODES"
echo "Training GPUs: $((TOTAL_GPUS / 2))"
echo "Inference GPUs: $((TOTAL_GPUS / 2))"
echo "Training global batch size: $training_batch_size"
echo "Prompts per step: $pps"
echo "Job time: $JOB_TIME"

EXTRA_OVERRIDES="policy.train_global_batch_size=${training_batch_size} \
grpo.num_prompts_per_step=${pps} \
grpo.num_generations_per_prompt=${gpp} \
policy.max_total_sequence_length=${seq_len} \
policy.generation.colocated.resources.num_nodes=${INFERENCE_NODES} \
policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE} \
grpo.max_num_steps=10 \
checkpointing.enabled=False \
logger.wandb_enabled=False"


wandb_log_name=CWDFW-${test_case}-refit-shrink-${TOTAL_GPUS}g


COMMAND="uv run ./examples/run_grpo.py \
--config examples/configs/recipes/llm/performance/${test_case}.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
logger.log_dir=logs/refit-shrink-scaling/${TOTAL_GPUS}g \
logger.wandb.name=${wandb_log_name} \
${EXTRA_OVERRIDES} \
logger.wandb.project='nemorl-refit-shrink'" \
CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo_rl.0707.sqsh \
HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home/cache \
GPUS_PER_NODE=${GPUS_PER_NODE} \
NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true} \
HF_HUB_OFFLINE=1 \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${SLURM_ACCOUNT} \
    --job-name=${SLURM_ACCOUNT}-perf.refit-shrink-${TOTAL_GPUS}g \
    --partition=batch \
    --time=${JOB_TIME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --dependency="${SLURM_DEPENDENCY}" \
    ray.sub
