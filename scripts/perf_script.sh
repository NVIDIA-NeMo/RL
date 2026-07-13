#!/bin/bash
# Run from the root of NeMo RL repo

SLURM_ACCOUNT=coreai_dlalgo_nemorl

test_case="grpo-qwen3-30ba3b-4n8g-async-1off"

if [[ "$test_case" =~ -([0-9]+)n([0-9]+)g ]]; then
  NUM_ACTOR_NODES="${BASH_REMATCH[1]}"
  GPUS_PER_NODE="${BASH_REMATCH[2]}"
  echo "Parsed NUM_ACTOR_NODES: $NUM_ACTOR_NODES"
  echo "Parsed GPUS_PER_NODE: $GPUS_PER_NODE"
else
  echo "Could not parse NUM_ACTOR_NODES and GPUS_PER_NODE from test_case: $test_case"
  exit 1
fi


pps=8
gpp=4
seq_len=512
training_batch_size=$((pps * gpp))


EXTRA_OVERRIDES="policy.train_global_batch_size=${training_batch_size} \
grpo.num_prompts_per_step=${pps} \
grpo.num_generations_per_prompt=${gpp} \
policy.max_total_sequence_length=${seq_len} \
grpo.max_num_steps=10 \
checkpointing.enabled=False"


wandb_log_name=CWDFW-${test_case}-refit-shrink


COMMAND="uv run ./examples/run_grpo.py \
--config examples/configs/recipes/llm/performance/${test_case}.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
logger.wandb.name=${wandb_log_name} \
${EXTRA_OVERRIDES} \
logger.wandb.project='nemorl-refit-shrink'" \
CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo_rl.0707.sqsh \
HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home/cache \
GPUS_PER_NODE=${GPUS_PER_NODE} \
NRL_FORCE_REBUILD_VENVS=true \
HF_HUB_OFFLINE=1 \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${SLURM_ACCOUNT} \
    --job-name=${SLURM_ACCOUNT}-perf.${test_case}-refit-shrink \
    --partition=batch \
    --time=00:15:00 \
    --gres=gpu:${GPUS_PER_NODE} \
    --dependency="" \
    ray.sub