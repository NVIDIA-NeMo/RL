#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

account=coreai_dlalgo_nemorl

# COMMAND="uv run ./examples/run_grpo_math.py --config examples/configs/async_grpo_math_1B.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 grpo.val_period=1000 policy.generation.colocated.enabled=false policy.generation.colocated.resources.num_nodes=1 policy.generation.colocated.resources.gpus_per_node=8 checkpointing.enabled=false logger.wandb_enabled=True logger.wandb.name='async-grpo-qwen1b_math' grpo.max_num_steps=10" \
# COMMAND="uv run ./examples/run_grpo_math.py --config examples/configs/async_grpo_math_8B.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 grpo.val_period=1000 policy.generation.colocated.enabled=false policy.generation.colocated.resources.num_nodes=1 policy.generation.colocated.resources.gpus_per_node=8 checkpointing.enabled=false logger.wandb_enabled=True logger.wandb.name='async-grpo-llama8b_math' grpo.max_num_steps=10" \
# CONTAINER=/home/youngeunk/lustre_home/sqsh/nemo_rl.sqsh \
CONTAINER=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo-rl:main-a010564b.sqsh \
HF_HOME=${HF_HOME} \
HF_DATASETS_CACHE=${HF_HOME}/cache \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/scratch:/scratch" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-test.interactive \
    --partition=batch \
    --gres=gpu:8 \
    --time=01:00:00 \
    ray_lax.sub
