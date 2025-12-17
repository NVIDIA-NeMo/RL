#!/bin/bash
# Run from the root of NeMo RL repo

huggingface-cli login --token $HF_TOKEN 


NUM_ACTOR_NODES=4
# test_case="grpo-deepseek-v3-64n8g-async-1off"
# test_case="grpo-llama3.1-8b-instruct-2n8g-async-1off"
test_case="grpo-qwen3-30ba3b-4n8g-async-1off"
# test_case="grpo-qwen3-30ba3b-4n8g"
# test_case="grpo-qwen3-32b-8n8g-async-1off"
# test_case="grpo-qwen3-235b-32n8g-async-1off"


COMMAND="uv run ./examples/run_grpo_math.py \
--config examples/configs/recipes/llm/performance/${test_case}.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
logger.wandb.project='async-grpo-perfscript-test'" \
CONTAINER=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo-rl:main-a010564b.sqsh \
HF_HOME=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home/cache \
NRL_FORCE_REBUILD_VENVS=true \
WANDB_API_KEY=$WANDB_API_KEY \
TORCH_CUDA_ARCH_LIST='9.0 10.0' \
MOUNTS="/scratch:/scratch" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=coreai_dlalgo_nemorl \
    --job-name=coreai_dlalgo_nemorl-async.test \
    --partition=batch \
    --time=01:00:00 \
    --gres=gpu:8 \
    ray_lax.sub