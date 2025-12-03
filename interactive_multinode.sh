#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=4
account=coreai_dlalgo_nemorl

# Keep COMMAND empty to start in interactive mode (idle cluster)
# COMMAND="" 

CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl.sqsh \ 
HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home \ 
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/cache \ 
WANDB_API_KEY=$WANDB_API_KEY \ 
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-test.interactive-multinode \
    --partition=batch \
    --gres=gpu:4 \
    --segment ${NUM_ACTOR_NODES} \
    --time=04:00:00 \
    ray.sub