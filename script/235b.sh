#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=16

huggingface-cli login --token $HF_TOKEN 

n_gen_nodes=8
n_train_nodes=$((NUM_ACTOR_NODES - n_gen_nodes))

account=coreai_dlalgo_nemorl

lag=1


COMMAND="uv run ./examples/run_grpo_math.py \
--config examples/configs/grpo_math_235B_megatron.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
grpo.async_grpo.enabled=true \
grpo.async_grpo.max_trajectory_age_steps=${lag} \
grpo.async_grpo.in_flight_weight_updates=true \
grpo.async_grpo.recompute_kv_cache_after_weight_updates=false \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.async_engine=true \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=${n_gen_nodes} \
policy.generation.colocated.resources.gpus_per_node=8 \
grpo.val_period=1000 \
checkpointing.enabled=false \
policy.sequence_packing.enabled=True \
grpo.max_num_steps=200 \
logger.wandb_enabled=True \
logger.wandb.project='async-grpo-benchmark' \
logger.wandb.name='async-qwen-235B-1215-lax-benchmark-lag${lag}-${n_train_nodes}T${n_gen_nodes}G'" \
CONTAINER=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo-rl:main-a010564b.sqsh \
HF_HOME=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home/cache \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/scratch:/scratch" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-async.test \
    --partition=batch \
    --time=01:10:00 \
    --gres=gpu:8 \
    ray_lax.sub
