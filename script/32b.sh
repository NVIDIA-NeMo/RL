#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=16
n_gen_nodes=8

huggingface-cli login --token $HF_TOKEN 

account=coreai_dlalgo_nemorl

n_train_nodes=$((NUM_ACTOR_NODES - n_gen_nodes))

SHARP_ONLY=true
inflight=false

seq_len=16384

lag=1

g_tp=2

t_tp=4
t_cp=4
t_pp=4

sp=True
if [ ${t_tp} -eq 1 ]; then
    sp=False
fi

# NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*" \
# NRL_NSYS_PROFILE_STEP_RANGE=3:5 \
# RAY_LOG_SYNC_FREQUENCY=30 \

COMMAND="uv run ./examples/run_grpo_math.py \
--config examples/configs/grpo_math_32B_megatron.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.async_grpo.enabled=true \
grpo.async_grpo.max_trajectory_age_steps=${lag} \
grpo.async_grpo.in_flight_weight_updates=${inflight} \
grpo.async_grpo.recompute_kv_cache_after_weight_updates=false \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.async_engine=true \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=${n_gen_nodes} \
policy.generation.colocated.resources.gpus_per_node=8 \
policy.generation.vllm_cfg.tensor_parallel_size=${g_tp} \
policy.max_total_sequence_length=${seq_len} \
policy.megatron_cfg.tensor_model_parallel_size=${t_tp} \
policy.megatron_cfg.context_parallel_size=${t_cp} \
policy.megatron_cfg.pipeline_model_parallel_size=${t_pp} \
policy.megatron_cfg.sequence_parallel=${sp} \
grpo.max_num_steps=15 \
logger.wandb_enabled=True \
logger.wandb.project='async-grpo-32b' \
logger.wandb.name='async-grpo-qwen3-32B-1202-shape-only-${SHARP_ONLY}-benchmark-lag${lag}-inflight${inflight}-${n_train_nodes}T${n_gen_nodes}G-Gtp${g_tp}Ttp${t_tp}Tcp${t_cp}Tpp${t_pp}-${seq_len}SL'" \
CONTAINER=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo-rl:main-a010564b.sqsh \
HF_HOME=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home/cache \
NRL_FORCE_REBUILD_VENVS=true \
TORCH_CUDA_ARCH_LIST='9.0 10.0' \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/scratch:/scratch" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-async.test \
    --partition=batch \
    --time=01:40:00 \
    --gres=gpu:8 \
    ray_lax.sub

    # --ntasks-per-node=8 \