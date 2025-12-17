#!/bin/bash
# Run from the root of NeMo RL repo
huggingface-cli login --token $HF_TOKEN 

account=coreai_dlalgo_nemorl

NUM_ACTOR_NODES=2
# n_gen_nodes=$((NUM_ACTOR_NODES / 2))
n_gen_nodes=1

seq_len=2048
lag=1

Tcp=1
Ttp=1
Gtp=1
inflight=true
areal=false

nprompt=16
grpo_group_size=32

logprob_batch_size=2


rollout_size=$((nprompt * grpo_group_size))
training_gbs=${rollout_size}

sp=True
if [ ${Ttp} -eq 1 ]; then
    sp=False
fi

n_train_nodes=$((NUM_ACTOR_NODES - n_gen_nodes))

COMMAND="uv run ./examples/run_grpo_math.py \
--config examples/configs/grpo_math_4B_megatron.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
grpo.val_period=1000 \
grpo.num_prompts_per_step=${nprompt} \
grpo.num_generations_per_prompt=${grpo_group_size} \
policy.logprob_batch_size=${logprob_batch_size} \
checkpointing.enabled=false \
grpo.async_grpo.enabled=true \
grpo.async_grpo.max_trajectory_age_steps=${lag} \
grpo.async_grpo.in_flight_weight_updates=${inflight} \
grpo.async_grpo.recompute_kv_cache_after_weight_updates=${areal} \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.async_engine=true \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=${n_gen_nodes} \
policy.generation.colocated.resources.gpus_per_node=8 \
policy.max_total_sequence_length=${seq_len} \
policy.megatron_cfg.tensor_model_parallel_size=${Ttp} \
policy.megatron_cfg.sequence_parallel=${sp} \
policy.megatron_cfg.context_parallel_size=${Tcp} \
policy.generation.vllm_cfg.tensor_parallel_size=${Gtp} \
policy.train_global_batch_size=${training_gbs} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.7 \
grpo.max_num_steps=100 \
grpo.async_grpo.recompute_kv_cache_after_weight_updates=false \
logger.wandb_enabled=True \
logger.wandb.project='async-grpo-4b' \
logger.wandb.name='async-grpo-qwen3-4B-shape-only-False-1124-asyncgrpo-areal${areal}-lag${lag}-${n_train_nodes}T${n_gen_nodes}G-${seq_len}SL'" \
CONTAINER=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo-rl:main-a010564b.sqsh \
HF_HOME=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home/cache \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
NRL_FORCE_REBUILD_VENVS=true \
TORCH_CUDA_ARCH_LIST='9.0 10.0' \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/scratch:/scratch" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-async.test \
    --partition=batch \
    --time=00:30:00 \
    --gres=gpu:8 \
    ray_lax.sub

    # --ntasks-per-node=8 \
# logger.wandb.name='yuki-8K-${n_train_nodes}T${n_gen_nodes}G--${seq_len}SL--lag${lag}--inflight${inflight}-Tcp${Tcp}-Ttp${Ttp}-Gtp${Gtp}-rollout${rollout_size}-training${training_gbs}-logprob${logprob_batch_size}'" \