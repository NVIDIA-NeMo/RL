#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=256

huggingface-cli login --token $HF_TOKEN 

account=coreai_dlalgo_nemorl

n_gen_nodes=128
n_train_nodes=$((NUM_ACTOR_NODES - n_gen_nodes))

lag=4

g_tp=32
g_ep=1

t_tp=1
t_cp=1
t_pp=16
t_ep=16

sp=True
if [ ${t_tp} -eq 1 ]; then
    sp=False
fi


# NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*" \
# NRL_NSYS_PROFILE_STEP_RANGE=2:3 \
# RAY_LOG_SYNC_FREQUENCY=30 \
# NRL_REFIT_BUFFER_MEMORY_RATIO=0.02 \
# NRL_REFIT_NUM_BUFFERS=3 \
# --config examples/configs/recipes/llm/performance/grpo-deepseek-v3-64n8g-async-1off.yaml \

COMMAND="uv run ./examples/run_grpo_math.py \
--config examples/configs/grpo_math_dsv3_megatron.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.async_grpo.enabled=true \
grpo.async_grpo.max_trajectory_age_steps=${lag} \
grpo.async_grpo.in_flight_weight_updates=true \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.async_engine=true \
policy.logprob_batch_size=2 \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=${n_gen_nodes} \
policy.generation.colocated.resources.gpus_per_node=8 \
policy.generation.vllm_cfg.tensor_parallel_size=${g_tp} \
policy.generation.vllm_cfg.expert_parallel_size=${g_ep} \
policy.megatron_cfg.pipeline_model_parallel_size=${t_pp} \
policy.megatron_cfg.expert_model_parallel_size=${t_ep} \
policy.megatron_cfg.tensor_model_parallel_size=${t_tp} \
policy.megatron_cfg.context_parallel_size=${t_cp} \
policy.megatron_cfg.sequence_parallel=${sp} \
policy.megatron_cfg.num_layers_in_first_pipeline_stage=3 \
policy.megatron_cfg.num_layers_in_last_pipeline_stage=2 \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='async-grpo-benchmark' \
logger.wandb.name='async-grpo-dsv3-1203-${NUM_ACTOR_NODES}n8g-tp${t_tp}-lp2-async-${lag}lag-${n_train_nodes}T${n_gen_nodes}G-Gtp${g_tp}Gep${g_ep}Ttp${t_tp}Tcp${t_cp}Tpp${t_pp}Tep${t_ep}'" \
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
    --time=01:30:00 \
    --gres=gpu:8 \
    ray_lax.sub

    # --ntasks-per-node=8 \