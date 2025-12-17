#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=32
huggingface-cli login --token $HF_TOKEN 

n_gen_nodes=16
n_train_nodes=$((NUM_ACTOR_NODES - n_gen_nodes))

seq_len=4096
lag=8
inflight=true


g_tp=2
g_ep=1

t_tp=1
t_cp=1
t_pp=2
t_ep=8

sp=True
if [ ${t_tp} -eq 1 ]; then
    sp=False
fi


account=coreai_dlalgo_nemorl

COMMAND="uv run ./examples/run_grpo_math.py \
--config examples/configs/grpo_math_qwen30ba3b_megatron.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
grpo.async_grpo.enabled=true \
grpo.async_grpo.max_trajectory_age_steps=${lag} \
grpo.async_grpo.in_flight_weight_updates=${inflight} \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.async_engine=true \
policy.max_total_sequence_length=${seq_len} \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=${n_gen_nodes} \
policy.generation.colocated.resources.gpus_per_node=8 \
policy.generation.vllm_cfg.tensor_parallel_size=${g_tp} \
policy.generation.vllm_cfg.expert_parallel_size=${g_ep} \
policy.megatron_cfg.tensor_model_parallel_size=${t_tp} \
policy.megatron_cfg.context_parallel_size=${t_cp} \
policy.megatron_cfg.pipeline_model_parallel_size=${t_pp} \
policy.megatron_cfg.expert_model_parallel_size=${t_ep} \
policy.megatron_cfg.sequence_parallel=${sp} \
grpo.val_period=100 \
checkpointing.enabled=false \
policy.sequence_packing.enabled=True \
grpo.max_num_steps=15 \
policy.generation.vllm_cfg.enable_vllm_metrics_logger=true \
policy.generation.vllm_cfg.vllm_metrics_logger_interval=0.5 \
logger.wandb_enabled=True \
logger.wandb.project='async-grpo-benchmark' \
logger.wandb.name='async-qwen-30B-1120-benchmark-lag${lag}-inflight${inflight}-${n_train_nodes}T${n_gen_nodes}G-${seq_len}SL-Gtp${g_tp}Gep${g_ep}Ttp${t_tp}Tcp${t_cp}Tpp${t_pp}Tep${t_ep}'" \
CONTAINER=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo-rl:main-a010564b.sqsh \
HF_HOME=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/scratch/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/hf_home/cache \
NRL_FORCE_REBUILD_VENVS=true \
WANDB_API_KEY=$WANDB_API_KEY \
TORCH_CUDA_ARCH_LIST='9.0 10.0' \
MOUNTS="/scratch:/scratch" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-async.test \
    --partition=batch \
    --time=00:40:00 \
    --gres=gpu:8 \
    ray_lax.sub

# NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*" \
# NRL_NSYS_PROFILE_STEP_RANGE=2:3 \
# RAY_LOG_SYNC_FREQUENCY=30 \
    # --ntasks-per-node=8 \