#!/bin/bash
# DeepSeek-V3 (671B MoE) Megatron: PP16EP16 (train) -> TP32 (gen)
# 64 nodes total (32T32G), 512 GPUs.  Uses the perf-recipe yaml as the
# default config and overrides the testing knobs + enables
# nccl_reshard_refit, matching the script/new_refit/ matrix.
NUM_ACTOR_NODES=64
n_gen_nodes=32
n_train_nodes=$((NUM_ACTOR_NODES - n_gen_nodes))

huggingface-cli login --token $HF_TOKEN

account=coreai_dlalgo_nemorl

seq_len=1024
lag=1
inflight=true
nprompt=4
grpo_group_size=16
logprob_batch_size=2
train_gbs=$((nprompt * grpo_group_size))

t_tp=1
t_cp=1
t_pp=16
t_ep=16

g_tp=32
g_ep=1

sp=True
if [ ${t_tp} -eq 1 ]; then
    sp=False
fi

dsv3_model_path=/lustre/fsw/portfolios/coreai/users/yifuw/hf_checkpoints/dsv3/DeepSeek-V3-BF16

COMMAND="uv run ./examples/run_grpo.py \
--config examples/configs/recipes/llm/performance/grpo-deepseek-v3-64n8g-async-1off.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
policy.model_name=${dsv3_model_path} \
grpo.async_grpo.enabled=true \
grpo.async_grpo.max_trajectory_age_steps=${lag} \
grpo.async_grpo.in_flight_weight_updates=${inflight} \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.async_engine=true \
policy.max_total_sequence_length=${seq_len} \
grpo.num_prompts_per_step=${nprompt} \
grpo.num_generations_per_prompt=${grpo_group_size} \
policy.logprob_batch_size=${logprob_batch_size} \
policy.train_global_batch_size=${train_gbs} \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=${n_gen_nodes} \
policy.generation.colocated.resources.gpus_per_node=8 \
policy.generation.vllm_cfg.tensor_parallel_size=${g_tp} \
policy.generation.vllm_cfg.expert_parallel_size=${g_ep} \
policy.generation.vllm_cfg.gpu_memory_utilization=0.6 \
policy.megatron_cfg.tensor_model_parallel_size=${t_tp} \
policy.megatron_cfg.context_parallel_size=${t_cp} \
policy.megatron_cfg.pipeline_model_parallel_size=${t_pp} \
policy.megatron_cfg.expert_model_parallel_size=${t_ep} \
policy.megatron_cfg.sequence_parallel=${sp} \
grpo.val_period=100 \
checkpointing.enabled=false \
policy.sequence_packing.enabled=True \
+policy.nccl_reshard_refit=true \
grpo.max_num_steps=10 \
logger.wandb_enabled=True \
logger.wandb.project='new-refit-integration' \
logger.wandb.name='dsv3-PP${t_pp}EP${t_ep}TP${t_tp}-TP${g_tp}EP${g_ep}-${n_train_nodes}T${n_gen_nodes}G'" \
CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo_rl.0505.sqsh \
HF_HOME=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home \
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/users/youngeunk/hf_home/cache \
HF_HUB_OFFLINE=1 \
NRL_FORCE_REBUILD_VENVS=true \
WANDB_API_KEY=$WANDB_API_KEY \
TORCH_CUDA_ARCH_LIST='9.0 10.0' \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-async.test \
    --partition=batch \
    --time=00:45:00 \
    --gres=gpu:8 \
    ray.sub
