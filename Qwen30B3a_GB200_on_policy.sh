#!/bin/bash
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=6

# huggingface-cli login --token $HF_TOKEN 


g_tp=4
g_ep=1
g_pp=1

t_tp=1
t_ep=24
t_pp=1

if [ ${t_tp} -eq 1 ]; then
    t_sp=False
else
    t_sp=True
fi

account=coreai_dlalgo_nemorl

COMMAND="NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo_math.py \
--config examples/configs/grpo_math_qwen30ba3b_megatron.yaml \
cluster.num_nodes=${NUM_ACTOR_NODES} \
cluster.gpus_per_node=4 \
policy.generation.vllm_cfg.tensor_parallel_size=${g_tp} \
policy.generation.vllm_cfg.expert_parallel_size=${g_ep} \
policy.generation.vllm_cfg.pipeline_parallel_size=${g_pp} \
policy.megatron_cfg.tensor_model_parallel_size=${t_tp} \
policy.megatron_cfg.expert_model_parallel_size=${t_ep} \
policy.megatron_cfg.pipeline_model_parallel_size=${t_pp} \
policy.megatron_cfg.sequence_parallel=${t_sp} \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.num_prompts_per_step=64 \
grpo.num_generations_per_prompt=32 \
policy.sequence_packing.enabled=True \
policy.train_global_batch_size=512 \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='sync-grpo-gb200' \
logger.wandb.name='sync-qwen-30B3a-seg${NUM_ACTOR_NODES}-Gtp${g_tp}-Gep${g_ep}-Gpp${g_pp}-Ttp${t_tp}-Tep${t_ep}-Tpp${t_pp}-${n_train_nodes}T${n_gen_nodes}G'" \
CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl.sqsh \
HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home \
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/cache \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=${account} \
    --job-name=${account}-sync.test \
    --partition=batch \
    --time=04:00:00 \
    --gres=gpu:4 \
    --segment ${NUM_ACTOR_NODES} \
    ray.sub