#!/bin/bash
# Qwen/Qwen3-32B
# Target: 32 GPUs (8 Nodes @ 4 GPUs/node)
# Scaling: Half of H100 64 GPUs

NUM_NODES=8
GPUS_PER_NODE=4

# Generation Parallelism
# TP=4 fits within one GB200 node (4 GPUs)
g_tp=4
g_ep=1
g_pp=1

# Training Parallelism
# Total GPUs = 32.
# Target MP = TP * PP = 4 * 4 = 16.
# DP = 32 / 16 = 2.
t_tp=4
t_ep=1
t_pp=4
t_cp=1
t_vpp=1

if [ ${t_tp} -eq 1 ]; then t_sp=False; else t_sp=True; fi

account=coreai_dlalgo_nemorl
CONFIG_FILE="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n8g.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file $CONFIG_FILE not found."
fi

COMMAND="NRL_FORCE_REBUILD_VENVS=true uv run ./examples/run_grpo_math.py \
--config ${CONFIG_FILE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
policy.generation.vllm_cfg.tensor_parallel_size=${g_tp} \
policy.generation.vllm_cfg.expert_parallel_size=${g_ep} \
policy.generation.vllm_cfg.pipeline_parallel_size=${g_pp} \
policy.megatron_cfg.tensor_model_parallel_size=${t_tp} \
policy.megatron_cfg.expert_model_parallel_size=${t_ep} \
policy.megatron_cfg.pipeline_model_parallel_size=${t_pp} \
policy.megatron_cfg.context_parallel_size=${t_cp} \
policy.megatron_cfg.num_layers_per_virtual_pipeline_stage=${t_vpp} \
policy.megatron_cfg.sequence_parallel=${t_sp} \
grpo.async_grpo.enabled=false \
grpo.val_period=1000 \
checkpointing.enabled=false \
grpo.num_prompts_per_step=64 \
grpo.num_generations_per_prompt=32 \
policy.sequence_packing.enabled=True \
policy.train_global_batch_size=512 \
grpo.max_num_steps=20 \
logger.wandb_enabled=True \
logger.wandb.project='sync-grpo-gb200-benchmark' \
logger.wandb.name='Qwen32B-N${NUM_NODES}-Gtp${g_tp}-Ttp${t_tp}pp${t_pp}'"

CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl.sqsh
HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/cache
MOUNTS="/lustre:/lustre"

sbatch \
    --nodes=${NUM_NODES} \
    --account=${account} \
    --job-name=qwen32b-sync \
    --partition=batch \
    --time=04:00:00 \
    --gres=gpu:4 \
    --segment ${NUM_NODES} \
    ray.sub
