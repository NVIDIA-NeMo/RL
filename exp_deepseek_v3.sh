#!/bin/bash
# Deepseek-V3
# Target: 256 GPUs (64 Nodes @ 4 GPUs/node)
# Scaling: Half of H100 512 GPUs

NUM_NODES=64
GPUS_PER_NODE=4

# Generation Parallelism
g_tp=64
g_ep=1
g_pp=1

# Training Parallelism
# Total GPUs = 256.
# Target MP = TP * PP = 1 * 8 = 8.
# DP = 256 / 8 = 32.
# EP=32 (Matches DP).
t_tp=1
t_ep=32
t_pp=8
t_cp=1
t_vpp=1

if [ ${t_tp} -eq 1 ]; then t_sp=False; else t_sp=True; fi

account=coreai_dlalgo_nemorl
CONFIG_FILE="examples/configs/recipes/llm/performance/grpo-deepseek-v3-32n8g.yaml"

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
logger.wandb.name='DeepseekV3-N${NUM_NODES}-Gtp${g_tp}-Ttp${t_tp}pp${t_pp}ep${t_ep}'"

CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl.sqsh
HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/cache
MOUNTS="/lustre:/lustre"

sbatch \
    --nodes=${NUM_NODES} \
    --account=${account} \
    --job-name=deepseekv3-sync \
    --partition=batch \
    --time=04:00:00 \
    --gres=gpu:4 \
    --segment 16 \
    ray.sub
