#!/bin/bash
# Qwen/Qwen3-30B-A3B
# Target: 8 GPUs (2 Nodes @ 4 GPUs/node)
# Scaling: Half of H100 16 GPUs

NUM_NODES=2
GPUS_PER_NODE=4

# Generation Parallelism
g_tp=2
g_ep=1
g_pp=1

# Training Parallelism
# Total GPUs = 8.
# Target MP = TP * PP = 2 * 1 = 2.
# Expert Parallelism: EP=4.
# DP logic: Total Replicas = Total GPUs / MP = 8/2 = 4.
# EP=4 fully utilizes the replicas.
t_tp=2
t_ep=4
t_pp=1
t_cp=1
t_vpp=1

if [ ${t_tp} -eq 1 ]; then t_sp=False; else t_sp=True; fi

account=coreai_dlalgo_nemorl
CONFIG_FILE="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml"

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
logger.wandb.name='Qwen30B_A3B-N${NUM_NODES}xG${GPUS_PER_NODE}-Train(tp${t_tp}.pp${t_pp}.ep${t_ep}.cp${t_cp}.vpp${t_vpp})-Gen(tp${g_tp}.pp${g_pp}.ep${g_ep})'" \
CONTAINER=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl/nemo_rl.sqsh \
HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home \
HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home/cache \
WANDB_API_KEY=$WANDB_API_KEY \
MOUNTS="/lustre:/lustre" \
sbatch \
    --nodes=${NUM_NODES} \
    --account=${account} \
    --job-name=qwen30b-a3b-N${NUM_NODES}xG${GPUS_PER_NODE}-T.tp${t_tp}.pp${t_pp}.ep${t_ep}-G.tp${g_tp}.pp${g_pp} \
    --partition=batch \
    --time=04:00:00 \
    --gres=gpu:4 \
    --segment ${NUM_NODES} \
    ray.sub
