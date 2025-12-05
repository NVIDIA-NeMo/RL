set -x

source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh
MOUNTS="\
$PWD:$PWD,\
/lustre:/lustre,\
/lustre/fsw/portfolios/coreai/users/yifuw/nemo_rl_checkpoints:/opt/checkpoints:ro,\
/lustre/fsw/portfolios/coreai/users/yuya/root/checkpoints:/opt/checkpoints_dvs3:ro\
"
# /lustre/fsw/portfolios/coreai/users/yifuw/nemo_rl_checkpoints:/opt/checkpoints:ro,\
# /lustre/fsw/portfolios/coreai/users/yuya/root/checkpoints/tron/deepseek-ai/DeepSeek-V3:/opt/checkpoints/tron/deepseek-ai/DeepSeek-V3:ro\

# MOUNTS="\
# $PWD:$PWD,\
# /lustre:/lustre,\
# /lustre/fsw/portfolios/coreai/users/zhiyul/nemo_rl_checkpoints:/opt/checkpoints\
# "

UV_CACHE_DIR=/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/uv_cache
NNODES=${NNODES:-1}

# overwrite the batch size to 64 to avoid gradient accumulation
# COMMAND=${COMMAND:-"uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml policy.train_global_batch_size=64 cluster.num_nodes=${NNODES} checkpointing.checkpoint_dir='results/llama8b_${NNODES}nodes' logger.wandb_enabled=True logger.wandb.name='grpo-llama8b_math'"}
COMMAND=${COMMAND:-""}

# Default to short run
LONG_RUN=${LONG_RUN:-false}

# Set partition and time based on LONG_RUN
if [ "$LONG_RUN" = true ]; then
  # Long run settings
  PARTITION="batch"
  TIME="4:00:00"
elif [ "$MIDDLE_RUN" = true ]; then
  PARTITION="batch_short"
  TIME="2:00:00"
else
  # Short run settings
  PARTITION="interactive"
  TIME="2:00:00"
fi


# CONTAINER="/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:flash-attn-51202eb.squashfs" \
# CONTAINER="/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:vllm-torch-bump-d7dfc91-dirty.squashfs" \
# CONTAINER="/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:main-3e5481f.squashfs" \  # 2025-06
# CONTAINER="/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:flash-attn-51202eb.squashfs" \
# CONTAINER="/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:vllm-torch-bump-d7dfc91-dirty.squashfs" \
# CONTAINER="/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:flash-attn-51202eb.squashfs" \
# CONTAINER="/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:flash-attn-2.7.4-f09cb2a2.squashfs" \
# CONTAINER="/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/gitlab-master.nvidia.com/terryk/images/nemo-rl:flash-attn-2.7.4-f09cb2a2.squashfs" \
# CONTAINER="/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/nvcr.io/nvidian/nemo-rl:bd2e645-37644239.squashfs" \
ACCOUNT=${ACCOUNT:-"coreai_dlalgo_nemorl"}
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
CONTAINER="/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-rl/nemo-rl-v0.4.0.sqsh" \
WANDB_API_KEY=${WANDB_API_KEY} \
HF_HOME=/lustre/fsw/portfolios/coreai/users/zhiyul/hf \
MOUNTS=$MOUNTS \
COMMAND=$COMMAND \
sbatch -N ${NNODES} \
    -t ${TIME} \
    --dependency=singleton \
    --account=${ACCOUNT} \
    --partition=${PARTITION} \
    --gres=gpu:8 \
    --job-name=${ACCOUNT}-rl:8b_vllm_reinforcer_${PARTITION}_0 \
    ray.sub

# ps -aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9