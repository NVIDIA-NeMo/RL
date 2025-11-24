set -x

source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh
MOUNTS="\
$PWD:$PWD,\
/lustre:/lustre"
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


ACCOUNT=${ACCOUNT:-"coreai_dlalgo_nemorl"}
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
CONTAINER="/lustre/fsw/portfolios/coreai/users/zhiyul/containers/nemo-rl-bd2e645-37644239-repro.sqsh" \
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