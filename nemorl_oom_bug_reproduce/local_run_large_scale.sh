# install

# apt-get update && apt-get install -y libxrender1 libxext6 libsm6 libxrandr2 libxfixes3 libxi6

source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh
export HF_HOME=/lustre/fsw/portfolios/coreai/users/zhiyul/hf

script_dir=$(dirname "$0")
cd $script_dir

export NRL_FORCE_REBUILD_VENVS=true
# Required for Megatron backend - specify CUDA architectures
# H100 is compute capability 9.0, A100 is 8.0, etc.
export TORCH_CUDA_ARCH_LIST='9.0 10.0'

CONFIG_NAME=configs/config_qwen3_235B_instruct.yaml

# large scale
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=3600 \
NCCL_PROTO=simple \
NCCL_NVLS_ENABLE=0 \
NCCL_BUFFSIZE=33554432 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
NCCL_SHM_DISABLE=1 \
uv run python ether1_train.py --config $CONFIG_NAME \
    cluster.num_nodes=16 \
    logger.wandb_enabled=true \
    logger.wandb.name='grpo-qwen-grpo-memory-oom-large-scale' \
    logger.wandb.project='grpo-dev-zhiyul' \
    checkpointing.checkpoint_dir=results/grpo-qwen-grpo-memory-oom-large-scale \
    grpo.val_at_start=false