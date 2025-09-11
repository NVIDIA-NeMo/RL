
## Ask @terry for container

## One time setup
```sh
git submodule update --init --recursive
# inside container build and checkout custom vllm
tools/build-custom-vllm.sh
```

## Run



```sh
# Using dtensor backend, VLLM_USE_PRECOMPILED=1 is needed to prevent vllm from re-compiling
VLLM_USE_PRECOMPILED=1 NRL_FORCE_REBUILD_VENVS=true NRL_VLLM_USE_V1=0 uv run examples/run_grpo_math.py --config examples/configs/grpo_math_1B.yaml \
    policy.model_name=nvidia/Nemotron-H-8B-Base-8K \
    policy.tokenizer.name=nvidia/Nemotron-H-8B-Base-8K \
    cluster.gpus_per_node=8 \
    cluster.num_nodes=1 \
    checkpointing.enabled=true \
    checkpointing.save_period=5 \
    grpo.val_period=5 \
    logger.wandb.project=grpo-dev-vlm-yifu \
    logger.wandb_enabled=True \
    logger.wandb.name=test-nh-dtensor \
    policy.dtensor_cfg.tensor_parallel_size=8 \
    policy.generation.vllm_cfg.tensor_parallel_size=8 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.6 \
    +policy.generation.vllm_cfg.enable_prefix_caching=False

```