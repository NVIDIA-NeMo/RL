#!/bin/bash

. init_env.sh

VLLM_LOGGING_LEVEL=DEBUG \
NCCL_NVLS_ENABLE=0 \
uv run python examples/run_eval.py \
    generation.model_name=Qwen/Qwen3-30B-A3B-Instruct-2507 \
    generation.vllm_cfg.async_engine=true \
    generation.vllm_cfg.expert_parallel_size=2 \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=8
