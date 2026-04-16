#!/bin/bash
# Nemotron-Omni GRPO training on MMPR-Tiny

# Automodel-omni provides the NemotronOmni model implementation
export PYTHONPATH=${PYTHONPATH}:/lustre/fsw/portfolios/coreai/users/yuekaiz/omni/automodel-omni

# Custom vLLM with NemotronH_Nano_VL_V2 support (3rdparty/vllm)
# Ray vLLM venv already has torch 2.11.0 + precompiled vllm 0.19.1
# Do NOT set NRL_FORCE_REBUILD_VENVS=true — it would revert torch to 2.10.0

mkdir -p ./results
LOG_FILE=./results/nemotron_omni_grpo_mmpr_tiny.log
exec > >(tee "${LOG_FILE}") 2>&1

# Ensure all dependencies (including mathruler) are installed
uv sync

config_file="yuekai_scripts/configs/vlm_grpo_nemotron_omni_mmpr_tiny.yaml"
uv run examples/run_vlm_grpo.py --config $config_file \
    cluster.gpus_per_node=8 \
    cluster.num_nodes=1
