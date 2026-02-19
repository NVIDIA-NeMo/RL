#!/bin/bash
# Qwen3-235B-A22B with HybridEP + CUDA Graph Performance Test
# GB200 cluster: 16 nodes x 4 GPUs = 64 GPUs
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# Unset conda environment variables to prevent x86_64 conda on ARM64 (GB200) nodes
# which causes "Exec format error" when uv tries to inspect the Python interpreter
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE _CE_CONDA _CE_M

# Disable NVLS to avoid OOM issue
export NCCL_NVLS_ENABLE=0

# HybridEP environment variables for DeepEP JIT compilation
# USE_MNNVL=1 enables multi-node NVLink support for HybridEP
export USE_MNNVL=1
unset NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN

# CUDA environment for JIT compilation (container should have these, but set explicitly for safety)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:${PATH}

# ===== BEGIN CONFIG =====
NUM_NODES=16
GPUS_PER_NODE=4
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=180
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
# NRL_FORCE_REBUILD_VENVS=true ensures deep_ep (HybridEP) is installed
cd $PROJECT_ROOT
NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo_math.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        'data["train/token_mult_prob_error"]["10"] < 1.1'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
