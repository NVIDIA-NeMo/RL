#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# Megatron backend requires TORCH_CUDA_ARCH_LIST. In local SSH environments
# this may be unset, so infer from nvidia-smi when possible.
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | xargs || true)
    if [[ -n "$GPU_CC" ]]; then
        export TORCH_CUDA_ARCH_LIST="$GPU_CC"
    else
        export TORCH_CUDA_ARCH_LIST="9.0"
    fi
    echo "[INFO] TORCH_CUDA_ARCH_LIST was unset; using $TORCH_CUDA_ARCH_LIST"
fi

# DeepEP/TE builds under mcore can consume large temporary space; prefer FSx
# over rootfs /tmp for local smoke tests.
export TMPDIR=${TMPDIR:-/fsx/peng/workspace/tmp}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/fsx/peng/workspace/.uv-cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/fsx/peng/workspace/.cache/torch_extensions}
export MAX_JOBS=${MAX_JOBS:-1}
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
echo "[INFO] TMPDIR=$TMPDIR UV_CACHE_DIR=$UV_CACHE_DIR TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR MAX_JOBS=$MAX_JOBS"

# Avoid mixed CUDA runtime discovery (e.g., libcudart.so.12 + libcudart.so.13),
# which can happen on custom SSH environments with multiple CUDA toolkits.
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.9}
if [[ ! -d "$CUDA_ROOT" ]]; then
    CUDA_ROOT=/usr/local/cuda-12.4
fi
if [[ ! -d "$CUDA_ROOT" ]]; then
    CUDA_ROOT=/usr/local/cuda-12.1
fi
export CUDA_HOME="$CUDA_ROOT"
unset CUDA_PATH
VENV_SITE_PKG="$PROJECT_ROOT/venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.12/site-packages"
VENV_NVIDIA_LIBS=""
if [[ -d "$VENV_SITE_PKG/nvidia" ]]; then
    # Include TE/PyTorch wheel-provided CUDA libs (cudnn, cublas, cudart, ...)
    # so we don't mix incompatible system CUDA runtimes.
    while IFS= read -r libdir; do
        if [[ -z "$VENV_NVIDIA_LIBS" ]]; then
            VENV_NVIDIA_LIBS="$libdir"
        else
            VENV_NVIDIA_LIBS="$VENV_NVIDIA_LIBS:$libdir"
        fi
    done < <(find "$VENV_SITE_PKG/nvidia" -maxdepth 3 -type d -name lib | sort)
fi
export LD_LIBRARY_PATH="$VENV_NVIDIA_LIBS:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
echo "[INFO] CUDA_HOME=$CUDA_HOME"


# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=100
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=25
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_dpo.py \
    --config $CONFIG_PATH \
    dpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=true \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    # Smoke checks: run completed and loss is finite/reasonable.
    uv run tests/check_metrics.py $JSON_METRICS \
        'data["train/loss"]["10"] > 0.0' \
        'data["train/loss"]["10"] < 20.0'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
