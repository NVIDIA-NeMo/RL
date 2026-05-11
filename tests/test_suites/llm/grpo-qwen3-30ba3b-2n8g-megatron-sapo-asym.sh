#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env
# ignore tensor parallel accuracy check
export NRL_IGNORE_TP_ACCURACY_CHECK=1

# Arm 2/3 of the SAPO vs GRPO reproduction: SAPO with ASYMMETRIC tau
# (tau_pos=1.0, tau_neg=1.05). Replaces the hard PPO clip with a smooth
# temperature-controlled sigmoid gate. This is the recommended SAPO
# setting per arXiv:2511.20347 (Eq. 5-6 + Fig. 5 asymmetric-tau ablation).
# See examples/configs/recipes/llm/grpo-qwen3-30ba3b-2n8g-megatron-sapo-repro-base.yaml
# for the shared base config the three arms inherit from.

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
NUM_NODES=2
STEPS_PER_RUN=30
MAX_STEPS=30
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=180
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_grpo.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    # Smoke checks: SAPO training is stable, reward is in a sane range,
    # and the soft gate is actually doing something (i.e. not all 1.0).
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        'data["train/reward"]["30"] > 0.40' \
        'data["train/reward"]["30"] < 1.0' \
        'data["train/sapo_gate_norm_mean"]["30"] > 0.0' \
        'data["train/sapo_gate_norm_mean"]["30"] < 1.0'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
