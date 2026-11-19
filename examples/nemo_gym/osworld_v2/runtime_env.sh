#!/bin/bash
set -euo pipefail

: "${OSWORLD_RUN_ROOT:?}"
: "${OSWORLD_CACHE_ROOT:?}"
: "${OSWORLD_RUNTIME_LIB_DIR:?}"
: "${OSWORLD_CHECKPOINT_DIR:?}"
: "${NANO_OMNI_MODEL_NAME:?}"
: "${NANO_OMNI_CHAT_TEMPLATE:?}"
# Each recipe is a launch_osworld_v2_cc.py flag of the same name.
case "${OSWORLD_RECIPE:=molt-b8k8-checkpoint}" in
  molt-b8k8-checkpoint|flashreinforce-b32n4) ;;
  *)
    echo "Unsupported OSWORLD_RECIPE: $OSWORLD_RECIPE" >&2
    exit 2
    ;;
esac
export OSWORLD_RECIPE
case "${OSWORLD_SANDBOX_PROVIDER:=opensandbox}" in
  opensandbox)
    if [[ -z "${OPENSANDBOX_BASE_URL:-}" && -n "${OPENSANDBOX_DOMAIN:-}" ]]; then
      export OPENSANDBOX_BASE_URL="$OPENSANDBOX_DOMAIN"
    fi
    : "${OPENSANDBOX_BASE_URL:?}"
    ;;
  agentenv)
    # submit.sh derives these so that Ray workers inherit them.
    : "${AGENTENV_URL:?}"
    : "${AGENTENV_API_KEY:?}"
    : "${AGENTENV_TEMPLATE:?}"
    if [[ -n "${AGENTENV_TLS_CA:-}" ]]; then
      test -r "${SSL_CERT_FILE:?}"
    fi
    ;;
  *)
    echo "OSWORLD_SANDBOX_PROVIDER must be opensandbox or agentenv" >&2
    exit 2
    ;;
esac
export OSWORLD_SANDBOX_PROVIDER
if [[ -n "${OSWORLD_EVAL_MODE:-}" ]]; then
  : "${OSWORLD_GRPO_VAL_DATA:?}"
else
  : "${OSWORLD_GRPO_TRAIN_DATA:?}"
fi

export OSWORLD_RL_ROOT=/opt/nemo-rl
export OSWORLD_GYM_ROOT="$OSWORLD_RL_ROOT/3rdparty/Gym-workspace/Gym"
OSWORLD_BRIDGE_ROOT="$OSWORLD_RL_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"

export OSWORLD_DRIVER_VENV="${OSWORLD_DRIVER_VENV:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker}"
export OSWORLD_DRIVER_PYTHON="$OSWORLD_DRIVER_VENV/bin/python"
OSWORLD_MANAGED_PYTHON_ROOT=""
for candidate_root in \
  /opt/uv/python \
  /root/.local/share/uv/python; do
  if [[ -x "$candidate_root/cpython-3.13.14-linux-x86_64-gnu/bin/python3" ]]; then
    OSWORLD_MANAGED_PYTHON_ROOT="$candidate_root"
    break
  fi
done
if [[ -z "${OSWORLD_COMPONENT_PYTHON:-}" ]]; then
  for candidate in \
    "${OSWORLD_MANAGED_PYTHON_ROOT:+$OSWORLD_MANAGED_PYTHON_ROOT/cpython-3.13.14-linux-x86_64-gnu/bin/python3}" \
    /opt/nemo_rl_venv/bin/python \
    "$OSWORLD_DRIVER_PYTHON"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      export OSWORLD_COMPONENT_PYTHON="$candidate"
      break
    fi
  done
fi
: "${OSWORLD_COMPONENT_PYTHON:?No container-owned Python runtime found}"

export PATH="$OSWORLD_DRIVER_VENV/bin:$PATH"
# Gym component locks require a newer uv than the container ships.
if [[ -n "${OSWORLD_UV_BIN_DIR:-}" ]]; then
  test -x "$OSWORLD_UV_BIN_DIR/uv"
  export PATH="$OSWORLD_UV_BIN_DIR:$PATH"
fi
export PYTHONPATH="$OSWORLD_RL_ROOT:$OSWORLD_GYM_ROOT:$OSWORLD_BRIDGE_ROOT/src:$OSWORLD_BRIDGE_ROOT/3rdparty/Megatron-LM"
export NEMO_GYM_PYTHON="$OSWORLD_COMPONENT_PYTHON"
unset UV_PYTHON
if [[ -n "$OSWORLD_MANAGED_PYTHON_ROOT" ]]; then
  export UV_PYTHON_INSTALL_DIR="$OSWORLD_MANAGED_PYTHON_ROOT"
  export UV_PYTHON_PREFERENCE=only-managed
else
  unset UV_PYTHON_INSTALL_DIR
  export UV_PYTHON_PREFERENCE=only-system
  export UV_PYTHON_DOWNLOADS=never
fi

for runtime_lib in libopenblas.so.0 libgfortran.so.5 libquadmath.so.0 libgcc_s.so.1; do
  test -r "$OSWORLD_RUNTIME_LIB_DIR/$runtime_lib"
done
export LD_LIBRARY_PATH="$OSWORLD_RUNTIME_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_VENV_DIR="${OSWORLD_GYM_VENV_DIR:-$OSWORLD_CACHE_ROOT/gym-venvs}"
export NEMO_GYM_EXTRA_ROOTS="$OSWORLD_GYM_ROOT:$OSWORLD_RL_ROOT"
export NRL_FORCE_REBUILD_VENVS=false
export NRL_IGNORE_VERSION_MISMATCH=1
export NEMO_LENS_ENABLED=0

if [[ -n "${OSWORLD_RLVR_FUNCTIONS_ROOT:-}" ]]; then
  test -d "$OSWORLD_RLVR_FUNCTIONS_ROOT"
  export OSWORLD_RLVR_FUNCTIONS_ROOT
fi
if [[ -n "${OSWORLD_CANONICAL_HARNESS_ROOT:-}" ]]; then
  : "${OSWORLD_CANONICAL_AGENT_SHA256:?}"
  : "${OSWORLD_CANONICAL_FEEDBACK_SHA256:?}"
  test -r "$OSWORLD_CANONICAL_HARNESS_ROOT/mm_agents/nvidia/nemotron_agent.py"
  test -r "$OSWORLD_CANONICAL_HARNESS_ROOT/desktop_env/execution_feedback.py"
  export OSWORLD_CANONICAL_HARNESS_ROOT OSWORLD_CANONICAL_AGENT_SHA256 OSWORLD_CANONICAL_FEEDBACK_SHA256
fi
# submit.sh mounts the evaluator tool rootfs and puts its wrappers on PATH.
if [[ -n "${OSWORLD_EVALUATOR_TOOLS_ROOT:-}" ]]; then
  test -x "$OSWORLD_EVALUATOR_TOOLS_ROOT/usr/bin/ffprobe"
  test -x "$OSWORLD_EVALUATOR_TOOLS_ROOT/usr/bin/pdftotext"
  test -d "$OSWORLD_EVALUATOR_TOOLS_ROOT/runtime-libs"
  test -x /usr/lib/libreoffice/program/soffice
  : "${OSWORLD_EVALUATOR_PYTHON:?}"
fi

export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.005}"
export NRL_REFIT_NUM_BUFFERS="${NRL_REFIT_NUM_BUFFERS:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# The Gym agent prefetches task-specific OSWorld files from Hugging Face before
# the first rollout. Override an offline setting inherited from the container;
# fully cached deployments can opt back in explicitly.
export HF_HUB_OFFLINE="${OSWORLD_HF_HUB_OFFLINE:-0}"

export OSWORLD_POOL_REF="${OSWORLD_POOL_REF:-osworld-kvm}"
export OSWORLD_MAX_STEPS="${OSWORLD_MAX_STEPS:-150}"
export OSWORLD_MAX_PARALLEL_ROLLOUTS="${OSWORLD_MAX_PARALLEL_ROLLOUTS:-8}"
export OPENSANDBOX_API_KEY="${OPENSANDBOX_API_KEY:-}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export SWANLAB_MODE="${SWANLAB_MODE:-disabled}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${OSWORLD_HF_HOME:-$OSWORLD_CACHE_ROOT/huggingface}"
export XDG_CACHE_HOME="$OSWORLD_CACHE_ROOT/xdg"
export WANDB_DIR="$OSWORLD_RUN_ROOT/wandb"
export WANDB_CACHE_DIR="$OSWORLD_CACHE_ROOT/wandb"
export WANDB_CONFIG_DIR="$OSWORLD_CACHE_ROOT/wandb-config"
export WANDB_DATA_DIR="$OSWORLD_CACHE_ROOT/wandb-data"
export SWANLAB_SAVE_DIR="$OSWORLD_CACHE_ROOT/swanlab"
export TORCHINDUCTOR_CACHE_DIR="$OSWORLD_CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$OSWORLD_CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$OSWORLD_CACHE_ROOT/cuda"
export TORCH_EXTENSIONS_DIR="$OSWORLD_CACHE_ROOT/torch-extensions"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0}"

mkdir -p \
  "$OSWORLD_RUN_ROOT/exp_001" \
  "$OSWORLD_CHECKPOINT_DIR" \
  "$NEMO_GYM_VENV_DIR" \
  "$HF_HOME" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR" \
  "$WANDB_DATA_DIR" \
  "$SWANLAB_SAVE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" \
  "$TORCH_EXTENSIONS_DIR"

test -x "$OSWORLD_DRIVER_PYTHON"
test -x "$OSWORLD_COMPONENT_PYTHON"
test -r "$NANO_OMNI_MODEL_NAME/config.json"
test -r "$NANO_OMNI_CHAT_TEMPLATE"
if [[ ! -f "$OSWORLD_BRIDGE_ROOT/src/megatron/bridge/__init__.py" ]] \
  || [[ ! -f "$OSWORLD_BRIDGE_ROOT/3rdparty/Megatron-LM/megatron/core/__init__.py" ]]; then
  echo "Megatron-Bridge submodules are unavailable under $OSWORLD_BRIDGE_ROOT" >&2
  exit 2
fi
if [[ -n "${OSWORLD_EVAL_MODE:-}" ]]; then
  test -s "$OSWORLD_GRPO_VAL_DATA"
else
  test -s "$OSWORLD_GRPO_TRAIN_DATA"
fi
