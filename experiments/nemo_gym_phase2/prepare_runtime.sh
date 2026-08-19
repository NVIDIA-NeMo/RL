#!/bin/bash

set -euo pipefail

: "${PHASE2_REPO:?PHASE2_REPO must be set}"
: "${PHASE2_RUNTIME_ROOT:?PHASE2_RUNTIME_ROOT must be set}"
: "${PHASE2_UV_BIN_DIR:?PHASE2_UV_BIN_DIR must be set}"
: "${PHASE2_RL_INSIGHT_SOURCE:?PHASE2_RL_INSIGHT_SOURCE must be set}"

export PATH=$PHASE2_UV_BIN_DIR:$PATH
uv --version | grep -E '^uv 0\.11\.28([[:space:]]|$)'

LOCK_SHA256=$(sha256sum "$PHASE2_REPO/uv.lock")
LOCK_SHA256=${LOCK_SHA256%% *}
PYPROJECT_SHA256=$(sha256sum "$PHASE2_REPO/pyproject.toml")
PYPROJECT_SHA256=${PYPROJECT_SHA256%% *}
RUNTIME_ID=${LOCK_SHA256:0:12}-${PYPROJECT_SHA256:0:12}
VENV_ROOT=$PHASE2_RUNTIME_ROOT/venvs-$RUNTIME_ID
UV_CACHE=$PHASE2_RUNTIME_ROOT/uv-cache-$RUNTIME_ID
PYTHON_INSTALL_DIR=$PHASE2_RUNTIME_ROOT/python-$RUNTIME_ID
VLLM_ENV=$VENV_ROOT/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
NEMO_GYM_ENV=$VENV_ROOT/nemo_rl.environments.nemo_gym.NemoGym
MANIFEST=$PHASE2_RUNTIME_ROOT/runtime-$RUNTIME_ID.env

mkdir -p "$VENV_ROOT" "$UV_CACHE" "$PYTHON_INSTALL_DIR"
export UV_CACHE_DIR=$UV_CACHE
export UV_LINK_MODE=copy
export UV_PYTHON_INSTALL_DIR=$PYTHON_INSTALL_DIR
# DeepGEMM defaults to appending a local-version suffix when uv's source build
# directory has no Git metadata. The checked-in dependency metadata and lock
# correctly identify this immutable source as 2.5.0, so build that exact wheel
# version instead of creating a perpetual 2.5.0 -> 2.5.0+local sync drift.
export DG_USE_LOCAL_VERSION=0
uv python install 3.13.14 --install-dir "$PYTHON_INSTALL_DIR" --no-bin
PYTHON_INTERPRETER=$(uv python find 3.13.14)
[[ $PYTHON_INTERPRETER == "$PYTHON_INSTALL_DIR"/* ]] || {
  echo "uv selected a Python outside the persistent runtime: $PYTHON_INTERPRETER" >&2
  exit 1
}

sync_environment() {
  local destination=$1
  shift
  uv venv --allow-existing --python "$PYTHON_INTERPRETER" "$destination"
  UV_PROJECT_ENVIRONMENT=$destination uv sync \
    --frozen --directory "$PHASE2_REPO"
  UV_PROJECT_ENVIRONMENT=$destination uv sync \
    --frozen --directory "$PHASE2_REPO" "$@"
}

sync_environment "$VLLM_ENV" --extra vllm
DEEP_GEMM_VERSION=$("$VLLM_ENV/bin/python" -c \
  'import importlib.metadata as m; print(m.version("deep-gemm"))')
if [[ $DEEP_GEMM_VERSION != 2.5.0 ]]; then
  echo "Rebuilding deep-gemm to match uv.lock: $DEEP_GEMM_VERSION -> 2.5.0"
  uv cache clean deep-gemm
  UV_PROJECT_ENVIRONMENT=$VLLM_ENV uv sync \
    --frozen --directory "$PHASE2_REPO" \
    --extra vllm --reinstall-package deep-gemm
  DEEP_GEMM_VERSION=$("$VLLM_ENV/bin/python" -c \
    'import importlib.metadata as m; print(m.version("deep-gemm"))')
fi
[[ $DEEP_GEMM_VERSION == 2.5.0 ]] || {
  echo "deep-gemm build version differs from uv.lock: $DEEP_GEMM_VERSION" >&2
  exit 1
}
sync_environment "$NEMO_GYM_ENV" \
  --extra nemo_gym --group nemo_gym_router

# `uv pip check` does not understand root-level override-dependencies or
# exclude-dependencies. This repository intentionally uses both (for example
# for NCCL, cuDNN, llguidance, setuptools, OpenCV, and CUTLASS libs-base), so a
# raw pip-style metadata check reports known false positives. Verify the
# selected lock projection with uv itself, then fail on every unmet package
# requirement that is not explicitly covered by the checked-in uv policy.
VLLM_VERIFICATION=$PHASE2_RUNTIME_ROOT/runtime-$RUNTIME_ID-vllm-verification.json
NEMO_GYM_VERIFICATION=$PHASE2_RUNTIME_ROOT/runtime-$RUNTIME_ID-nemo-gym-verification.json
"$VLLM_ENV/bin/python" \
  "$PHASE2_REPO/experiments/nemo_gym_phase2/verify_runtime.py" \
  --repo "$PHASE2_REPO" \
  --environment "$VLLM_ENV" \
  --python-install-dir "$PYTHON_INSTALL_DIR" \
  --uv-bin "$PHASE2_UV_BIN_DIR/uv" \
  --output "$VLLM_VERIFICATION" \
  --label vllm \
  --extra vllm
"$NEMO_GYM_ENV/bin/python" \
  "$PHASE2_REPO/experiments/nemo_gym_phase2/verify_runtime.py" \
  --repo "$PHASE2_REPO" \
  --environment "$NEMO_GYM_ENV" \
  --python-install-dir "$PYTHON_INSTALL_DIR" \
  --uv-bin "$PHASE2_UV_BIN_DIR/uv" \
  --output "$NEMO_GYM_VERIFICATION" \
  --label nemo_gym \
  --extra nemo_gym \
  --group nemo_gym_router
grep -E '^uv = 0\.11\.28$' "$VLLM_ENV/pyvenv.cfg"
grep -E '^uv = 0\.11\.28$' "$NEMO_GYM_ENV/pyvenv.cfg"

PYTHONPATH=$PHASE2_REPO:$PHASE2_RL_INSIGHT_SOURCE \
  "$VLLM_ENV/bin/python" -c \
  'import nemo_rl, ray, rl_insight, vllm; print("driver ray=" + ray.__version__ + " vllm=" + vllm.__version__ + " rl_insight=" + rl_insight.__version__)'
PYTHONPATH=$PHASE2_REPO:$PHASE2_REPO/3rdparty/Gym-workspace/Gym \
  "$NEMO_GYM_ENV/bin/python" -c \
  'import importlib.metadata as m, nemo_gym, ray, vllm_router; print("gym ray=" + ray.__version__ + " router=" + m.version("vllm-router"))'

DRIVER_RAY=$(PYTHONPATH=$PHASE2_REPO "$VLLM_ENV/bin/python" -c \
  'import ray; print(ray.__version__)')
GYM_RAY=$(PYTHONPATH=$PHASE2_REPO:$PHASE2_REPO/3rdparty/Gym-workspace/Gym \
  "$NEMO_GYM_ENV/bin/python" -c 'import ray; print(ray.__version__)')
[[ $DRIVER_RAY == "$GYM_RAY" ]] || {
  echo "Driver/Gym Ray version mismatch: $DRIVER_RAY != $GYM_RAY" >&2
  exit 1
}

{
  printf 'export PHASE2_RUNTIME_ID=%q\n' "$RUNTIME_ID"
  printf 'export PHASE2_RUNTIME_ENV=%q\n' "$VLLM_ENV"
  printf 'export PHASE2_NEMO_GYM_ENV=%q\n' "$NEMO_GYM_ENV"
  printf 'export PHASE2_VENV_DIR=%q\n' "$VENV_ROOT"
  printf 'export PHASE2_UV_CACHE_DIR=%q\n' "$UV_CACHE"
  printf 'export PHASE2_PYTHON_INSTALL_DIR=%q\n' "$PYTHON_INSTALL_DIR"
  printf 'export PHASE2_UV_LOCK_SHA256=%q\n' "$LOCK_SHA256"
  printf 'export PHASE2_PYPROJECT_SHA256=%q\n' "$PYPROJECT_SHA256"
  printf 'export PHASE2_RAY_VERSION=%q\n' "$DRIVER_RAY"
  printf 'export PHASE2_VLLM_ENV_VERIFICATION=%q\n' "$VLLM_VERIFICATION"
  printf 'export PHASE2_NEMO_GYM_ENV_VERIFICATION=%q\n' "$NEMO_GYM_VERIFICATION"
} > "$MANIFEST"

printf '%s\n' "$MANIFEST"
