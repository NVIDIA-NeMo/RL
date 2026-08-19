#!/bin/bash

set -euo pipefail

: "${PHASE2_REPO:?PHASE2_REPO must be set}"
: "${PHASE2_RUNTIME_ENV:?PHASE2_RUNTIME_ENV must be set}"
: "${PHASE2_NEMO_GYM_ENV:?PHASE2_NEMO_GYM_ENV must be set}"
: "${PHASE2_RL_INSIGHT_SOURCE:?PHASE2_RL_INSIGHT_SOURCE must be set}"
: "${PHASE2_PROMETHEUS_BIN:?PHASE2_PROMETHEUS_BIN must be set}"
: "${PHASE2_UV_BIN_DIR:?PHASE2_UV_BIN_DIR must be set}"
: "${PHASE2_PYTHON_INSTALL_DIR:?PHASE2_PYTHON_INSTALL_DIR must be set}"
: "${PHASE2_UV_CACHE_DIR:?PHASE2_UV_CACHE_DIR must be set}"
: "${PHASE2_VLLM_ENV_VERIFICATION:?PHASE2_VLLM_ENV_VERIFICATION must be set}"
: "${PHASE2_NEMO_GYM_ENV_VERIFICATION:?PHASE2_NEMO_GYM_ENV_VERIFICATION must be set}"

export PATH=$PHASE2_UV_BIN_DIR:$PHASE2_RUNTIME_ENV/bin:$PATH
export UV_PYTHON_INSTALL_DIR=$PHASE2_PYTHON_INSTALL_DIR
export DG_USE_LOCAL_VERSION=0
export PYTHONPATH=$PHASE2_REPO:$PHASE2_REPO/3rdparty/Gym-workspace/Gym:$PHASE2_RL_INSIGHT_SOURCE
PYTEST_PYTHONPATH=$PYTHONPATH
if [[ -n ${PHASE2_PYTEST_PYTHONPATH:-} ]]; then
  PYTEST_PYTHONPATH=$PHASE2_PYTEST_PYTHONPATH:$PYTEST_PYTHONPATH
fi

uv --version | grep -F "uv 0.11.28"
UV_CACHE_DIR=$PHASE2_UV_CACHE_DIR "$PHASE2_RUNTIME_ENV/bin/python" \
  "$PHASE2_REPO/experiments/nemo_gym_phase2/verify_runtime.py" \
  --repo "$PHASE2_REPO" \
  --environment "$PHASE2_RUNTIME_ENV" \
  --python-install-dir "$PHASE2_PYTHON_INSTALL_DIR" \
  --uv-bin "$PHASE2_UV_BIN_DIR/uv" \
  --output "$PHASE2_VLLM_ENV_VERIFICATION" \
  --label vllm \
  --extra vllm
UV_CACHE_DIR=$PHASE2_UV_CACHE_DIR "$PHASE2_NEMO_GYM_ENV/bin/python" \
  "$PHASE2_REPO/experiments/nemo_gym_phase2/verify_runtime.py" \
  --repo "$PHASE2_REPO" \
  --environment "$PHASE2_NEMO_GYM_ENV" \
  --python-install-dir "$PHASE2_PYTHON_INSTALL_DIR" \
  --uv-bin "$PHASE2_UV_BIN_DIR/uv" \
  --output "$PHASE2_NEMO_GYM_ENV_VERIFICATION" \
  --label nemo_gym \
  --extra nemo_gym \
  --group nemo_gym_router
PYTHONPATH=$PYTEST_PYTHONPATH "$PHASE2_RUNTIME_ENV/bin/python" -m pytest --version
"$PHASE2_RUNTIME_ENV/bin/python" -c \
  'import nemo_rl, ray, rl_insight, vllm; print("nemo_rl=" + nemo_rl.__file__); print("driver ray=" + ray.__version__ + " vllm=" + vllm.__version__ + " rl_insight=" + rl_insight.__version__)'
"$PHASE2_NEMO_GYM_ENV/bin/python" -c \
  'import importlib.metadata as m, nemo_gym, ray, vllm_router; print("nemo_gym=" + nemo_gym.__file__); print("gym ray=" + ray.__version__ + " router=" + m.version("vllm-router"))'
DRIVER_RAY=$("$PHASE2_RUNTIME_ENV/bin/python" -c 'import ray; print(ray.__version__)')
GYM_RAY=$("$PHASE2_NEMO_GYM_ENV/bin/python" -c 'import ray; print(ray.__version__)')
[[ $DRIVER_RAY == "$GYM_RAY" ]]
PYTHONPATH=$PYTEST_PYTHONPATH "$PHASE2_RUNTIME_ENV/bin/python" -m pytest -q \
  tests/unit/evals/test_eval.py \
  tests/unit/evals/test_run_eval.py \
  tests/unit/test_run_grpo_rollout_benchmark.py \
  tests/unit/environments/test_prometheus.py \
  tests/unit/environments/test_vllm_router.py \
  tests/unit/tools/test_nemo_gym_phase2_verify_runtime.py \
  tests/unit/tools/test_nemo_gym_phase2_validate_ray_runtime_env.py \
  tests/unit/tools/test_nemo_gym_phase2_report.py \
  tests/unit/tools/test_nemo_gym_phase2_compare.py

OBSERVABILITY_ID=${SLURM_JOB_ID:-$$}
OBSERVABILITY_ROOT=${PHASE2_OBSERVABILITY_VALIDATION_ROOT:-${SLURM_TMPDIR:-/tmp}/nemo-gym-phase2-observability-$OBSERVABILITY_ID}
OBSERVABILITY_OUTPUT=${PHASE2_OBSERVABILITY_VERIFICATION:-$(dirname "$PHASE2_VLLM_ENV_VERIFICATION")/observability-validation-$OBSERVABILITY_ID.json}
"$PHASE2_RUNTIME_ENV/bin/python" \
  "$PHASE2_REPO/experiments/nemo_gym_phase2/validate_observability.py" \
  --prometheus-bin "$PHASE2_PROMETHEUS_BIN" \
  --prometheus-base-config \
  "$PHASE2_REPO/examples/nemo_gym/rl_insight_phase2/prometheus.yml" \
  --stack-root "$OBSERVABILITY_ROOT" \
  --output "$OBSERVABILITY_OUTPUT"
