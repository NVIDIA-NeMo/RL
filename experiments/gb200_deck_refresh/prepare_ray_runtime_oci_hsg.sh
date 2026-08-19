#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-test-only}
BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
REPO=${REPO:-${BASE}/RL-gb200-deck-refresh-20260818}
CONTAINER=${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/mkar/containers/nemo-rl-nightly-ngc-20260815_212622.sqsh}
CACHE_ROOT=${CACHE_ROOT:-${BASE}/.cache/gb200-deck-refresh}
PYTHON_ROOT=${PYTHON_ROOT:-${CACHE_ROOT}/python-3.13.14}
RAY_RUNTIME_VENV=${RAY_RUNTIME_VENV:-${CACHE_ROOT}/ray-runtime-py31314}
UV_CACHE_DIR=${UV_CACHE_DIR:-${CACHE_ROOT}/uv-cache}
ACCOUNT=${SLURM_ACCOUNT:-nemotron_sw_post}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-short}
LOG_DIR=${LOG_DIR:-${BASE}/experiments/gb200-deck-refresh/runtime-setup}

case "${ACTION}" in
  test-only) SBATCH_ACTION=(--test-only) ;;
  submit) SBATCH_ACTION=() ;;
  *) echo "ACTION must be test-only or submit" >&2; exit 2 ;;
esac

test -f "${CONTAINER}"
test -f "${REPO}/ray.sub"
mkdir -p "${LOG_DIR}" "${CACHE_ROOT}" "${UV_CACHE_DIR}"

export SETUP_COMMAND=$(cat <<EOF
set -euo pipefail
RUNTIME='${RAY_RUNTIME_VENV}'
if [[ -x "\${RUNTIME}/bin/python" ]] && \
   "\${RUNTIME}/bin/python" -c \
     'import ray, requests, sys, urllib3; assert sys.version_info[:3] == (3, 13, 14); assert ray.__version__ == "2.56.1"' 2>/dev/null; then
  "\${RUNTIME}/bin/python" --version
  "\${RUNTIME}/bin/ray" --version
  exit 0
fi

UV=\$(command -v uv)
rm -rf "\${RUNTIME}"
"\${UV}" python install 3.13.14 \
  --install-dir '${PYTHON_ROOT}' \
  --no-bin \
  --cache-dir '${UV_CACHE_DIR}'
PYTHON=\$(find '${PYTHON_ROOT}' -path '*/bin/python3.13' -type f | head -1)
test -n "\${PYTHON}"
"\${UV}" venv --python "\${PYTHON}" "\${RUNTIME}"
"\${UV}" pip install \
  --python "\${RUNTIME}/bin/python" \
  --cache-dir '${UV_CACHE_DIR}' \
  'ray[default,client,data]==2.56.1'
"\${RUNTIME}/bin/python" -c \
  'import ray, requests, sys, urllib3; assert sys.version_info[:3] == (3, 13, 14); assert ray.__version__ == "2.56.1"; print(ray.__version__, requests.__version__, urllib3.__version__)'
EOF
)

export BASE_LOG_DIR=${LOG_DIR}
export COMMAND="${RAY_RUNTIME_VENV}/bin/python -c 'import ray, requests, urllib3; assert ray.__version__ == \"2.56.1\"'"
export CONTAINER
export CONTAINER_REMAP_ROOT=1
export GPUS_PER_NODE=4
export MOUNTS=/lustre:/lustre
export UV_CACHE_DIR_OVERRIDE=${UV_CACHE_DIR}

exec sbatch "${SBATCH_ACTION[@]}" \
  --nodes=1 \
  --gres=gpu:4 \
  --exclusive \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --time=00:20:00 \
  --segment=1 \
  --job-name="${ACCOUNT}-gb200-deck.ray-runtime" \
  --output="${LOG_DIR}/slurm-%j.out" \
  "${REPO}/ray.sub"
