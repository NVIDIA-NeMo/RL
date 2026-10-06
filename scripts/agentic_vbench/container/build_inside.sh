#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Reproduce docker/Dockerfile's base -> hermetic -> release stages inside a
# writable Enroot container (no Docker daemon on this cluster). Runs as root
# inside the container. SRC_DIR is the NeMo-RL checkout to bake in; the
# resulting rootfs is exported by build_container.sbatch.
set -exu -o pipefail
: "${SRC_DIR:?Path to the NeMo-RL checkout (mounted read-only)}"
# Keep every cache inside the image rootfs (the Dockerfile's layout); the
# worker venvs hardlink into /root/.cache/uv.
export HOME=/root
export UV_CACHE_DIR=/root/.cache/uv
mkdir -p "$UV_CACHE_DIR"
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles
UV_VERSION=${UV_VERSION:-0.11.28}
PYTHON_VERSION=${PYTHON_VERSION:-3.13.14}
CMAKE_VERSION=${CMAKE_VERSION:-4.0.3}
export MAX_JOBS=${MAX_JOBS:-24}
export NVTE_BUILD_MAX_JOBS=${NVTE_BUILD_MAX_JOBS:-16}
export NVTE_BUILD_THREADS_PER_JOB=${NVTE_BUILD_THREADS_PER_JOB:-2}
export NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS:-"90;100"}
export NVTE_FRAMEWORK=pytorch
export NVTE_WITH_NCCL_EP=0

# ---- base stage -----------------------------------------------------------
apt-get update
apt-get install -y --no-install-recommends jq curl git rsync wget less vim ccache protobuf-compiler gnupg \
  libibverbs-dev libnvidia-ml-dev
apt-get remove -y --purge ffmpeg || true
apt-get autoremove -y
ARCH=$(uname -m)
CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}"
curl --retry 3 --retry-delay 2 -fsSL -o "/tmp/${CMAKE_INSTALLER}.tar.gz" \
  "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}.tar.gz"
tar -xzf "/tmp/${CMAKE_INSTALLER}.tar.gz" -C /tmp
cp -r "/tmp/${CMAKE_INSTALLER}/bin/"* /usr/local/bin/
cp -r "/tmp/${CMAKE_INSTALLER}/share/"* /usr/local/share/
rm -rf "/tmp/${CMAKE_INSTALLER}" "/tmp/${CMAKE_INSTALLER}.tar.gz"
export PATH="/root/.cargo/bin:$PATH"
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal
fi
export UV_PYTHON_INSTALL_DIR=/opt/uv/python
if ! command -v uv >/dev/null; then
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
fi
uv python install "${PYTHON_VERSION}"

# ---- environment shared by hermetic + release stages ------------------------
export NRL_CONTAINER=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_VENV_DIR=/opt/gym_venvs
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_LINK_MODE=copy
export TORCH_CUDA_ARCH_LIST="9.0 10.0"
export HYBRID_EP_MULTINODE=1
export CUDA_HOME=/usr/local/cuda
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl
export CUDNN_HOME=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export CUDNN_PATH=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH="/opt/nemo_rl_venv/lib/python3.13/site-packages/z3/lib:/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
# Persist the same variables for runtime shells (ray.sub reads several of them).
cat > /etc/profile.d/nemo-rl.sh <<'PROFILE'
export NRL_CONTAINER=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_VENV_DIR=/opt/gym_venvs
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_LINK_MODE=copy
export UV_PYTHON_INSTALL_DIR=/opt/uv/python
export TORCH_CUDA_ARCH_LIST="9.0 10.0"
export HYBRID_EP_MULTINODE=1
export CUDA_HOME=/usr/local/cuda
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl
export CUDNN_HOME=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export CUDNN_PATH=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH="/opt/nemo_rl_venv/lib/python3.13/site-packages/z3/lib:/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PATH="/opt/nemo_rl_venv/bin:/root/.local/bin:${PATH}"
PROFILE
mkdir -p /root/.local/bin && ln -sfn /usr/local/bin/uv /root/.local/bin/uv

# ---- source ---------------------------------------------------------------
mkdir -p /opt/nemo-rl
rsync -a --delete \
  --exclude='.venv' --exclude='__pycache__' --exclude='results' --exclude='logs' \
  --exclude='data/agentic_vbench' --exclude='*.sqsh' \
  "${SRC_DIR}/" /opt/nemo-rl/
cd /opt/nemo-rl
git config --global --add safe.directory '*'

# ---- hermetic stage: dependency warmup + worker venvs ------------------------
uv venv --seed
uv cache clean deep-ep || true
UV_LINK_MODE=hardlink uv sync --frozen --no-install-project
UV_LINK_MODE=hardlink uv sync --frozen --extra vllm --no-install-project
UV_LINK_MODE=hardlink uv sync --frozen --extra mcore --no-install-project
UV_LINK_MODE=hardlink uv sync --frozen --extra automodel --no-install-project
UV_LINK_MODE=hardlink uv sync --frozen --extra modelopt --no-install-project
UV_LINK_MODE=hardlink uv sync --frozen --all-groups --no-install-project
"${UV_PROJECT_ENVIRONMENT}/bin/python" nemo_rl/distributed/actor_environments.py all sglang trtllm > /opt/actor_venvs.tsv
test -s /opt/actor_venvs.tsv
while IFS=$'\t' read -r venv_name stage extras; do
  venv_path="${NEMO_RL_VENV_DIR}/${venv_name}"
  uv venv --allow-existing "$venv_path"
  UV_PROJECT_ENVIRONMENT="$venv_path" UV_LINK_MODE=hardlink uv sync --frozen $extras --no-install-project
done < /opt/actor_venvs.tsv

# ---- release stage: editable project installs + wrapper scripts ----------------
UV_LINK_MODE=hardlink uv sync --frozen --all-groups
UV_LINK_MODE=hardlink uv run --no-sync nemo_rl/utils/prefetch_venvs.py $(cut -f1 /opt/actor_venvs.tsv)
"${UV_PROJECT_ENVIRONMENT}/bin/python" tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint

apt-get purge -y libnvidia-ml-dev
apt-get autoremove -y
apt-get clean
rm -rf /var/lib/apt/lists/*
find /root/.cache/uv -type d -path "*ray/_private/runtime_env/agent/thirdparty_files/aiohttp*" -exec rm -rf {} + || true
echo "BUILD_INSIDE_OK"
