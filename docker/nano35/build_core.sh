#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Runs in a fresh writable ARM64 CUDA image; inputs live at /opt/nemo-rl.
set -euo pipefail
[[ $(uname -m) == aarch64 ]] || { echo 'ARM64 builder required' >&2; exit 1; }
cd /opt/nemo-rl
source docker/nano35/environment.sh
export DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles
export UV_LINK_MODE=hardlink
export HYBRID_EP_MULTINODE=1
export NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=100 NVTE_WITH_NCCL_EP=0
export NVTE_BUILD_MAX_JOBS=16 NVTE_BUILD_THREADS_PER_JOB=2
export MAX_JOBS=24 TRTLLM_BUILD_JOBS=24 BUILD_CUSTOM_TRTLLM_ARCH=100-real
export UV_HTTP_TIMEOUT=180 UV_HTTP_RETRIES=5
mkdir -p /opt/nano35-metadata

apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl git git-lfs rsync wget jq ccache protobuf-compiler \
    build-essential libnvidia-ml-dev libnuma-dev libopenmpi-dev openmpi-bin \
    libhdf5-dev pkg-config patchelf tmux nodejs npm umoci skopeo pigz zstd

CMAKE_VERSION=4.0.3
curl --retry 3 --retry-delay 2 -fsSL \
    "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz" \
    -o /tmp/nano35-cmake.tar.gz
tar -xzf /tmp/nano35-cmake.tar.gz -C /opt
ln -sf "/opt/cmake-${CMAKE_VERSION}-linux-aarch64/bin/cmake" /usr/local/bin/cmake
ln -sf "/opt/cmake-${CMAKE_VERSION}-linux-aarch64/bin/ctest" /usr/local/bin/ctest
rm /tmp/nano35-cmake.tar.gz

curl --retry 3 -LsSf https://astral.sh/uv/0.11.28/install.sh \
    | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
uv python install 3.13.14
uv venv --python 3.13.14 --seed "$UV_PROJECT_ENVIRONMENT"
uv lock --check
uv sync --locked
printf 'driver-ready\n' > /opt/nano35-metadata/build-stage

# Build each required actor in its actual isolated environment. All of these
# environments share the uv cache through hardlinks in this image layer.
uv run --no-sync nemo_rl/utils/prefetch_venvs.py \
    nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker \
    nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector \
    nemo_rl.algorithms.async_utils.ReplayBuffer \
    nemo_rl.environments.nemo_gym.NemoGym
printf 'training-collector-gym-actors-ready\n' > /opt/nano35-metadata/build-stage
uv run --no-sync nemo_rl/utils/prefetch_venvs.py \
    nemo_rl.models.generation.trtllm.trtllm_worker_async.TrtllmAsyncGenerationWorker
printf 'trt-actor-ready\n' > /opt/nano35-metadata/build-stage

bash docker/install_apptainer.sh
python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint
uv --version > /opt/nano35-metadata/uv-version.txt
uv pip freeze --python /opt/nemo_rl_venv/bin/python > /opt/nano35-metadata/driver-packages.txt
while IFS= read -r actor; do
    [[ -x "$actor/bin/python" ]] || continue
    uv pip freeze --python "$actor/bin/python" \
        > "/opt/nano35-metadata/$(basename "$actor").packages.txt"
done < <(find /opt/ray_venvs -mindepth 1 -maxdepth 1 -type d | sort)
dpkg-query -W > /opt/nano35-metadata/os-packages.txt
git rev-parse HEAD > /opt/nano35-metadata/nemo-rl-commit.txt
git submodule status --recursive > /opt/nano35-metadata/submodules.txt
git diff --binary > /opt/nano35-metadata/nemo-rl.patch
sha256sum pyproject.toml uv.lock tools/trtllm-nano35.patch \
    > /opt/nano35-metadata/source-checksums.txt

python -c 'import torch, nemo_rl; print("driver", torch.__version__, torch.version.cuda)'
python-MegatronPolicyWorker -c 'import torch, megatron.core, transformer_engine; print("training", torch.__version__, transformer_engine.__version__)'
python-TrtllmAsyncGenerationWorker -c 'import torch, tensorrt_llm; from tensorrt_llm._torch.async_llm import AsyncLLM; print("inference", torch.__version__, tensorrt_llm.__version__)'
python-AsyncTrajectoryCollector -c 'import nemo_rl.algorithms.async_utils; print("collector import passed")'
apptainer --version
singularity --version
printf 'core-imports-passed\n' > /opt/nano35-metadata/build-stage
apt-get clean
rm -rf /var/lib/apt/lists/*
printf 'Core image build passed; SWE harness prebuild and GPU E2E validation remain required.\n'
