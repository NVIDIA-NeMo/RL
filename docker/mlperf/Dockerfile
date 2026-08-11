# syntax=docker/dockerfile:1
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Qwen3.5 397B GRPO (nemo-rl) benchmark container. Stages base/hermetic/release
# are vendored from nemo-rl docker/Dockerfile (its own uv-managed stack on the
# Rubin TRT-LLM base — not the dgx pytorch base), plus the MLPerf overlay from
# the qwen35 fork's docker/mlperf/Dockerfile.gym_qwen35.
#
# NeMo-RL is built from upstream NVIDIA-NeMo/RL main at image build time
# (override NEMO_RL_REVISION with a SHA for a reproducible build); the qwen35
# commits on top of it are carried in patches/nemo-rl/ (see its README and
# README.md source audit). A patch that no longer applies fails the build.
#
# The base ARG is NOT named FROM_IMAGE_NAME: ci/build.yml force-passes
# FROM_IMAGE_NAME=<dgx pytorch image>, which would replace the Rubin base.

ARG NEMO_RL_BASE_IMAGE=urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-rubin-py3-sbsa-ubuntu24.04-trt10.16.1.11-20260724-upgraded
ARG NEMO_RL_REPO=https://github.com/NVIDIA-NeMo/RL.git
ARG NEMO_RL_REVISION=opt/dev-rubin
ARG HYBRID_EP_REPO=https://github.com/deepseek-ai/DeepEP.git
ARG HYBRID_EP_REVISION=e0a5b1d9848ab3e7b4a67842bf06f067bfac67f8
# 10.7 = Rubin (sm_107), matching [tool.trtllm]'s Rubin-targeted ref.
ARG HYBRID_EP_CUDA_ARCH_LIST="10.0;10.3;10.7"
ARG HYBRID_EP_MAX_JOBS=16

## Stage: base (vendored from nemo-rl docker/Dockerfile)
FROM ${NEMO_RL_BASE_IMAGE} AS base
ENV NRL_CONTAINER=1
USER root

RUN <<"EOF" bash -exu -o pipefail
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles
apt-get update
apt-get install -y --no-install-recommends \
    jq \
    curl \
    git \
    patch \
    rsync \
    wget \
    less \
    vim \
    ccache \
    libzmq3-dev \

# Nsight
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/$(uname -m | sed 's/aarch64/sbsa/')/7fa2af80.pub
apt update
apt install -y nsight-systems-cli-2026.3.1

apt install -y --only-upgrade gnupg
apt-get install -y --only-upgrade rsync
apt-get remove -y --purge ffmpeg || true
apt-get autoremove -y
apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

# CMake (for CUDA extension builds: transformer-engine, mamba-ssm, etc.)
RUN CMAKE_VERSION=4.0.3 \
    && ARCH=$(uname -m) \
    && CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}" \
    && curl --retry 3 --retry-delay 2 -fsSL -o "${CMAKE_INSTALLER}.tar.gz" \
        "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}.tar.gz" \
    && tar -xzf "${CMAKE_INSTALLER}.tar.gz" \
    && cp -r "${CMAKE_INSTALLER}/bin/"* /usr/local/bin/ \
    && cp -r "${CMAKE_INSTALLER}/share/"* /usr/local/share/ \
    && rm -rf "${CMAKE_INSTALLER}" "${CMAKE_INSTALLER}.tar.gz"

ARG UV_VERSION=0.11.28
ARG PYTHON_VERSION=3.13.14
ENV PATH="/root/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    uv python install ${PYTHON_VERSION}

# KV cache transceiver (TRT-LLM prefill/decode disaggregation).
#
# TensorRT-LLM compiles every non-MPI cache-transceiver backend out of the wheel
# unless it is built against UCX, and the NIXL backend additionally needs NIXL
# headers/libs at cmake time (cpp/CMakeLists.txt only runs find_package(NIXL)
# inside its ENABLE_UCX block). backend=DEFAULT resolves to NIXL, so an image
# without NIXL can only serve an explicit backend=UCX.
#
# UCX itself ships in the NGC base image under /usr/local/ucx, but cmake's
# find_package(ucx) does not search that prefix on its own -- hence ucx_ROOT.
# libzmq (installed above) is the UCX wrapper's control plane: a hard cmake
# requirement to build, and a runtime dependency on every node that runs an
# engine, since the wrapper is dlopen()ed on the first KV transfer.
ENV ucx_ROOT=/usr/local/ucx
ENV LD_LIBRARY_PATH="/usr/local/ucx/lib:${LD_LIBRARY_PATH}"

ARG NIXL_VERSION=v1.3.1
RUN <<"EOF" bash -exu -o pipefail
if [[ ! -d /usr/local/ucx/lib/cmake/ucx ]]; then
    echo "UCX missing from the base image; NIXL's UCX plugin cannot build" >&2
    exit 1
fi

# NIXL builds with meson/ninja and links Python bindings against pybind11. Like
# the other from-source builds in this repo (tools/build-custom-flashinfer.sh,
# tools/build-custom-vllm.sh), the build-time Python deps go into a uv venv
# rather than the system interpreter -- and this one is thrown away in the same
# layer, so nothing survives into the runtime environment.
uv venv /tmp/nixl-build-venv
uv pip install --python /tmp/nixl-build-venv/bin/python meson ninja pybind11 setuptools
export PATH="/tmp/nixl-build-venv/bin:${PATH}"

GDS_PATH=/usr/local/cuda/targets/x86_64-linux
if [[ "$(uname -m)" != "x86_64" ]]; then
    GDS_PATH=/usr/local/cuda/targets/sbsa-linux
fi
# NIXL links against the CUDA driver stub, which does not live on the default
# loader path inside the build container.
CUDA_SO_DIR=$(dirname "$(find /usr/local -name libcuda.so.1 | head -n1)")
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_SO_DIR}"

cd /tmp
git clone --depth 1 -b "${NIXL_VERSION}" https://github.com/ai-dynamo/nixl.git
cd nixl
meson setup builddir \
    -Ducx_path=/usr/local/ucx \
    -Dcudapath_lib=/usr/local/cuda/lib64 \
    -Dcudapath_inc=/usr/local/cuda/include \
    -Dgds_path="${GDS_PATH}" \
    -Dinstall_headers=true \
    -Ddisable_plugins=POSIX \
    -Dbuild_tests=false \
    -Dbuild_examples=false \
    --buildtype=release
cd builddir && ninja install
cd /tmp && rm -rf /tmp/nixl /tmp/nixl-build-venv

test -f /opt/nvidia/nvda_nixl/include/nixl.h
EOF
# Both arch dirs are listed so this stays correct on x86_64 and aarch64; the
# missing one is simply skipped by the loader.
ENV LD_LIBRARY_PATH="/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:/opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu:/opt/nvidia/nvda_nixl/lib64:${LD_LIBRARY_PATH}"


ENV RAY_USAGE_STATS_ENABLED=0
ENV RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs
ENV NEMO_GYM_VENV_DIR=/opt/gym_venvs

## Stage: nemo-rl-src — clone nemo-rl at the pinned repo/revision
FROM base AS nemo-rl-src

## 1. Clone nemo-rl (+ 3rdparty submodules) and record provenance.
# Submodule ARGs: SKIP = keep the pin from the nemo-rl revision.
ARG NEMO_RL_BASE_IMAGE
ARG NEMO_RL_REPO
ARG NEMO_RL_REVISION
ARG NEMO_GYM_REVISION=SKIP
ARG MBRIDGE_REVISION=SKIP
ARG AUTOMODEL_REVISION=SKIP
RUN git config --global user.name "a" && git config --global user.email "a" && \
    git clone ${NEMO_RL_REPO} /opt/nemo-rl-src && \
    cd /opt/nemo-rl-src && \
    git checkout ${NEMO_RL_REVISION} && \
    git submodule update --init --recursive && \
    if [ "${NEMO_GYM_REVISION}" != SKIP ]; then \
        git -C 3rdparty/Gym-workspace/Gym fetch --unshallow origin || git -C 3rdparty/Gym-workspace/Gym fetch origin && \
        git -C 3rdparty/Gym-workspace/Gym checkout ${NEMO_GYM_REVISION}; fi && \
    if [ "${MBRIDGE_REVISION}" != SKIP ]; then \
        git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge fetch --unshallow origin || git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge fetch origin && \
        git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge checkout ${MBRIDGE_REVISION}; fi && \
    if [ "${AUTOMODEL_REVISION}" != SKIP ]; then \
        git -C 3rdparty/Automodel-workspace/Automodel fetch --unshallow origin || git -C 3rdparty/Automodel-workspace/Automodel fetch origin && \
        git -C 3rdparty/Automodel-workspace/Automodel checkout ${AUTOMODEL_REVISION}; fi && \
    echo NEMO_RL_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /NEMO_RL_COMMIT_HASH.env && \
    echo $(git -C 3rdparty/Gym-workspace/Gym rev-parse HEAD) > /NEMO_GYM_COMMIT_HASH.env && \
    echo $(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge rev-parse HEAD) > /MBRIDGE_COMMIT_HASH.env && \
    echo $(git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    echo $(git -C 3rdparty/Automodel-workspace/Automodel rev-parse HEAD) > /AUTOMODEL_COMMIT_HASH.env && \
    printf '%s\n' "${NEMO_RL_REPO}" > /NEMO_RL_REPO.env && \
    printf '%s\n' "${NEMO_RL_REVISION}" > /NEMO_RL_REVISION.env && \
    printf '%s\n' "${NEMO_RL_BASE_IMAGE}" > /NEMO_RL_BASE_IMAGE.env && \
    git submodule status --recursive > /NEMO_RL_SUBMODULE_STATUS.txt

## 2. NeMo Gym patches. Guarded: already-applied is ok, a real conflict fails.
COPY patches/gym/ /tmp/patches/gym/
RUN <<"EOF" bash -exu
cd /opt/nemo-rl-src/3rdparty/Gym-workspace/Gym
for p in /tmp/patches/gym/*.patch; do
    if git apply --check "$p"; then
        git apply "$p"
    elif git apply --reverse --check "$p"; then
        echo "$(basename "$p") is already applied"
    else
        git apply --check "$p"
    fi
done
EOF

## 3. Carried NeMo-RL patches (see patches/nemo-rl/README.md)
# Temporarily disabled: patches conflict with shuyixiong/RL@support_editable_trtllm
# which already includes the Python changes (via 41354a9c). Apply locally and
# mount nemo_rl/ into the container at runtime instead.
# COPY patches/nemo-rl/ /tmp/patches/nemo-rl/
# RUN <<"EOF" bash -exu
# cd /opt/nemo-rl-src
# shopt -s nullglob
# for p in /tmp/patches/nemo-rl/*.patch; do
#     git apply --3way "$p"
# done
# EOF

## 4. Benchmark-owned files, copied into the source tree (override fork copies)
COPY mlperf_grpo_logging.py /opt/nemo-rl-src/nemo_rl/algorithms/mlperf_grpo_logging.py
COPY run_grpo_nemo_gym.py /opt/nemo-rl-src/examples/nemo_gym/run_grpo_nemo_gym.py
COPY conf/ /opt/nemo-rl-src/qwen_35/configs/

## Stage: hermetic — uv sync of the dependency stack (vendored from nemo-rl)
# Empty default; CI can override with --build-context to seed pre-compiled wheels
FROM scratch AS uv-cache-seed

FROM base AS hermetic
WORKDIR /opt/nemo-rl

ARG MAX_JOBS
ARG NVTE_BUILD_THREADS_PER_JOB
ARG BUILD_CUSTOM_VLLM
ARG BUILD_CUSTOM_VLLM_URL
ARG BUILD_CUSTOM_VLLM_REF
ARG BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION
ARG BUILD_CUSTOM_FLASHINFER
ARG BUILD_CUSTOM_FLASHINFER_URL
ARG BUILD_CUSTOM_FLASHINFER_REF
# TensorRT-LLM does not publish a cp313 release wheel on PyPI — the official
# releases target cp312 and below. Until an official cp313 wheel is available,
# we must build from source. The URL and commit ref are pinned in
# 3rdparty/TensorRT-LLM-workspace/pyproject.toml under [tool.trtllm]; the
# PEP 517 backend (_backend.py) reads them directly from there.
# BUILD_CUSTOM_TRTLLM_ARCH overrides the target GPU architecture list.
ARG BUILD_CUSTOM_TRTLLM_ARCH
ENV BUILD_CUSTOM_TRTLLM_ARCH=${BUILD_CUSTOM_TRTLLM_ARCH}
# [tool.trtllm].url points at the private tekit GitLab and embeds a
# ${GITLAB_CLONE_ACCESS_TOKEN} placeholder that _backend.py expands from the
# process environment, so the token must be present in THIS stage -- the
# `uv sync --extra trtllm` below is where the clone happens, long before the
# mlperf stage declares it. Deliberately ARG and not ENV: build args already
# reach RUN as environment variables here (see NEMO_RL_BASE_IMAGE in the sync
# heredoc), and an ENV would bake the token into the shipped image layers.
ARG GITLAB_CLONE_ACCESS_TOKEN

ARG SKIP_VLLM_BUILD
# SGLang is not used by this benchmark; skip it by default to cut build time.
ARG SKIP_SGLANG_BUILD=1
ARG SKIP_TRTLLM_BUILD
ARG NEMO_RL_BASE_IMAGE
ARG UV_VERSION

ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
ENV UV_LINK_MODE=copy
# 10.7 = Rubin (sm_107); torch extensions built during `uv sync` (mamba-ssm,
# causal-conv1d, transformer-engine) need it to emit native Rubin kernels.
ENV TORCH_CUDA_ARCH_LIST="9.0 10.0 10.3 10.7"
ENV CUDA_HOME=/usr/local/cuda
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl

# Point TE at the pip-installed cuDNN, not the system one (see nemo-rl docker/Dockerfile)
ENV CUDNN_HOME=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
ENV CUDNN_PATH=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
ENV LD_LIBRARY_PATH="/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
# Force mamba-ssm to compile from source instead of downloading a prebuilt
# wheel. On Rubin the CUDA driver reports a non-standard version string
# ('gpgpu.38400451') that packaging.version.parse() rejects with InvalidVersion,
# and mamba-ssm has no automatic fallback to source compilation.
ENV MAMBA_FORCE_BUILD=TRUE

# First copy only the dependency files (source COPY happens in release)
COPY --from=nemo-rl-src /opt/nemo-rl-src/pyproject.toml /opt/nemo-rl-src/uv.lock ./
COPY --from=nemo-rl-src /opt/nemo-rl-src/nemo_rl/__init__.py /opt/nemo-rl-src/nemo_rl/package_info.py ./nemo_rl/
COPY --from=nemo-rl-src /opt/nemo-rl-src/tools/build-custom-vllm.sh ./tools/build-custom-vllm.sh
COPY --from=nemo-rl-src /opt/nemo-rl-src/tools/build-custom-flashinfer.sh ./tools/build-custom-flashinfer.sh
COPY --from=nemo-rl-src /opt/nemo-rl-src/tools/build-custom-trtllm.sh ./tools/build-custom-trtllm.sh
COPY --from=nemo-rl-src --link /opt/nemo-rl-src/research/ ./research/
COPY --from=nemo-rl-src --link /opt/nemo-rl-src/3rdparty/ ./3rdparty/

ARG PIP_INDEX_URL=https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple
ARG UV_INDEX_URL=https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple

RUN --mount=type=ssh \
    --mount=type=bind,from=uv-cache-seed,source=.,target=/tmp/uv-cache-seed <<"EOF" bash -exu
CACHE_KEY=$(echo "${NEMO_RL_BASE_IMAGE}-${UV_VERSION}" | md5sum | cut -c1-12)
SEED_KEY=$(cat /tmp/uv-cache-seed/.cache-key 2>/dev/null || echo "")
if [ -n "$SEED_KEY" ] && [ "$SEED_KEY" != "$CACHE_KEY" ]; then
    echo "Cache key mismatch (base image or uv version changed) — skipping stale seed"
elif [ -n "$SEED_KEY" ]; then
    echo "Seeding uv cache (key=${CACHE_KEY})"
    mkdir -p /root/.cache/uv
    rsync -a --exclude='.cache-key' /tmp/uv-cache-seed/ /root/.cache/uv/ 2>/dev/null || true
fi

uv venv --seed
# The custom build scripts will alter the pyproject.toml and uv.lock
if [[ -n "${BUILD_CUSTOM_VLLM:-}" ]]; then
    UV_LINK_MODE=hardlink bash tools/build-custom-vllm.sh ${BUILD_CUSTOM_VLLM_URL:-} ${BUILD_CUSTOM_VLLM_REF:-} ${BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION:-}
    source 3rdparty/vllm/nemo-rl.env
fi
if [[ -n "${BUILD_CUSTOM_FLASHINFER:-}" ]]; then
    bash tools/build-custom-flashinfer.sh ${BUILD_CUSTOM_FLASHINFER_URL:-} ${BUILD_CUSTOM_FLASHINFER_REF:-}
fi

# Sync each training + inference backend one at a time (since they may
# conflict) to warm the uv cache, then sync the default dependencies.
UV_LINK_MODE=hardlink uv sync --locked --no-install-project
if [[ -z "${SKIP_VLLM_BUILD:-}" ]]; then
    UV_LINK_MODE=hardlink uv sync --locked --extra vllm --no-install-project
fi
if [[ -z "${SKIP_SGLANG_BUILD:-}" ]]; then
    UV_LINK_MODE=hardlink uv sync --locked --extra sglang --no-install-project
fi
if [[ -z "${SKIP_TRTLLM_BUILD:-}" ]]; then
    # The custom PEP 517 backend in 3rdparty/TensorRT-LLM-workspace/_backend.py
    # invokes tools/build-custom-trtllm.sh during this sync (≈60 min).
    # URL and ref are pinned in 3rdparty/TensorRT-LLM-workspace/pyproject.toml
    # under [tool.trtllm] — no longer passed via env vars.
    # BUILD_CUSTOM_TRTLLM_ARCH overrides the target GPU arch list if set.
    uv sync --link-mode symlink --locked --extra trtllm --no-install-project
fi
uv sync --link-mode symlink --locked --extra mcore --no-install-project
uv sync --link-mode symlink --locked --extra automodel --no-install-project
uv sync --link-mode symlink --locked --extra modelopt --no-install-project
uv sync --link-mode symlink --locked --all-groups --no-install-project

# Remove the aiohttp in this uv cache dir to fully address CVE GHSA-mqqc-3gqh-h2x8
find /root/.cache/uv -type d -path "*ray/_private/runtime_env/agent/thirdparty_files/aiohttp*" -exec rm -rf {} +

echo "${CACHE_KEY}" > /root/.cache/uv/.cache-key
EOF

ENV PATH="/opt/nemo_rl_venv/bin:$PATH"

## Stage: release — source + ray venvs (vendored from nemo-rl)
FROM hermetic AS release
ARG SKIP_VLLM_BUILD
ARG SKIP_SGLANG_BUILD=1
ARG SKIP_TRTLLM_BUILD
ARG TARGETARCH
ARG HYBRID_EP_REPO
ARG HYBRID_EP_REVISION
ARG HYBRID_EP_CUDA_ARCH_LIST
ARG HYBRID_EP_MAX_JOBS

ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

# AWS EFA OFI plugin discovery (see nemo-rl docker/Dockerfile)
RUN printf '%s\n' /opt/amazon/ofi-nccl/lib /opt/amazon/efa/lib \
        > /etc/ld.so.conf.d/aws-efa.conf
ENV LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"

# pyproject/uv.lock excluded: the custom build scripts may alter the hermetic copies
COPY --from=nemo-rl-src --exclude=pyproject.toml --exclude=uv.lock /opt/nemo-rl-src/ /opt/nemo-rl

# Provenance files (surfaced to JET by run.sub)
COPY --from=nemo-rl-src /NEMO_RL_COMMIT_HASH.env /NEMO_GYM_COMMIT_HASH.env /MBRIDGE_COMMIT_HASH.env /MCORE_COMMIT_HASH.env /AUTOMODEL_COMMIT_HASH.env /NEMO_RL_REPO.env /NEMO_RL_REVISION.env /NEMO_RL_BASE_IMAGE.env /NEMO_RL_SUBMODULE_STATUS.txt /
RUN <<"PY" python
import json
import tomllib
from pathlib import Path

lock_path = Path("/opt/nemo-rl/uv.lock")
packages = tomllib.loads(lock_path.read_text())["package"]


def records(name: str) -> list[dict]:
    selected = []
    for package in packages:
        if package["name"] != name:
            continue
        selected.append(
            {
                key: package[key]
                for key in ("version", "source", "wheels", "resolution-markers")
                if key in package
            }
        )
    if not selected:
        raise RuntimeError(f"{name} is absent from {lock_path}")
    versions = {record["version"] for record in selected}
    if len(versions) != 1:
        raise RuntimeError(f"{name} has ambiguous locked versions: {versions}")
    return selected


resolved = {
    "deep-ep": records("deep-ep"),
    "transformer-engine": records("transformer-engine"),
    "vllm": records("vllm"),
}
Path("/VLLM_VERSION.env").write_text(resolved["vllm"][0]["version"] + "\n")
Path("/TRANSFORMER_ENGINE_VERSION.env").write_text(
    resolved["transformer-engine"][0]["version"] + "\n"
)
Path("/NEMO_RL_DEPENDENCY_PROVENANCE.json").write_text(
    json.dumps(resolved, indent=2, sort_keys=True) + "\n"
)
PY
RUN sha256sum /opt/nemo-rl/uv.lock | cut -d ' ' -f 1 > /NEMO_RL_UV_LOCK_SHA256.env
RUN <<"EOF" bash -eu
{
    echo "NEMO_RL_REPO=$(cat /NEMO_RL_REPO.env)"
    echo "NEMO_RL_REVISION=$(cat /NEMO_RL_REVISION.env)"
    echo "NEMO_RL_COMMIT_HASH=$(cat /NEMO_RL_COMMIT_HASH.env)"
    echo "NEMO_GYM_COMMIT_HASH=$(cat /NEMO_GYM_COMMIT_HASH.env)"
    echo "MBRIDGE_COMMIT_HASH=$(cat /MBRIDGE_COMMIT_HASH.env)"
    echo "MCORE_COMMIT_HASH=$(cat /MCORE_COMMIT_HASH.env)"
    echo "AUTOMODEL_COMMIT_HASH=$(cat /AUTOMODEL_COMMIT_HASH.env)"
    echo "VLLM_VERSION=$(cat /VLLM_VERSION.env)"
    echo "TRANSFORMER_ENGINE_VERSION=$(cat /TRANSFORMER_ENGINE_VERSION.env)"
    echo "NEMO_RL_BASE_IMAGE=$(cat /NEMO_RL_BASE_IMAGE.env)"
    echo "NEMO_RL_UV_LOCK_SHA256=$(cat /NEMO_RL_UV_LOCK_SHA256.env)"
    echo "DEPENDENCY_LOCK_BEGIN"
    cat /NEMO_RL_DEPENDENCY_PROVENANCE.json
    echo "DEPENDENCY_LOCK_END"
    echo "SUBMODULES_BEGIN"
    cat /NEMO_RL_SUBMODULE_STATUS.txt
    echo "SUBMODULES_END"
} > /NEMO_RL_PROVENANCE.txt
EOF

# Prefetch the per-worker ray venvs into the image
RUN <<"EOF" bash -exu
# MLPERF: NeMo RL keys venv paths by actor FQN even when actors share the same
# VLLM py_executable, so prefetch one VLLM tree and alias the paths it expects.
# We also restrict venv prefetch to the ones we're actually using, all venvs together
# have O(1M) symlinks that take minutes to extract.
VLLM_ACTOR=nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
MCORE_ACTOR=nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
NEMO_GYM_ACTOR=nemo_rl.environments.nemo_gym.NemoGym
FILTERS=("${VLLM_ACTOR}" "${MCORE_ACTOR}" "${NEMO_GYM_ACTOR}")
NEGATIVE_FILTERS=""
if [[ -n "${SKIP_VLLM_BUILD:-}" ]]; then
    NEGATIVE_FILTERS="$NEGATIVE_FILTERS vllm"
fi
if [[ -n "${SKIP_SGLANG_BUILD:-}" ]]; then
    NEGATIVE_FILTERS="$NEGATIVE_FILTERS sglang"
fi
if [[ -n "$NEGATIVE_FILTERS" ]]; then
    UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py "${FILTERS[@]}" --negative-filters $NEGATIVE_FILTERS
else
    UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py "${FILTERS[@]}"
fi

VLLM_VENV="${NEMO_RL_VENV_DIR}/${VLLM_ACTOR}"
if [[ -d "${VLLM_VENV}" ]]; then
    for actor in \
        nemo_rl.algorithms.async_utils.ReplayBuffer \
        nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector \
        nemo_rl.experience.sync_rollout_actor.SyncRolloutActor; do
        ln -s "${VLLM_ACTOR}" "${NEMO_RL_VENV_DIR}/${actor}"
        ln -s "python-VllmAsyncGenerationWorker" "/usr/local/bin/python-${actor##*.}"
    done
fi

for actor in "${MCORE_ACTOR}" "${NEMO_GYM_ACTOR}"; do
    test -x "${NEMO_RL_VENV_DIR}/${actor}/bin/python"
    test -x "/usr/local/bin/python-${actor##*.}"
done
if [[ -z "${SKIP_VLLM_BUILD:-}" ]]; then
    for actor in \
        "${VLLM_ACTOR}" \
        nemo_rl.algorithms.async_utils.ReplayBuffer \
        nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector \
        nemo_rl.experience.sync_rollout_actor.SyncRolloutActor; do
        test -x "${NEMO_RL_VENV_DIR}/${actor}/bin/python"
        test -x "/usr/local/bin/python-${actor##*.}"
    done
fi
EOF

# This benchmark's supported build target is linux/arm64 GB300. The 413 arm64
# lock predates the HybridEP fixes qualified by AWS job 2427903.
# Rebuild an unmodified upstream checkout only in the Megatron policy venv;
# vLLM keeps its separate locked DeepEP installation.
RUN <<"EOF" bash -exu -o pipefail
POLICY_VENV=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
POLICY_PYTHON="${POLICY_VENV}/bin/python"
HYBRID_EP_SRC=$(mktemp -d)
trap 'rm -rf "${HYBRID_EP_SRC}"' EXIT

test -x "${POLICY_PYTHON}"
git -C "${HYBRID_EP_SRC}" init
git -C "${HYBRID_EP_SRC}" fetch --depth 1 \
    "${HYBRID_EP_REPO}" "${HYBRID_EP_REVISION}"
git -C "${HYBRID_EP_SRC}" checkout --detach FETCH_HEAD
test "$(git -C "${HYBRID_EP_SRC}" rev-parse HEAD)" = "${HYBRID_EP_REVISION}"
test -z "$(git -C "${HYBRID_EP_SRC}" status --porcelain)"

HYBRID_EP_MULTINODE=0 \
MAX_JOBS="${HYBRID_EP_MAX_JOBS}" \
TORCH_CUDA_ARCH_LIST="${HYBRID_EP_CUDA_ARCH_LIST}" \
USE_NIXL=0 \
UV_NO_CONFIG=1 \
    uv pip install \
        --python "${POLICY_PYTHON}" \
        --no-build-isolation \
        --no-deps \
        --reinstall \
        "${HYBRID_EP_SRC}"

"${POLICY_PYTHON}" - <<'PY'
import importlib.metadata as metadata
import importlib.util
from pathlib import Path

specs = {}
for module_name in ("deep_ep", "deep_ep_cpp", "hybrid_ep_cpp"):
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise SystemExit(f"{module_name} is unavailable")
    specs[module_name] = spec

if "HybridEPBuffer" not in Path(specs["deep_ep"].origin).read_text():
    raise SystemExit("deep_ep does not export HybridEPBuffer")

print("deep-ep", metadata.version("deep-ep"))
print("deep_ep_cpp", specs["deep_ep_cpp"].origin)
print("hybrid_ep_cpp", specs["hybrid_ep_cpp"].origin)
PY

DEEP_EP_VERSION=$("${POLICY_PYTHON}" -c \
    'import importlib.metadata as m; print(m.version("deep-ep"))')
printf '%s\n' "${HYBRID_EP_REVISION}" > /opt/hybrid_ep_commit
{
    echo "HYBRID_EP_POLICY_OVERRIDE_BEGIN"
    echo "HYBRID_EP_POLICY_REPO=${HYBRID_EP_REPO}"
    echo "HYBRID_EP_POLICY_REVISION=${HYBRID_EP_REVISION}"
    echo "HYBRID_EP_POLICY_CUDA_ARCH_LIST=${HYBRID_EP_CUDA_ARCH_LIST}"
    echo "HYBRID_EP_POLICY_MAX_JOBS=${HYBRID_EP_MAX_JOBS}"
    echo "HYBRID_EP_POLICY_MULTINODE=0"
    echo "HYBRID_EP_POLICY_USE_NIXL=0"
    echo "HYBRID_EP_POLICY_VENV=${POLICY_VENV}"
    echo "HYBRID_EP_INSTALLED_VERSION=${DEEP_EP_VERSION}"
    echo "HYBRID_EP_POLICY_SOURCE_CLEAN=1"
    echo "HYBRID_EP_POLICY_OVERRIDE_END"
} >> /NEMO_RL_PROVENANCE.txt
EOF

## Stage: mlperf (final) — gym overlay (vendored from Dockerfile.gym_qwen35) + benchmark bits
FROM release AS mlperf
ARG NEMO_RL_BASE_IMAGE
ARG NEMO_RL_REPO
ARG NEMO_RL_REVISION
ARG TARGETARCH
ARG GITLAB_CLONE_ACCESS_TOKEN
ARG GIT_COMMIT_ID
ARG NTRACE_REVISION=f5ad3e960709435b4837ca7947a1ecefd25fc439
ARG HYBRID_EP_REPO
ARG HYBRID_EP_REVISION
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID
LABEL com.nvidia.nemo-rl.repository="${NEMO_RL_REPO}" \
      com.nvidia.nemo-rl.revision="${NEMO_RL_REVISION}" \
      com.nvidia.hybrid-ep.repository="${HYBRID_EP_REPO}" \
      com.nvidia.hybrid-ep.revision="${HYBRID_EP_REVISION}" \
      org.opencontainers.image.base.name="${NEMO_RL_BASE_IMAGE}" \
      org.opencontainers.image.revision="${GIT_COMMIT_ID}"

USER root
WORKDIR /opt/nemo-rl
RUN test -n "${GIT_COMMIT_ID}"

# Apptainer is required for the SWE SIF sandboxes.
RUN <<"EOF" bash -exu -o pipefail
if ! command -v apptainer >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
        ca-certificates \
        software-properties-common
    add-apt-repository -y ppa:apptainer/ppa
    apt-get update
    apt-get install -y --no-install-recommends apptainer
    apt-get clean
    rm -rf /var/lib/apt/lists/*
fi
ln -sf "$(command -v apptainer)" /usr/bin/singularity
EOF

# SWE-agent frameworks (clone external repos at build; network egress
# required). SHAs below are HEAD of the fork overlay's refs (main /
# local-eval) as of 2026-07-08, pinned for reproducible builds.
COPY patches/r2e/ /tmp/patches/r2e/
RUN <<"EOF" bash -exu -o pipefail
SWE_AGENTS_DIR=/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents

# The setup scripts fetch unpinned latest uv via the astral installer, which
# by default prepends its uv dir to PATH in /root/.bashrc + /root/.profile.
# That shadows the image's pinned uv for every container shell, and uv >= 0.12
# runs scripts outside /opt/nemo-rl on a bare interpreter instead of the
# project venv (broke the benchmark driver with ModuleNotFoundError: wandb).
export UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1

SETUP_DIR="${SWE_AGENTS_DIR}/swe_openhands_setup"
mkdir -p "${SETUP_DIR}"
SETUP_DIR="${SETUP_DIR}" \
    MINIFORGE_DIR="${SETUP_DIR}/miniforge3" \
    OPENHANDS_DIR="${SETUP_DIR}/OpenHands" \
    AGENT_FRAMEWORK_REPO="https://github.com/sdevare-nv/nv-OpenHands.git" \
    AGENT_FRAMEWORK_COMMIT="0d766ad06b2be64a42e6f0175b9ebcc4a06599d9" \
        "${SWE_AGENTS_DIR}/setup_scripts/openhands.sh"

SETUP_DIR="${SWE_AGENTS_DIR}/swe_swebench_setup"
mkdir -p "${SETUP_DIR}"
SETUP_DIR="${SETUP_DIR}" \
    UV_DIR="${SETUP_DIR}/uv" \
    PYTHON_DIR="${SETUP_DIR}/python" \
    SWEBENCH_DIR="${SETUP_DIR}/SWE-bench" \
    SWEBENCH_REPO="https://github.com/HeyyyyyyG/SWE-bench.git" \
    SWEBENCH_COMMIT="d546b5e4be7e5fd5eaec6ee67b7588d85306a422" \
        "${SWE_AGENTS_DIR}/setup_scripts/swebench.sh"

SETUP_DIR="${SWE_AGENTS_DIR}/swe_swebench_multilingual_setup"
mkdir -p "${SETUP_DIR}"
SETUP_DIR="${SETUP_DIR}" \
    UV_DIR="${SETUP_DIR}/uv" \
    PYTHON_DIR="${SETUP_DIR}/python" \
    SWEBENCH_DIR="${SETUP_DIR}/SWE-bench_Multilingual" \
    SWEBENCH_REPO="https://github.com/Kipok/SWE-bench.git" \
    SWEBENCH_COMMIT="c6cd63f0f48956fa3db9ba1e0fadff3a4cd2e2c6" \
        "${SWE_AGENTS_DIR}/setup_scripts/swebench_multilingual.sh"

SETUP_DIR="${SWE_AGENTS_DIR}/swe_r2e_gym_setup"
mkdir -p "${SETUP_DIR}"
SETUP_DIR="${SETUP_DIR}" \
    UV_DIR="${SETUP_DIR}/uv" \
    PYTHON_DIR="${SETUP_DIR}/python" \
    R2E_GYM_DIR="${SETUP_DIR}/R2E-Gym" \
    EVAL_HARNESS_REPO="https://github.com/sdevare-nv/nv-R2E-Gym.git" \
    EVAL_HARNESS_COMMIT="6823e64f94ae645f5265c03af0eb2e8523530a0d" \
        "${SWE_AGENTS_DIR}/setup_scripts/r2e_gym.sh"
for p in /tmp/patches/r2e/*.patch; do
    git -C "${SETUP_DIR}/R2E-Gym" apply --check "$p"
    git -C "${SETUP_DIR}/R2E-Gym" apply "$p"
done

SETUP_DIR="${SWE_AGENTS_DIR}/swe_rebench_setup"
mkdir -p "${SETUP_DIR}"
SETUP_DIR="${SETUP_DIR}" \
    REBENCH_DIR="${SETUP_DIR}/SWE-rebench-V2" \
        "${SWE_AGENTS_DIR}/setup_scripts/swe_rebench.sh"
EOF

# Belt and braces for the PATH shadowing above: strip any installer PATH
# edits that slipped through, then assert none remain.
RUN sed -i '\#swe_agents/swe_[a-z0-9_]*_setup/uv/env#d' /root/.bashrc /root/.profile && \
    ! grep -n "_setup/uv/env" /root/.bashrc /root/.profile

# Pre-bake the Gym server venvs (inert stub paths stand in for runtime inputs)
COPY patches/installed/ /tmp/patches/installed/
ARG NEMO_GYM_PREFETCH_CONFIGS=qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml
ARG NEMO_GYM_CUDA=cu130
ARG NEMO_GYM_VLLM_VERSION=0.20.0
RUN <<"EOF" bash -exu -o pipefail
if [[ -z "${NEMO_GYM_PREFETCH_CONFIGS:-}" ]]; then
    echo "NEMO_GYM_PREFETCH_CONFIGS is empty; skipping gym venv prefetch."
    exit 0
fi
UV_OVERRIDE_ARGS=()
if [[ -n "${NEMO_GYM_VLLM_VERSION:-}" ]]; then
    case "${TARGETARCH:-amd64}" in
        arm64) WHEEL_ARCH=aarch64 ;;
        *) WHEEL_ARCH=x86_64 ;;
    esac
    GYM_VLLM_OVERRIDE=/tmp/gym-vllm-override.txt
    printf 'vllm @ https://github.com/vllm-project/vllm/releases/download/v%s/vllm-%s+%s-cp38-abi3-manylinux_2_35_%s.whl\n' \
        "${NEMO_GYM_VLLM_VERSION}" "${NEMO_GYM_VLLM_VERSION}" "${NEMO_GYM_CUDA}" "${WHEEL_ARCH}" > "${GYM_VLLM_OVERRIDE}"
    UV_OVERRIDE_ARGS=(UV_OVERRIDE="${GYM_VLLM_OVERRIDE}")
fi

PREFETCH_INPUT_ROOT=/tmp/nemo-gym-prefetch
mkdir -p "${PREFETCH_INPUT_ROOT}/model" "${PREFETCH_INPUT_ROOT}/sif"
touch "${PREFETCH_INPUT_ROOT}/train.jsonl" "${PREFETCH_INPUT_ROOT}/validation.jsonl"

env \
    CONTAINER_HF_CKPT_PATH="${CONTAINER_HF_CKPT_PATH:-${PREFETCH_INPUT_ROOT}/model}" \
    NEMO_GYM_SWE_TRAIN_DATA_PATH="${NEMO_GYM_SWE_TRAIN_DATA_PATH:-${PREFETCH_INPUT_ROOT}/train.jsonl}" \
    NEMO_GYM_SWE_VALIDATION_DATA_PATH="${NEMO_GYM_SWE_VALIDATION_DATA_PATH:-${PREFETCH_INPUT_ROOT}/validation.jsonl}" \
    NEMO_GYM_SWE_SIF_DIR="${NEMO_GYM_SWE_SIF_DIR:-${PREFETCH_INPUT_ROOT}/sif}" \
    UV_TORCH_BACKEND="${NEMO_GYM_CUDA}" \
    UV_LINK_MODE=hardlink \
    "${UV_OVERRIDE_ARGS[@]}" \
    uv run python examples/nemo_gym/prefetch_venvs.py ${NEMO_GYM_PREFETCH_CONFIGS}

rm -f "${GYM_VLLM_OVERRIDE:-}"

# Gym's dry-run spinup ignores its setup subprocesses' exit codes: a failed
# install bakes pip-only venvs that fail only at job runtime (hecate job
# 353773). Every app venv installs nemo-gym, whose deps include aiohttp + ray.
mapfile -t GYM_VENVS < <(find "${NEMO_GYM_VENV_DIR:-/opt/gym_venvs}" -mindepth 3 -maxdepth 3 -name .venv -type d 2>/dev/null | sort)
if (( ${#GYM_VENVS[@]} < 2 )); then
    echo "ERROR: expected at least 2 baked gym venvs (vllm_model, swe_agents); found ${#GYM_VENVS[@]}" >&2
    exit 1
fi
for v in "${GYM_VENVS[@]}"; do
    echo "verifying gym venv ${v}"
    "${v}/bin/python" -c "import aiohttp, ray, nemo_gym"
done
EOF

# fla-core GDN backport per installed copy: upstream resolves 0.5.1 and 0.4.2
# (via megatron-bridge), each gets its version-matched patch; no match fails.
RUN <<"EOF" bash -exu -o pipefail
apply_installed_patch() {
    local suffix="$1" label="$2"
    shift 2
    local -a patch_files=("$@")
    local target root patch_file applied
    local -a targets

    mapfile -t targets < <(
        find -L /root/.cache/uv /opt/nemo_rl_venv /opt/ray_venvs /opt/gym_venvs \
            -type f -path "*${suffix}" 2>/dev/null \
            | xargs -r -n1 realpath | sort -u
    )
    if (( ${#targets[@]} == 0 )); then
        echo "ERROR: no installed target found for ${label}" >&2
        exit 1
    fi
    for target in "${targets[@]}"; do
        root="${target%${suffix}}"
        applied=0
        for patch_file in "${patch_files[@]}"; do
            if patch --batch --forward --dry-run -d "$root" -p1 < "$patch_file" > /dev/null; then
                patch --batch --forward -d "$root" -p1 < "$patch_file"
                echo "$(realpath "$target"): applied $(basename "$patch_file")"
                applied=1
                break
            elif patch --batch --reverse --dry-run -d "$root" -p1 < "$patch_file" > /dev/null; then
                echo "$(realpath "$target") already contains $(basename "$patch_file")"
                applied=1
                break
            fi
        done
        if (( ! applied )); then
            echo "ERROR: no ${label} patch matches $target" >&2
            exit 1
        fi
    done
}

apply_installed_patch \
    /fla/ops/gated_delta_rule/wy_fast.py \
    "fla-core Blackwell GDN backport" \
    /tmp/patches/installed/fla-core-0.5.1-blackwell-gdn.patch \
    /tmp/patches/installed/fla-core-0.4.2-blackwell-gdn.patch
EOF


# Patch only the Transformer Engine copy imported by the Megatron policy
# actor. uv also retains source/build cache copies that are not runtime
# installs and need not have the same generated-file context. Materialize the
# actor's symlink as a venv-local file before changing the exact qualified
# allowlist string, leaving the shared uv cache untouched.
RUN <<"PY" /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
import importlib.metadata as metadata
import shutil
import sysconfig
from pathlib import Path

version = metadata.version("transformer-engine")
if not version.startswith("2.15.0+42b8400"):
    raise RuntimeError(f"unexpected Transformer Engine version: {version}")

target = (
    Path(sysconfig.get_paths()["purelib"])
    / "transformer_engine/pytorch/attention/dot_product_attention/utils.py"
)
if not target.is_file():
    raise RuntimeError(f"Transformer Engine attention utilities not found: {target}")

old = "((8, 0), (9, 0), (10, 0), (12, 0))"
new = "((8, 0), (9, 0), (10, 0), (10, 3), (12, 0))"
content = target.read_text()
old_count = content.count(old)
new_count = content.count(new)

if old_count == 1 and new_count == 0:
    source = target.resolve(strict=True)
    temporary = target.with_name(f".{target.name}.sm103.tmp")
    temporary.write_text(content.replace(old, new, 1))
    shutil.copymode(source, temporary)
    temporary.replace(target)
elif old_count == 0 and new_count == 1:
    print(f"{target} already contains the Transformer Engine SM103 FA2 backport")
else:
    raise RuntimeError(
        "could not identify exactly one Transformer Engine FA2 architecture "
        f"allowlist in {target}: old={old_count}, new={new_count}"
    )

patched = target.read_text()
if patched.count(old) != 0 or patched.count(new) != 1:
    raise RuntimeError(f"failed to verify Transformer Engine SM103 FA2 backport: {target}")
with Path("/NEMO_RL_PROVENANCE.txt").open("a") as provenance:
    provenance.write(f"TRANSFORMER_ENGINE_POLICY_VERSION={version}\n")
    provenance.write("TRANSFORMER_ENGINE_SM103_FA2_BACKPORT=1\n")
print(f"Transformer Engine {version}: verified SM103 FA2 in {target}")
PY

## MLPerf logging deps (already in the fork's uv.lock; keeps them present after
## the upstream flip — then run the driver with UV_NO_SYNC=1 or patch pyproject,
## since plain `uv run` prunes packages not in the lockfile).
COPY requirements-mlperf.txt /tmp/requirements-mlperf.txt
RUN uv pip install --no-cache-dir -r /tmp/requirements-mlperf.txt

## Benchmark scripts
WORKDIR /workspace/llm
COPY . .
ENV PYTHONPATH="/workspace/llm:${PYTHONPATH}"
WORKDIR /opt/nemo-rl

# Downstream provenance + fingerprint + INTERNAL tooling (ntrace, JET) are one
# RUN on purpose: overlay2 caps the lower-dir stack (max_depth) and this image
# is already deep. Add new finalization steps INSIDE this block rather than
# appending another RUN. The mlperf dependency install above is deliberately
# left separate -- it sits before `COPY . .` so a source-only change does not
# reinstall it.
RUN <<"EOF" bash -exu -o pipefail
# Downstream patch provenance. Paths are absolute, so WORKDIR (/opt/nemo-rl)
# does not matter here.
find /workspace/llm/patches -type f -name '*.patch' -exec sha256sum {} + | sort > /NEMO_RL_DOWNSTREAM_PATCHES.sha256
{
    echo "OPTIMIZED_GIT_COMMIT_ID=${GIT_COMMIT_ID}"
    echo "DOWNSTREAM_PATCHES_BEGIN"
    cat /NEMO_RL_DOWNSTREAM_PATCHES.sha256
    echo "DOWNSTREAM_PATCHES_END"
} >> /NEMO_RL_PROVENANCE.txt

# Regenerate the container fingerprint (frozen-environment support). Relative
# path -- relies on WORKDIR /opt/nemo-rl above.
python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint

# INTERNAL: ntrace
NTRACE_BUILD_CUPTI_CPP=ON uv pip install --no-cache-dir "git+https://oauth2:${GITLAB_CLONE_ACCESS_TOKEN}@gitlab-master.nvidia.com/compute/e2etrain/ntrace.git@${NTRACE_REVISION}"
printf '%s\n' "${NTRACE_REVISION}" > /NTRACE_REVISION.env
printf '%s\n' "NTRACE_REVISION=${NTRACE_REVISION}" >> /NEMO_RL_PROVENANCE.txt

# INTERNAL: JET
curl -fsSL https://jet.nvidia.com/docs/assets/install.sh | sh
EOF
ENV PATH="/root/.local/bin:${PATH}"

# TODO: mlperf-sysjson.sh requires this to be defined if NVIDIA_PRODUCT_NAME is CUDA
ENV NVIDIA_CUDA_VERSION=${CUDA_VERSION}
