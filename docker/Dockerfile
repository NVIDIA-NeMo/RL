# syntax=docker/dockerfile:1
# Usage:
#   Self-contained build (default: builds from main):
#     docker buildx build -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .
#
#   Self-contained build (specific git ref):
#     docker buildx build -f docker/Dockerfile --build-arg NRL_GIT_REF=r0.6.0 --tag <registry>/nemo-rl:r0.6.0 --push .
#
#   Self-contained build (remote NeMo RL source; no need for a local clone of NeMo RL):
#     docker buildx build -f docker/Dockerfile --build-arg NRL_GIT_REF=r0.6.0 --tag <registry>/nemo-rl:r0.6.0 --push https://github.com/NVIDIA-NeMo/RL.git
#
#   Local NeMo RL source override:
#     docker buildx build --build-context nemo-rl=. -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .
#
#   Custom setup:
#     docker buildx build --build-context custom-setup=my-setup-dir/ --build-arg CUSTOM_SETUP_FNAME=my_script.sh -f docker/Dockerfile ...
#
#   To skip custom setup:
#     docker buildx build --build-arg CUSTOM_SETUP_FNAME= -f docker/Dockerfile ...
#
# Optional build args to skip vLLM or SGLang dependencies:
#   --build-arg SKIP_VLLM_BUILD=1    # Skip vLLM dependencies
#   --build-arg SKIP_SGLANG_BUILD=1  # Skip SGLang dependencies
#   --build-arg SKIP_TRTLLM_BUILD=1  # Skip TRT-LLM dependencies

ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.03-cuda13.2-devel-ubuntu24.04
FROM scratch AS nemo-rl
ARG NRL_GIT_REF=main
ADD --keep-git-dir=true https://github.com/NVIDIA-NeMo/RL.git#${NRL_GIT_REF} /

# Empty default; CI overrides with --build-context to inject pre-compiled wheels
FROM scratch AS uv-cache-seed

# Default custom-setup stage: installs apptainer.
# Override with: --build-context custom-setup=<your-dir>
FROM scratch AS custom-setup
COPY docker/install_apptainer.sh /

FROM ${BASE_IMAGE} AS base
# An environment variable to indicate that we are in a container.
ENV NRL_CONTAINER=1

# It is more convenient for users to run as root
USER root

RUN <<"EOF" bash -exu -o pipefail
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

apt-get update
apt-get install -y --no-install-recommends \
    jq \
    curl \
    git \
    rsync \
    wget \
    less \
    vim \
    ccache \

# Nsight
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/$(uname -m | sed 's/aarch64/sbsa/')/7fa2af80.pub
apt update
apt install -y nsight-systems-cli

# To fix CVE-2025-68973
apt install -y --only-upgrade gnupg

# Upgrade rsync to address security vulnerabilities
apt-get install -y --only-upgrade rsync

# Remove ffmpeg (from base image) to reduce attack surface
apt-get remove -y --purge ffmpeg || true
apt-get autoremove -y

apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

# CMake (for CUDA extension builds: transformer-engine, mamba-ssm, etc.)
RUN GITHUB_ARTIFACTORY=github.com \
    && CMAKE_VERSION=4.0.3 \
    && ARCH=$(uname -m) \
    && CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}" \
    && curl --retry 3 --retry-delay 2 -fsSL -o "${CMAKE_INSTALLER}.tar.gz" \
        "https://${GITHUB_ARTIFACTORY}/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}.tar.gz" \
    && tar -xzf "${CMAKE_INSTALLER}.tar.gz" \
    && cp -r "${CMAKE_INSTALLER}/bin/"* /usr/local/bin/ \
    && cp -r "${CMAKE_INSTALLER}/share/"* /usr/local/share/ \
    && rm -rf "${CMAKE_INSTALLER}" "${CMAKE_INSTALLER}.tar.gz"

# Install uv and python
ARG UV_VERSION=0.11.18
ARG PYTHON_VERSION=3.13.13
ENV PATH="/root/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    uv python install ${PYTHON_VERSION}

# Disable usage stats by default for users who are sensitive to sharing usage.
# Users are encouraged to enable if the wish.
ENV RAY_USAGE_STATS_ENABLED=0
# After ray>=2.47, this feature is enabled by default which creates uv venvs for any py_executable starting with `uv run`.
# There is severe contention and performance issues with this enabled considering our dependencies are so large and occasionally
# need to be compiled, so NeMo RL has an implementation in nemo_rl/utils/venv.py that does it once per node as opposed to once per task.
ENV RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs
ENV NEMO_GYM_VENV_DIR=/opt/gym_venvs


FROM base AS hermetic

WORKDIR /opt/nemo-rl

# Variables to control the build of TE. If there are issues with parallelization, consider
# setting these to 1.
ARG MAX_JOBS
ARG NVTE_BUILD_THREADS_PER_JOB
# Only use for custom vllm installs. Learn more at https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/use-custom-vllm.md
ARG BUILD_CUSTOM_VLLM
ARG BUILD_CUSTOM_VLLM_URL
ARG BUILD_CUSTOM_VLLM_REF
ARG BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION
# Only use for custom flashinfer installs.
ARG BUILD_CUSTOM_FLASHINFER
ARG BUILD_CUSTOM_FLASHINFER_URL
ARG BUILD_CUSTOM_FLASHINFER_REF
# TRT-LLM build args. Unlike vLLM/SGLang, TRT-LLM is always built from source
# because there are no cp313/cu13/aarch64 wheels on PyPI for v1.3.0rc15. Override
# the URL/REF if you want a different fork or tag. Modelopt 0.37 is built
# alongside because TRT-LLM v1.3.0rc15's requirements.txt pins
# `nvidia-modelopt[torch]~=0.37.0` and PyPI lacks a cp313 wheel for that line.
ARG BUILD_CUSTOM_TRTLLM_URL
ARG BUILD_CUSTOM_TRTLLM_REF
ARG BUILD_CUSTOM_TRTLLM_MODELOPT_URL
ARG BUILD_CUSTOM_TRTLLM_MODELOPT_REF
# Skip building vLLM, SGLang, or TRT-LLM dependencies (set to any non-empty value to skip)
ARG SKIP_VLLM_BUILD
ARG SKIP_SGLANG_BUILD
ARG SKIP_TRTLLM_BUILD
ARG BASE_IMAGE
ARG UV_VERSION
ARG BASE_IMAGE
ARG UV_VERSION

ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
ENV UV_LINK_MODE=copy

# Ensure DeepEP is built for H100 and B200 (also mcore inference unified memory API now invokes a torch API that requires these to be set)
ENV TORCH_CUDA_ARCH_LIST="9.0 10.0"

# Ensure HybridEP/DeepEP JIT compilation can find nvcc at runtime
ENV CUDA_HOME=/usr/local/cuda

## CUDA 13 moved standard library headers under include/cccl; expose that path so
## builds that include <cuda/std/...> (e.g., deep_gemm) can find them.
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl

# First copy only the dependency files
COPY --from=nemo-rl pyproject.toml uv.lock ./
# Copy in the top level __init__.py/package_info.py since build-custom-vllm.sh needs the nemo_rl package to exist.
COPY --from=nemo-rl nemo_rl/__init__.py nemo_rl/package_info.py ./nemo_rl/
COPY --from=nemo-rl tools/build-custom-vllm.sh ./tools/build-custom-vllm.sh
COPY --from=nemo-rl tools/build-custom-flashinfer.sh ./tools/build-custom-flashinfer.sh
COPY --from=nemo-rl tools/build-custom-trtllm.sh ./tools/build-custom-trtllm.sh
COPY --from=nemo-rl --link research/ ./research/
COPY --from=nemo-rl --link 3rdparty/ ./3rdparty/

RUN --mount=type=ssh \
    --mount=type=bind,from=uv-cache-seed,source=.,target=/tmp/uv-cache-seed <<"EOF" bash -exu
CACHE_KEY=$(echo "${BASE_IMAGE}-${UV_VERSION}" | md5sum | cut -c1-12)
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

# Sync each training + inference backend one at a time (since they may conflict)
# to warm the uv cache, then at the end just sync the default dependencies.
# Do everything in one layer to prevent large layers.

# The venv is symlinked to avoid bloating the layer size
UV_LINK_MODE=hardlink uv sync --locked --no-install-project
if [[ -z "${SKIP_VLLM_BUILD:-}" ]]; then
    UV_LINK_MODE=hardlink uv sync --locked --extra vllm --no-install-project
fi
if [[ -z "${SKIP_SGLANG_BUILD:-}" ]]; then
    UV_LINK_MODE=hardlink uv sync --locked --extra sglang --no-install-project
fi
uv sync --link-mode symlink --locked --extra mcore --no-install-project
uv sync --link-mode symlink --locked --extra automodel --no-install-project
uv sync --link-mode symlink --locked --extra modelopt --no-install-project
uv sync --link-mode symlink --locked --all-groups --no-install-project
if [[ -z "${SKIP_TRTLLM_BUILD:-}" ]]; then
    # The custom PEP 517 backend in 3rdparty/trtllm-workspace/_backend.py
    # invokes tools/build-custom-trtllm.sh during this sync (≈60 min).
    # Build coordinates are passed via env vars:
    #   BUILD_CUSTOM_TRTLLM_URL / BUILD_CUSTOM_TRTLLM_REF
    uv sync --link-mode symlink --locked --extra trtllm --no-install-project
fi

# Remove the aiohttp in this uv cache dir to fully address CVE GHSA-mqqc-3gqh-h2x8
# The ray install will include the older aiohttp version in its cache
find /root/.cache/uv -type d -path "*ray/_private/runtime_env/agent/thirdparty_files/aiohttp*" -exec rm -rf {} +

echo "${CACHE_KEY}" > /root/.cache/uv/.cache-key
EOF

ENV PATH="/opt/nemo_rl_venv/bin:$PATH"
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs
# Point TE at the pip-installed cuDNN (nvidia-cudnn-cu12) instead of the system one.
# The container's system cuDNN (/lib/x86_64-linux-gnu/libcudnn*.so.9) may be a different
# version than what pip installed. TE prioritizes system libraries by default, causing a
# version mismatch crash (e.g. system 9.10.1 vs pip 9.19.0). CUDNN_HOME makes TE's Python
# code find the pip version first, and LD_LIBRARY_PATH makes the dynamic linker resolve
# cuDNN sub-libraries from pip when loading libtransformer_engine.so.
ENV CUDNN_HOME=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
ENV LD_LIBRARY_PATH="/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
# Verify with: python -c "import transformer_engine.pytorch as te; print(te.get_cudnn_version())"

# Custom setup layer (override with: --build-context custom-setup=<dir> --build-arg CUSTOM_SETUP_FNAME=<script>)
# To skip: --build-arg CUSTOM_SETUP_FNAME=
ARG CUSTOM_SETUP_FNAME=install_apptainer.sh
RUN --mount=from=custom-setup,src=/,dst=/tmp/custom-setup <<"EOF" bash -exu
if [[ -z "${CUSTOM_SETUP_FNAME}" ]]; then
    echo "CUSTOM_SETUP_FNAME is empty, skipping custom setup"
    exit 0
fi
cp /tmp/custom-setup/${CUSTOM_SETUP_FNAME} /opt/${CUSTOM_SETUP_FNAME}
bash -x /opt/${CUSTOM_SETUP_FNAME}
EOF

WORKDIR /opt/nemo-rl

FROM hermetic AS release

# Re-declare build args for this stage
ARG SKIP_VLLM_BUILD
ARG SKIP_SGLANG_BUILD
# Space-separated config paths whose NeMo Gym venvs are prefetched into the image
# (empty default = skip). Consumed by the gym prefetch RUN below.
ARG NEMO_GYM_PREFETCH_CONFIGS=
ARG NEMO_RL_COMMIT
ARG NVIDIA_BUILD_ID
ARG NVIDIA_BUILD_REF
ARG RC_DATE=00.00
ARG TARGETARCH
ENV NEMO_RL_COMMIT=${NEMO_RL_COMMIT:-<unknown>}
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-<unknown>}
ENV NVIDIA_BUILD_REF=${NVIDIA_BUILD_REF:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

# AWS EFA OFI plugin discovery (p4d / p5 / p5en multi-node NCCL via SRD).
# The host's aws-efa-installer places libnccl-net-ofi.so under
# /opt/amazon/ofi-nccl/lib and libfabric under /opt/amazon/efa/lib. Without
# these on the linker search path, NCCL silently falls back to its socket
# transport. Register both via ld.so.conf.d (so `ldconfig` populates the cache
# once host paths are mounted at runtime) and LD_LIBRARY_PATH (so NCCL's
# dlopen resolves the plugin even before `ldconfig` is re-run).
RUN printf '%s\n' /opt/amazon/ofi-nccl/lib /opt/amazon/efa/lib \
        > /etc/ld.so.conf.d/aws-efa.conf
ENV LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"

# Copy in source from build context (defaults to cloned repo, can be overridden)
# Exclude pyproject.toml and uv.lock since those may be altered by build-custom-vllm.sh
COPY --from=nemo-rl --exclude=pyproject.toml --exclude=uv.lock . /opt/nemo-rl
# Unshallow the repo to get the full history (in the case it was from the scratch layer).
# Potentially not necessary if the repo is passed in as a complete repository (w/ full git history),
# so do a quick check before trying to unshallow.
RUN git rev-parse --is-shallow-repository | grep -q true && git fetch --unshallow || true
RUN <<"EOF" bash -exu
NEGATIVE_FILTERS=""
if [[ -n "${SKIP_VLLM_BUILD:-}" ]]; then
    NEGATIVE_FILTERS="$NEGATIVE_FILTERS vllm"
fi
if [[ -n "${SKIP_SGLANG_BUILD:-}" ]]; then
    NEGATIVE_FILTERS="$NEGATIVE_FILTERS sglang"
fi
if [[ -n "$NEGATIVE_FILTERS" ]]; then
    UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py --negative-filters $NEGATIVE_FILTERS
else
    UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py
fi
EOF

# Prefetch NeMo Gym internal venvs (gym servers like code_gen, math_with_judge, etc.)
# into the image. Gated on NEMO_GYM_PREFETCH_CONFIGS (no-op by default). Runs after the
# source COPY above so examples/nemo_gym/prefetch_venvs.py is present.
# Gym pins vllm==0.20.0; override it to vLLM's prebuilt cu-wheel (PyPI default is cu12) so Gym's
# venvs share RL's torch/nvidia stack. 0.20.0's default GitHub wheel is the cu130 build (the
# same wheel the policy pins in pyproject.toml); UV_TORCH_BACKEND keeps torch on the same CUDA.
ARG NEMO_GYM_CUDA=cu130
ARG NEMO_GYM_VLLM_VERSION=0.20.0
RUN <<"EOF" bash -exu
if [[ -n "${NEMO_GYM_PREFETCH_CONFIGS:-}" ]]; then
    case "${TARGETARCH:-amd64}" in arm64) WHEEL_ARCH=aarch64 ;; *) WHEEL_ARCH=x86_64 ;; esac
    GYM_VLLM_OVERRIDE=/opt/nemo-rl/.gym-vllm-override.txt
    printf 'vllm @ https://github.com/vllm-project/vllm/releases/download/v%s/vllm-%s-cp38-abi3-manylinux_2_35_%s.whl\n' \
        "${NEMO_GYM_VLLM_VERSION}" "${NEMO_GYM_VLLM_VERSION}" "${WHEEL_ARCH}" > "${GYM_VLLM_OVERRIDE}"
    UV_TORCH_BACKEND="${NEMO_GYM_CUDA}" \
    UV_OVERRIDE="${GYM_VLLM_OVERRIDE}" \
    UV_LINK_MODE=symlink uv run python examples/nemo_gym/prefetch_venvs.py $NEMO_GYM_PREFETCH_CONFIGS
    rm -f "${GYM_VLLM_OVERRIDE}"
fi
EOF
# Generate container fingerprint for frozen environment support
# Store outside /opt/nemo-rl to avoid being overwritten by user mounts
RUN python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint

# NOTICES.txt file points to where the OSS source code is archived
RUN echo "This distribution includes open source which is archived at the following URL: https://opensource.nvidia.com/oss/teams/nvidia/nemo-rl/${RC_DATE}:linux-${TARGETARCH}/index.html" > NOTICES.txt && \
    echo "For further inquiries or assistance, contact us at oss-requests@nvidia.com" >> NOTICES.txt
