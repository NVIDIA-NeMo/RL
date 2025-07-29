ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-devel-ubuntu24.04
FROM ${BASE_IMAGE} AS base

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

# Nsight
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install -y nsight-systems-cli


apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

# Install uv and python
ARG UV_VERSION=0.7.2
ARG PYTHON_VERSION=3.12
ENV PATH="/root/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    if echo "$BASE_IMAGE" | grep -q "cuda-dl"; then \
        uv python install ${PYTHON_VERSION}; \
    fi

# Disable usage stats by default for users who are sensitive to sharing usage.
# Users are encouraged to enable if the wish.
ENV RAY_USAGE_STATS_ENABLED=0
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

# Build vLLM from source if using PyTorch base image
FROM base AS build_vllm
ARG BASE_IMAGE
ARG MAX_JOBS=32
WORKDIR /opt

# Copy the uv.lock file to read the vLLM version
COPY uv.lock /tmp/uv.lock

# Extract vLLM version from uv.lock and build that version
RUN if echo "$BASE_IMAGE" | grep -q "pytorch"; then \
        echo "Building vLLM from source for PyTorch base image" && \
        VLLM_VERSION=$(grep -A 1 'name = "vllm"' /tmp/uv.lock | grep 'version =' | sed 's/version = "\(.*\)"/\1/') && \
        echo "Building vLLM version: $VLLM_VERSION" && \
        git clone https://github.com/vllm-project/vllm.git && \
        cd vllm && \
        git checkout v$VLLM_VERSION && \
        python use_existing_torch.py && \
        pip install -r requirements/build.txt && \
        pip wheel --no-build-isolation -v .; \
    else \
        echo "Skipping vLLM build for non-PyTorch base image"; \
    fi

FROM base AS hermetic

WORKDIR /opt/nemo-rl

# Variables to control the build of TE. If there are issues with parallelization, consider
# setting these to 1.
ARG MAX_JOBS
ARG NVTE_BUILD_THREADS_PER_JOB

ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
ENV UV_LINK_MODE=copy

# Define the no-install-package arguments for PyTorch base images
ARG BASE_IMAGE
ARG UV_NO_INSTALL_PACKAGES
ENV UV_NO_INSTALL_PACKAGES=${UV_NO_INSTALL_PACKAGES}
ENV PATH="/opt/nemo_rl_venv/bin:$PATH"

# This step is to warm the uv cache with flash-attn without invalidating it due to COPY layers
# This layer has to be manually updated
RUN <<"EOF" bash -exu
if echo "$BASE_IMAGE" | grep -q "pytorch"; then
    echo "Creating venv with system-site-packages for PyTorch base image"
    uv venv --system-site-packages ${UV_PROJECT_ENVIRONMENT}
else
    echo "Creating venv without system-site-packages"
    uv venv ${UV_PROJECT_ENVIRONMENT}
    VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip install --link-mode symlink setuptools torch==2.7.0 psutil ninja --torch-backend=cu128
    VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip install --link-mode symlink flash-attn==2.7.4.post1 --no-build-isolation
fi
EOF

# First copy only the dependency files
COPY pyproject.toml uv.lock ./
COPY --link 3rdparty/ ./3rdparty/

# Install vLLM wheel if it was built in the build_vllm stage
ARG BASE_IMAGE
RUN --mount=type=bind,from=build_vllm,source=/opt/,target=/tmp/build_vllm/ \
    if echo "$BASE_IMAGE" | grep -q "pytorch"; then \
        echo "Installing vLLM wheel from build stage" && \
        pip install --no-cache-dir /tmp/build_vllm/vllm/vllm*.whl; \
    fi

RUN <<"EOF" bash -exu
# Load the no-install-package arguments if they exist
if [ -f /tmp/uv_args.env ]; then
    source /tmp/uv_args.env
fi

# uv sync has a more reliable resolver than simple uv pip install which can fail

# Sync each training + inference backend one at a time (since they may conflict)
# to warm the uv cache, then at the end just sync the default dependencies.
# Do everything in one layer to prevent large layers.

# The venv is symlinked to avoid bloating the layer size
uv sync --link-mode symlink --locked --no-install-project $UV_NO_INSTALL_PACKAGES
uv sync --link-mode symlink --locked --inexact --extra vllm --no-install-project $UV_NO_INSTALL_PACKAGES
uv sync --link-mode symlink --locked --inexact --extra mcore --no-install-project $UV_NO_INSTALL_PACKAGES
uv sync --link-mode symlink --locked --inexact --all-groups --no-install-project $UV_NO_INSTALL_PACKAGES
EOF

ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

WORKDIR /opt/nemo-rl

FROM hermetic AS release

ARG NEMO_RL_COMMIT
ARG NVIDIA_BUILD_ID
ARG NVIDIA_BUILD_REF
ARG UV_NO_INSTALL_PACKAGES
ENV UV_NO_INSTALL_PACKAGES=${UV_NO_INSTALL_PACKAGES}
ENV UV_NO_SYNC=1
ENV NEMO_RL_COMMIT=${NEMO_RL_COMMIT:-<unknown>}
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-<unknown>}
ENV NVIDIA_BUILD_REF=${NVIDIA_BUILD_REF:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

# Copy in source and prefetch all virtual environments
COPY . /opt/nemo-rl
RUN <<"EOF" bash -exu
echo "Installing nemo-rl"
UV_NO_SYNC=0 UV_LINK_MODE=symlink uv sync --locked --inexact $UV_NO_INSTALL_PACKAGES
EOF
