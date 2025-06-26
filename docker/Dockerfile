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
    uv python install ${PYTHON_VERSION}

# Disable usage stats by default for users who are sensitive to sharing usage.
# Users are encouraged to enable if the wish.
ENV RAY_USAGE_STATS_ENABLED=0
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs


FROM base AS hermetic

WORKDIR /opt/nemo-rl

# First copy only the dependency files
COPY pyproject.toml uv.lock ./
COPY --link 3rdparty/ ./3rdparty/

# Variables to control the build of TE. If there are issues with parallelization, consider
# setting these to 1.
ARG MAX_JOBS=4
ARG NVTE_BUILD_THREADS_PER_JOB=1

ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
ENV UV_LINK_MODE=copy

# Create and activate virtual environment
RUN <<"EOF" bash -exu
uv venv ${UV_PROJECT_ENVIRONMENT}
# uv sync has a more reliable resolver than simple uv pip install which can fail

# Sync each training + inference backend one at a time (since they may conflict)
# to warm the uv cache, then at the end just sync the default dependencies.
# Do everything in one layer to prevent large layers.

# The venv is symlinked to avoid bloating the layer size
uv sync --link-mode symlink --locked --no-install-project
uv sync --link-mode symlink --locked --extra vllm --no-install-project
uv sync --link-mode symlink --locked --extra mcore --no-install-project
uv sync --link-mode symlink --locked --all-groups --no-install-project
EOF

ENV PATH="/opt/nemo_rl_venv/bin:$PATH"
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

WORKDIR /opt/nemo-rl

FROM hermetic AS release

ARG NEMO_RL_COMMIT
ARG NVIDIA_BUILD_ID
ARG NVIDIA_BUILD_REF
ENV NEMO_RL_COMMIT=${NEMO_RL_COMMIT:-<unknown>}
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-<unknown>}
ENV NVIDIA_BUILD_REF=${NVIDIA_BUILD_REF:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

# Copy in source and prefetch all virtual environments
COPY . /opt/nemo-rl
RUN UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py
