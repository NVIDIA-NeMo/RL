# syntax=docker/dockerfile:1
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Qwen3.5 397B GRPO (nemo-rl) benchmark container. Stages base/hermetic/release
# are vendored from nemo-rl docker/Dockerfile (its own uv-managed stack on
# cuda-dl-base — not the dgx pytorch base), plus the MLPerf overlay from the
# fork's docker/mlperf/Dockerfile.gym_qwen35.
#
# Fork now, upstream later: NEMO_RL_REPO/NEMO_RL_REVISION default to the
# integration fork branch; after upstreaming, flip both to NVIDIA-NeMo/RL@main
# and carry the remainder in patches/nemo-rl/ (see its README).
#
# The base ARG is NOT named FROM_IMAGE_NAME: ci/build.yml force-passes
# FROM_IMAGE_NAME=<dgx pytorch image>, which would replace cuda-dl-base.

ARG NEMO_RL_BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.03-cuda13.2-devel-ubuntu24.04

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

ARG UV_VERSION=0.11.18
ARG PYTHON_VERSION=3.13.13
ENV PATH="/root/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    uv python install ${PYTHON_VERSION}

ENV RAY_USAGE_STATS_ENABLED=0
ENV RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs
ENV NEMO_GYM_VENV_DIR=/opt/gym_venvs

## Stage: nemo-rl-src — clone nemo-rl at the pinned repo/revision
FROM base AS nemo-rl-src

## 1. Clone nemo-rl (+ 3rdparty submodules) and record provenance.
# Submodule ARGs: SKIP = keep the pin from the nemo-rl revision.
ARG NEMO_RL_REPO=https://github.com/shuyixiong/RL.git
ARG NEMO_RL_REVISION=trtllm-agentic-swe-mlperf
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
    echo $(git -C 3rdparty/Automodel-workspace/Automodel rev-parse HEAD) > /AUTOMODEL_COMMIT_HASH.env

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

## 3. Carried nemo-rl patches (see patches/nemo-rl/README.md)
COPY patches/nemo-rl/ /tmp/patches/nemo-rl/
RUN <<"EOF" bash -exu
cd /opt/nemo-rl-src
shopt -s nullglob
for p in /tmp/patches/nemo-rl/*.patch; do
    git apply --3way "$p"
done
EOF

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
# we must build from source. These ARGs control which commit is compiled:
#   BUILD_CUSTOM_TRTLLM_URL  git remote (default: github.com/NVIDIA/TensorRT-LLM)
#   BUILD_CUSTOM_TRTLLM_REF  commit hash / branch / tag to checkout
# The build is handled by 3rdparty/TensorRT-LLM-workspace/_backend.py via the
# PEP 517 hook invoked during `uv sync --extra trtllm`. Defaults below are used
# unless overridden with `--build-arg`; promoting the ARGs to ENV puts them in
# the `uv sync` process environment so _backend.py reads them (an ARG alone is
# only a build-time substitution var, not an env var).
ARG BUILD_CUSTOM_TRTLLM_URL=https://github.com/hchings/TensorRT-LLM.git
ARG BUILD_CUSTOM_TRTLLM_REF=983e7ff57dd26ce662e1b34cce94b76c9181be05
ENV BUILD_CUSTOM_TRTLLM_URL=${BUILD_CUSTOM_TRTLLM_URL}
ENV BUILD_CUSTOM_TRTLLM_REF=${BUILD_CUSTOM_TRTLLM_REF}
ARG SKIP_VLLM_BUILD
# SGLang is not used by this benchmark; skip it by default to cut build time.
ARG SKIP_SGLANG_BUILD=1
ARG SKIP_TRTLLM_BUILD
ARG NEMO_RL_BASE_IMAGE
ARG UV_VERSION

ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
ENV UV_LINK_MODE=copy
ENV TORCH_CUDA_ARCH_LIST="9.0 10.0 10.3"
ENV CUDA_HOME=/usr/local/cuda
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl

# Point TE at the pip-installed cuDNN, not the system one (see nemo-rl docker/Dockerfile)
ENV CUDNN_HOME=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
ENV CUDNN_PATH=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
ENV LD_LIBRARY_PATH="/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

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
    # invokes tools/build-custom-trtllm.sh during this sync (≈60 min). Build
    # coordinates come from the BUILD_CUSTOM_TRTLLM_URL / BUILD_CUSTOM_TRTLLM_REF
    # ENV set above (from the ARGs / --build-arg), so no inline override here.
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

ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

# AWS EFA OFI plugin discovery (see nemo-rl docker/Dockerfile)
RUN printf '%s\n' /opt/amazon/ofi-nccl/lib /opt/amazon/efa/lib \
        > /etc/ld.so.conf.d/aws-efa.conf
ENV LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH}"

# pyproject/uv.lock excluded: the custom build scripts may alter the hermetic copies
COPY --from=nemo-rl-src --exclude=pyproject.toml --exclude=uv.lock /opt/nemo-rl-src/ /opt/nemo-rl

# Provenance files (surfaced to JET by run.sub)
COPY --from=nemo-rl-src /NEMO_RL_COMMIT_HASH.env /NEMO_GYM_COMMIT_HASH.env /MBRIDGE_COMMIT_HASH.env /AUTOMODEL_COMMIT_HASH.env /

# Prefetch the per-worker ray venvs into the image
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

## Stage: mlperf (final) — gym overlay (vendored from Dockerfile.gym_qwen35) + benchmark bits
FROM release AS mlperf
ARG TARGETARCH
ARG GITLAB_CLONE_ACCESS_TOKEN
ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

USER root
WORKDIR /opt/nemo-rl

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
RUN <<"EOF" bash -exu -o pipefail
SWE_AGENTS_DIR=/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents

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

SETUP_DIR="${SWE_AGENTS_DIR}/swe_rebench_setup"
mkdir -p "${SETUP_DIR}"
SETUP_DIR="${SETUP_DIR}" \
    REBENCH_DIR="${SETUP_DIR}/SWE-rebench-V2" \
        "${SWE_AGENTS_DIR}/setup_scripts/swe_rebench.sh"
EOF

# Pre-bake the Gym server venvs (inert stub paths stand in for runtime inputs)
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
EOF

## Bake the runtime fixes (the fork launcher bind-mounted these at runtime over
## image-specific paths). Retirement: fla_* once the fla pin includes fla
## PR #1000 (set APPLY_FLA_RUNTIME_FIXES=0); te_dpa_utils.py once TE includes
## the sm10.3 hdim-256 gate (APPLY_TE_RUNTIME_FIXES=0); Gym files once the
## Gym pin absorbs them.
COPY runtime_fixes/ /tmp/runtime_fixes/
ARG APPLY_FLA_RUNTIME_FIXES=1
ARG APPLY_TE_RUNTIME_FIXES=1
# Versions the fla/TE fixes were written against; a pin bump fails the build
# loudly instead of silently clobbering newer files.
ARG FLA_RUNTIME_FIX_VERSION=0.4.2
ARG TE_RUNTIME_FIX_VERSION=2.15
RUN <<"EOF" bash -exu -o pipefail
# Fail unless every installed <dist> version matches expected (prefix match)
require_dist_version() {
    local dist="$1" expected="$2" versions v
    versions=$( { find -L /root/.cache/uv /opt/nemo_rl_venv /opt/ray_venvs /opt/gym_venvs -type d -name "${dist}-*.dist-info" 2>/dev/null || true; } | sed -E "s|.*/${dist}-([^/]+)\.dist-info|\1|" | sort -u )
    if [[ -z "$versions" ]]; then
        echo "ERROR: no ${dist} dist-info found; cannot verify runtime-fix compatibility" >&2
        exit 1
    fi
    for v in $versions; do
        if [[ "$v" != "$expected" && "$v" != "$expected".* ]]; then
            echo "ERROR: ${dist} $v != expected ${expected}* — re-evaluate the runtime fix or set its APPLY_* arg to 0" >&2
            exit 1
        fi
    done
}

# Overwrite every real file matching a pattern; fail unless >=1 target found.
overlay() {
    local src="$1"; shift
    local pattern="$1"; shift
    local targets
    # -L: venv entries are symlinks into the uv cache; realpath dedupes them.
    # Exclude TensorRT-LLM's vendored fla copy (tensorrt_llm/_torch/modules/fla):
    # the broad "*/fla/utils.py" glob otherwise clobbers it with the standalone
    # fla-core file, whose input_guard() signature (no_guard_contiguous) does not
    # match trtllm's own fused_recurrent.py calls (exclude_args) -> import crash.
    targets=$( { find -L /root/.cache/uv /opt/nemo_rl_venv /opt/ray_venvs /opt/gym_venvs -type f -path "$pattern" -not -path "*/tensorrt_llm/*" 2>/dev/null || true; } | xargs -r -n1 realpath | sort -u )
    if [[ -z "$targets" ]]; then
        echo "ERROR: no target found for runtime fix $src (pattern $pattern)" >&2
        exit 1
    fi
    local t
    for t in $targets; do
        echo "overlaying $t with $src"
        cp "$src" "$t"
    done
}

if [[ "${APPLY_TE_RUNTIME_FIXES}" == "1" ]]; then
    require_dist_version transformer_engine "${TE_RUNTIME_FIX_VERSION}"
    overlay /tmp/runtime_fixes/te_dpa_utils.py \
        "*/transformer_engine/pytorch/attention/dot_product_attention/utils.py"
fi

# No site-packages/ prefix: uv cache archive paths lack that component
if [[ "${APPLY_FLA_RUNTIME_FIXES}" == "1" ]]; then
    require_dist_version fla_core "${FLA_RUNTIME_FIX_VERSION}"
    overlay /tmp/runtime_fixes/fla_wy_fast.py "*/fla/ops/gated_delta_rule/wy_fast.py"
    overlay /tmp/runtime_fixes/fla_utils.py "*/fla/utils.py"
fi

# Gym server + R2E-Gym eval-harness repairs (fixed paths inside the Gym tree)
GYM_DIR=/opt/nemo-rl/3rdparty/Gym-workspace/Gym
R2E_GYM_SRC="${GYM_DIR}/responses_api_agents/swe_agents/swe_r2e_gym_setup/R2E-Gym/src/r2egym"
test -f "${GYM_DIR}/nemo_gym/server_utils.py"
cp /tmp/runtime_fixes/server_utils.py "${GYM_DIR}/nemo_gym/server_utils.py"
test -d "${R2E_GYM_SRC}/agenthub/utils"
cp /tmp/runtime_fixes/log.py "${R2E_GYM_SRC}/agenthub/utils/log.py"
cp /tmp/runtime_fixes/utils.py "${R2E_GYM_SRC}/agenthub/utils/utils.py"
# Sanitized LocalRuntime: the harness python env leaked into testbed
# subprocesses, failing every py3.8 evaluation (all-zero rewards)
test -f "${R2E_GYM_SRC}/agenthub/runtime/local.py"
cp /tmp/runtime_fixes/r2e_local_runtime.py "${R2E_GYM_SRC}/agenthub/runtime/local.py"
EOF

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

# Regenerate the container fingerprint (frozen-environment support)
RUN python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint

# INTERNAL
# ntrace
RUN NTRACE_BUILD_CUPTI_CPP=ON uv pip install --no-cache-dir "git+https://${GITLAB_CLONE_ACCESS_TOKEN}@gitlab-master.nvidia.com/compute/e2etrain/ntrace.git@main"
# JET
RUN curl -fsSL https://jet.nvidia.com/docs/assets/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
