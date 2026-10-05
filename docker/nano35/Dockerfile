# syntax=docker/dockerfile:1
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Public source build; prepare once and reuse the image across training jobs.
FROM nvcr.io/nvidia/cuda-dl-base@sha256:8c9b9b6d878238915e3c379bb75ef56087b45f5c2c910fcf5331749e0008c2c4
USER root
WORKDIR /opt/nemo-rl
# Copy and clean the read-only build context in the same layer. A separate COPY
# layer would keep deleted builder metadata recoverable in the exported image.
RUN --mount=type=bind,source=.,target=/tmp/nano35-build-source \
    cp -a --no-preserve=ownership /tmp/nano35-build-source/. /opt/nemo-rl/ \
    && bash docker/nano35/build_core.sh \
    && bash docker/nano35/build_swe.sh \
    && bash docker/nano35/finalize_runtime.sh
ENV NRL_CONTAINER=1 \
    RAY_USAGE_STATS_ENABLED=0 \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv \
    NEMO_RL_VENV_DIR=/opt/ray_venvs \
    NEMO_GYM_VENV_DIR=/opt/gym_venvs \
    TRTLLM_WHEEL_CACHE_DIR=/opt/trtllm_wheels \
    TRTLLM_REQUIRE_CACHED_WHEEL=1 \
    NRL_FORCE_REBUILD_VENVS=false \
    PATH=/opt/nemo_rl_venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin
ENTRYPOINT ["bash", "/opt/nemo-rl/docker/nano35/entrypoint.sh"]
CMD ["bash"]
