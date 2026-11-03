# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build context: repository root. See ../README.md and the Flint model guide.
ARG FLINT_BASE_IMAGE
FROM ${FLINT_BASE_IMAGE}

COPY nemo_rl /opt/flint/RL/nemo_rl
COPY examples /opt/flint/RL/examples
COPY integrations/flint /opt/flint/RL/integrations/flint
COPY pyproject.toml README.md LICENSE /opt/flint/RL/
COPY .flint-build/flint-sglang /opt/flint/flint-sglang
COPY .flint-build/flashinfer-pr4319 /flint-flashinfer-artifact
ENV PYTHONPATH=/opt/flint/RL:/opt/flint/flint-sglang/src \
    NEMO_RL_PY_EXECUTABLES_SYSTEM=1 \
    SGLANG_EXTERNAL_MODEL_PACKAGE=flint_sglang \
    FLASHINFER_USE_CUDA_NORM=1 \
    NVIDIA_TF32_OVERRIDE=0 \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
    NCCL_CUMEM_ENABLE=0 \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    FLINT_FLASHINFER_BUNDLE=/flint-flashinfer-artifact \
    FLINT_FLASHINFER_MANIFEST_SHA256=9895306c87a69bab22280744969bdb6d0cda787c2477bd81a5d3573c32bc355b
RUN /opt/flint/bin/python-driver /opt/flint/RL/integrations/flint/runtime/tools/install.py \
    && chmod 0755 /opt/flint/RL/integrations/flint/runtime/tools/python-sglang.sh \
    && /opt/flint/bin/python-sglang /opt/flint/RL/integrations/flint/runtime/tools/verify_runtime.py \
       --manifest-sha256 "$FLINT_FLASHINFER_MANIFEST_SHA256" \
       --receipt-dir /opt/flint/verification/pr4319-image
WORKDIR /opt/flint/RL
