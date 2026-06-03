# syntax=docker/dockerfile:1.6
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Overlay image: base dynamo vllm worker + ModelExpress v2 Python client + NIXL.
#
# The base image (BASE_IMAGE arg) is the same arm64 + Blackwell build we use
# for the non-MX dynamo path
# (e.g. jwillthomson/dynamo-arm-rl-tokenize-endpoint-91cdc71:latest). This
# overlay adds:
#
#   * The `modelexpress` Python client from `kavink/nemo_rl_moe` on
#     ai-dynamo/modelexpress (head 1d8049d5 as of 2026-05-11) — contains
#     MxV2RefitReceiver, the shape registry codecs, and the picker.
#   * NIXL userspace + UCX bindings, required for RDMA transfers.
#
# The dynamo-side refit Python code (dynamo.vllm.mx_refit.*) is NOT baked
# in here — it lives in the dynamo repo and is either part of the base image
# (if the base was built after Phase 2 lands on jthomson04/tokenize-endpoint-merge-main-05-07)
# or picked up via the PYTHONPATH dev override on the DGD manifest. See
# infra/nrl_k8s/dynamo_mx/README.md for the no-rebuild dev loop.
#
# Build:
#   docker buildx build --platform linux/arm64 \
#     --build-arg BASE_IMAGE=jwillthomson/dynamo-arm-rl-tokenize-endpoint-91cdc71:latest \
#     --build-arg MX_REF=1d8049d5 \
#     -t jwillthomson/dynamo-arm-rl-tokenize-endpoint-91cdc71-mx:1d8049d5 \
#     -f infra/nrl_k8s/dynamo_mx/Dockerfile .
#
# Push:
#   docker push jwillthomson/dynamo-arm-rl-tokenize-endpoint-91cdc71-mx:1d8049d5
#
# Then update infra/nrl_k8s/examples_dgd/qwen3_4b_thinking_gb300_mx.yaml's
# image tag to match.

ARG BASE_IMAGE=jwillthomson/dynamo-arm-rl-tokenize-endpoint-91cdc71:latest
FROM ${BASE_IMAGE}

# `kavink/nemo_rl_moe` is the v2-refit branch. Pin to a SHA so rebuilds are
# reproducible.
ARG MX_REF=1d8049d5

# NIXL: GitHub release of nixl + ucx. The dynamo image typically already
# ships UCX; this just ensures the Python bindings match.
# TODO: confirm whether the base image already has nixl-python. If yes,
# remove the pip install below.

USER root

# System deps for NIXL: librdmacm, libibverbs already in the base via mlnx_ofed;
# we just need to pip-install the bindings + the modelexpress client.
RUN python3 -m pip install --no-cache-dir \
      "git+https://github.com/ai-dynamo/modelexpress.git@${MX_REF}#subdirectory=modelexpress_client/python" \
 && python3 -c "from modelexpress import MxV2RefitReceiver, MxV2TrainingPublisher; print('modelexpress imports OK')"

# Smoke: import the dynamo refit extension + check it can be resolved as a
# worker_extension_cls.
RUN python3 -c "\
from dynamo.vllm.mx_refit.extension import MxRefitWorkerExtension; \
assert hasattr(MxRefitWorkerExtension, 'update_weights_via_mx'); \
assert hasattr(MxRefitWorkerExtension, 'prepare_refit_info'); \
print('MxRefitWorkerExtension import OK')"

# Default to the same entrypoint as the base — DGD manifests override this.
