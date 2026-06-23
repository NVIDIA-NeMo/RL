# syntax=docker/dockerfile:1.6
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Overlay image: base Dynamo vLLM worker + ModelExpress v2 Python client + NIXL.
#
# The base image (BASE_IMAGE arg) is the same arm64 + Blackwell build we use
# for the non-MX dynamo path
# (e.g. jwillthomson/dynamo-arm-rl-tokenize-endpoint-91cdc71:latest). This
# overlay adds:
#
#   * The `modelexpress` Python client from `kavink/nemo_rl_moe` on
#     ai-dynamo/modelexpress (head e0c9447939f0420f6815464b09455724e57d7553
#     as of 2026-06-23) — contains
#     MxV2RefitReceiver, the shape registry codecs, and the picker.
#   * NIXL userspace + UCX bindings, required for RDMA transfers.
#
# The Dynamo-side refit Python code (dynamo.vllm.mx_refit.*) is expected to be
# part of the base image, built from the Dynamo integration branch. This
# Dockerfile only pins the ModelExpress client/NIXL layer; it does not copy a
# runtime source overlay from the NeMo-RL checkout.
#
# Build:
#   docker buildx build --platform linux/arm64 \
#     --build-arg BASE_IMAGE=jwillthomson/dynamo-arm-rl-tokenize-endpoint-91cdc71:latest \
#     --build-arg MX_REF=e0c9447939f0420f6815464b09455724e57d7553 \
#     -t jwillthomson/dynamo-arm-rl-tokenize-endpoint-91cdc71-mx:e0c9447 \
#     -f infra/nrl_k8s/dynamo_mx/Dockerfile .
#
# Push:
#   docker push jwillthomson/dynamo-arm-rl-tokenize-endpoint-91cdc71-mx:e0c9447

ARG BASE_IMAGE=jwillthomson/dynamo-arm-rl-tokenize-endpoint-91cdc71:latest
FROM ${BASE_IMAGE}

# `kavink/nemo_rl_moe` is the v2-refit branch. Pin to a SHA so rebuilds are
# reproducible.
ARG MX_REF=e0c9447939f0420f6815464b09455724e57d7553

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
print('MxRefitWorkerExtension import OK')"

# Default to the same entrypoint as the base — DGD manifests override this.
