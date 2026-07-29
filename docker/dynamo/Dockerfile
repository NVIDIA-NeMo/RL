# syntax=docker/dockerfile:1.7-labs

ARG NEMO_RL_BASE_IMAGE
FROM ${NEMO_RL_BASE_IMAGE}

ARG TARGETARCH
ARG DYNAMO_PYTHON_VERSION=3.12.11
ARG ETCD_VERSION=v3.5.21
ARG NATS_VERSION=v2.11.6

USER root
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

COPY docker/dynamo/pyproject.toml docker/dynamo/uv.lock /opt/dynamo_project/
COPY docker/dynamo/patches /opt/dynamo_patches
COPY docker/dynamo/install.sh /opt/dynamo_install.sh

RUN /opt/dynamo_install.sh

ENV DYNAMO_PYTHON=/opt/dynamo_venv/bin/python

RUN /opt/dynamo_venv/bin/python -c \
      'import importlib.metadata as m; assert m.version("ai-dynamo") == "1.3.0.post1"; assert m.version("vllm") == "0.23.0"' && \
    test -s /opt/dynamo_venv/VLLM_BACKPORTS && \
    grep -Fqx \
      'vllm PR #44814 merge commit c9e5bf813530fb9ce06024e075da0f520b0718c8' \
      /opt/dynamo_venv/VLLM_BACKPORTS && \
    etcd --version && \
    nats-server --version
