#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install the etcd and nats-server binaries the managed Dynamo services run.
# ai-dynamo itself ships in NeMo RL's vLLM worker venvs (the vllm extra), so
# Dynamo serves the same vLLM release NeMo RL pins; nothing here touches Python.

set -euo pipefail

bin_dir=${NEMO_RL_DYNAMO_BIN_DIR:-/opt/dynamo/bin}
etcd_version=${ETCD_VERSION:-v3.5.21}
nats_version=${NATS_VERSION:-v2.11.6}

target_arch=${TARGETARCH:-}
if [[ -z "${target_arch}" ]]; then
  case "$(uname -m)" in
    x86_64) target_arch=amd64 ;;
    aarch64) target_arch=arm64 ;;
    *)
      echo "Unsupported host architecture: $(uname -m)" >&2
      exit 2
      ;;
  esac
fi
case "${target_arch}" in
  amd64|arm64) ;;
  *)
    echo "Unsupported TARGETARCH: ${target_arch}" >&2
    exit 2
    ;;
esac

mkdir -p "${bin_dir}"
download_dir=$(mktemp -d "${TMPDIR:-/tmp}/nemorl-dynamo-install.XXXXXX")
trap 'rm -rf "${download_dir}"' EXIT

curl --fail --location --retry 3 \
  "https://github.com/etcd-io/etcd/releases/download/${etcd_version}/etcd-${etcd_version}-linux-${target_arch}.tar.gz" \
  --output "${download_dir}/etcd.tgz"
tar -xzf "${download_dir}/etcd.tgz" -C "${download_dir}"
install -m 0755 \
  "${download_dir}/etcd-${etcd_version}-linux-${target_arch}/etcd" \
  "${bin_dir}/etcd"

curl --fail --location --retry 3 \
  "https://github.com/nats-io/nats-server/releases/download/${nats_version}/nats-server-${nats_version}-linux-${target_arch}.tar.gz" \
  --output "${download_dir}/nats.tgz"
tar -xzf "${download_dir}/nats.tgz" -C "${download_dir}"
install -m 0755 \
  "${download_dir}/nats-server-${nats_version}-linux-${target_arch}/nats-server" \
  "${bin_dir}/nats-server"

"${bin_dir}/etcd" --version
"${bin_dir}/nats-server" --version
