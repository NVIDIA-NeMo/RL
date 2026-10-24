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

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=${NEMO_RL_DYNAMO_PROJECT_DIR:-${script_dir}}
repo_root=$(realpath "${script_dir}/../..")
dynamo_venv_dir=${NEMO_RL_DYNAMO_VENV_DIR:-${repo_root}/venvs/dynamo}
dynamo_python_version=${DYNAMO_PYTHON_VERSION:-3.12.11}
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

uv python install "${dynamo_python_version}"
uv venv --python "${dynamo_python_version}" "${dynamo_venv_dir}"
UV_PROJECT_ENVIRONMENT="${dynamo_venv_dir}" uv sync \
  --directory "${project_dir}" \
  --locked \
  --no-dev \
  --no-install-project \
  --link-mode copy

dynamo_python=${dynamo_venv_dir}/bin/python
vllm_version=$("${dynamo_python}" -c \
  'from importlib.metadata import version; print(version("vllm"))')
if [[ "${vllm_version}" != "0.28.0" ]]; then
  echo "Expected vllm==0.28.0 from ai-dynamo[vllm]==1.5.1; got ${vllm_version}" >&2
  exit 1
fi

# Layerwise weight reload must count only the elements each composed loader
# actually writes (vLLM PR #44814). Without it, trailing NemotronH/Mamba2
# parameters such as mixer.D stay unloaded after a refit and corrupt logits.
# vLLM 0.28.0 ships the fix; fail the build if a future pin drops it.
"${dynamo_python}" - <<'PY'
import inspect

import vllm.model_executor.model_loader.reload.meta as meta

if "min(numel," not in inspect.getsource(meta):
    raise SystemExit("vLLM is missing the PR #44814 layerwise-reload fix")
PY

download_dir=$(mktemp -d "${TMPDIR:-/tmp}/nemorl-dynamo-install.XXXXXX")
trap 'rm -rf "${download_dir}"' EXIT

curl --fail --location --retry 3 \
  "https://github.com/etcd-io/etcd/releases/download/${etcd_version}/etcd-${etcd_version}-linux-${target_arch}.tar.gz" \
  --output "${download_dir}/etcd.tgz"
tar -xzf "${download_dir}/etcd.tgz" -C "${download_dir}"
install -m 0755 \
  "${download_dir}/etcd-${etcd_version}-linux-${target_arch}/etcd" \
  "${dynamo_venv_dir}/bin/etcd"

curl --fail --location --retry 3 \
  "https://github.com/nats-io/nats-server/releases/download/${nats_version}/nats-server-${nats_version}-linux-${target_arch}.tar.gz" \
  --output "${download_dir}/nats.tgz"
tar -xzf "${download_dir}/nats.tgz" -C "${download_dir}"
install -m 0755 \
  "${download_dir}/nats-server-${nats_version}-linux-${target_arch}/nats-server" \
  "${dynamo_venv_dir}/bin/nats-server"

"${dynamo_python}" -c \
  'import importlib.metadata as m; assert m.version("ai-dynamo") == "1.5.1"; assert m.version("vllm") == "0.28.0"; assert m.version("nvidia-nccl-cu13") == "2.30.7"'
"${dynamo_venv_dir}/bin/etcd" --version
"${dynamo_venv_dir}/bin/nats-server" --version
