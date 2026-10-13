#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start vllm-project/router from the same file-backed backend registry used by
# the existing NeMo RL proxy. This process owns only OpenAI generation traffic;
# the Python proxy remains available on a separate port for refit fan-out.

set -euo pipefail

if (( $# != 7 )); then
  echo "Usage: $0 PORT STATE_DIR GROUP_ID REPLICAS PREFILL_REPLICAS STARTUP_TIMEOUT LOG_FILE" >&2
  exit 2
fi

router_port="$1"
state_dir="$2"
group_id="$3"
replicas="$4"
prefill_replicas="$5"
startup_timeout="$6"
log_file="$7"

: "${EXTERNAL_VLLM_TOOLS_DIR:=/opt/external-vllm-tools}"
: "${VLLM_ROUTER_PYTHON:=python3}"
: "${VLLM_ROUTER_PREFILL_POLICY:=cache_aware}"
: "${VLLM_ROUTER_DECODE_POLICY:=cache_aware}"
: "${VLLM_ROUTER_REQUEST_TIMEOUT_S:=86400}"
: "${VLLM_ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE:=1}"

export EXTERNAL_VLLM_STATE_DIR="${state_dir}"
export EXTERNAL_VLLM_GROUP_ID="${group_id}"
source "${EXTERNAL_VLLM_TOOLS_DIR}/vllm_backend_registry.sh"

deadline=$((SECONDS + startup_timeout))
while true; do
  mapfile -t registry_entries < <(registry_list | awk '$5 == "ready" { print $2 ":" $3 " " $6 }')
  if (( ${#registry_entries[@]} == replicas )); then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for ${replicas} registered vLLM workers" >&2
    exit 1
  fi
  sleep 5
done

worker_urls=()
prefill_urls=()
decode_urls=()
for entry in "${registry_entries[@]}"; do
  read -r endpoint role <<< "${entry}"
  if (( prefill_replicas > 0 )); then
    case "${role}" in
      prefill) prefill_urls+=("http://${endpoint}") ;;
      decode) decode_urls+=("http://${endpoint}") ;;
      *)
        echo "Rust P/D router requires prefill/decode registry roles; got ${role}" >&2
        exit 1
        ;;
    esac
  else
    case "${role}" in
      standard) worker_urls+=("http://${endpoint}") ;;
      *)
        echo "Rust router regular mode requires standard registry roles; got ${role}" >&2
        exit 1
        ;;
    esac
  fi
done
if (( prefill_replicas > 0 )); then
  if (( ${#prefill_urls[@]} != prefill_replicas )); then
    echo "Expected ${prefill_replicas} prefill workers, found ${#prefill_urls[@]}" >&2
    exit 1
  fi
  if (( ${#decode_urls[@]} != replicas - prefill_replicas )); then
    echo "Expected $((replicas - prefill_replicas)) decode workers, found ${#decode_urls[@]}" >&2
    exit 1
  fi
elif (( ${#worker_urls[@]} != replicas )); then
  echo "Expected ${replicas} standard workers, found ${#worker_urls[@]}" >&2
  exit 1
fi

router_python="${VLLM_ROUTER_PYTHON}"
if [[ -n "${VLLM_ROUTER_SITE_PACKAGES:-}" && -n "${VLLM_ROUTER_WHEEL:-}" ]]; then
  echo "Set only one of VLLM_ROUTER_SITE_PACKAGES and VLLM_ROUTER_WHEEL" >&2
  exit 1
fi
if [[ -n "${VLLM_ROUTER_SITE_PACKAGES:-}" ]]; then
  if [[ ! -d "${VLLM_ROUTER_SITE_PACKAGES}" ]]; then
    echo "VLLM_ROUTER_SITE_PACKAGES does not exist: ${VLLM_ROUTER_SITE_PACKAGES}" >&2
    exit 1
  fi
  export PYTHONPATH="${VLLM_ROUTER_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
elif [[ -n "${VLLM_ROUTER_WHEEL:-}" ]]; then
  if [[ ! -f "${VLLM_ROUTER_WHEEL}" ]]; then
    echo "VLLM_ROUTER_WHEEL does not exist: ${VLLM_ROUTER_WHEEL}" >&2
    exit 1
  fi
  router_site=$(mktemp -d /tmp/vllm-router.XXXXXX)
  "${router_python}" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --target "${router_site}" \
    "${VLLM_ROUTER_WHEEL}"
  export PYTHONPATH="${router_site}${PYTHONPATH:+:${PYTHONPATH}}"
fi

router_args=(
  --host 0.0.0.0
  --port "${router_port}"
  --request-timeout-secs "${VLLM_ROUTER_REQUEST_TIMEOUT_S}"
  --worker-startup-timeout-secs "${startup_timeout}"
  --intra-node-data-parallel-size "${VLLM_ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE}"
  --log-level "${VLLM_ROUTER_LOG_LEVEL:-info}"
)

if (( prefill_replicas > 0 )); then
  router_args+=(
    --prefill-policy "${VLLM_ROUTER_PREFILL_POLICY}"
    --decode-policy "${VLLM_ROUTER_DECODE_POLICY}"
    --vllm-pd-disaggregation
  )
  for url in "${prefill_urls[@]}"; do
    router_args+=(--prefill "${url}")
  done
  for url in "${decode_urls[@]}"; do
    router_args+=(--decode "${url}")
  done
  echo "Starting vllm-router with ${#prefill_urls[@]} prefill and ${#decode_urls[@]} decode workers"
else
  router_args+=(--policy "${VLLM_ROUTER_POLICY:-cache_aware}" --worker-urls)
  for url in "${worker_urls[@]}"; do
    router_args+=("${url}")
  done
  echo "Starting vllm-router with ${#worker_urls[@]} standard workers"
fi
exec "${router_python}" -m vllm_router.launch_router "${router_args[@]}" >> "${log_file}" 2>&1
