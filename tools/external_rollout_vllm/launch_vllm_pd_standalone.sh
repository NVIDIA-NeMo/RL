#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Standalone vLLM NIXL prefill/decode smoke test for HSG.
#
# The script self-submits a four-node job unless it is already running inside
# its allocation. Each node hosts one TP4 vLLM server: two prefill producers
# and two decode consumers. Once all servers are healthy, a small Python client
# executes vLLM's two-request P/D handshake directly against both pairs.

set -euo pipefail

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")

ACCOUNT="${ACCOUNT:-nemotron_sw_post}"
PARTITION="${PARTITION:-batch}"
QOS="${QOS:-normal}"
TIME_LIMIT="${TIME_LIMIT:-1:00:00}"
JOB_NAME="${JOB_NAME:-vllm-pd-standalone}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/guyueh/container_images/vllm+vllm-openai+v0.29.0-aarch64.sqsh}"
VLLM_EXECUTABLE="${VLLM_EXECUTABLE:-/usr/local/bin/vllm}"
MODEL_PATH="${MODEL_PATH:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-policy}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/${USER:-$(id -un)}/cache/hf_home}"

HTTP_PORT="${HTTP_PORT:-8000}"
NIXL_SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT:-5557}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"

RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/llmservice/users/guyueh/grpo-runs/vllm-pd-standalone-$(date +%Y%m%d)}"

if [[ "${1:-}" != "--inside-allocation" ]]; then
  mkdir -p "${RUN_ROOT}"
  echo "Submitting standalone vLLM P/D test"
  echo "  container: ${CONTAINER}"
  echo "  model:     ${MODEL_PATH}"
  echo "  topology:  2 prefill + 2 decode, TP=${GPUS_PER_NODE}"
  echo "  logs:      ${RUN_ROOT}/slurm-%j.out and ${RUN_ROOT}/<job-id>/"
  submit_command=(sbatch \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --qos="${QOS}" \
    --job-name="${JOB_NAME}" \
    --nodes="${NUM_NODES}" \
    --exclusive \
    --mem=0 \
    --gres="gpu:${GPUS_PER_NODE}" \
    --time="${TIME_LIMIT}" \
    --output="${RUN_ROOT}/slurm-%j.out" \
    --export=ALL \
    "${SCRIPT_PATH}" --inside-allocation)
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN: '
    printf '%q ' "${submit_command[@]}"
    printf '\n'
    exit 0
  fi
  exec "${submit_command[@]}"
fi

if (( NUM_NODES != 4 )); then
  echo "ERROR: this 2P/2D smoke requires NUM_NODES=4" >&2
  exit 2
fi

RUN_DIR="${RUN_ROOT}/${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if (( ${#NODES[@]} != NUM_NODES )); then
  echo "ERROR: expected ${NUM_NODES} allocated nodes, got ${#NODES[@]}" >&2
  exit 2
fi

resolve_node_ip() {
  local node="$1"
  local ip
  ip=$(getent ahostsv4 "${node}" | awk 'NR == 1 { print $1 }')
  if [[ -z "${ip}" ]]; then
    echo "ERROR: could not resolve ${node}" >&2
    return 1
  fi
  printf '%s\n' "${ip}"
}

PREFILL_URLS=()
DECODE_URLS=()
for index in "${!NODES[@]}"; do
  node_ip=$(resolve_node_ip "${NODES[${index}]}")
  if (( index < 2 )); then
    PREFILL_URLS+=("http://${node_ip}:${HTTP_PORT}")
  else
    DECODE_URLS+=("http://${node_ip}:${HTTP_PORT}")
  fi
done

export CONTAINER VLLM_EXECUTABLE MODEL_PATH SERVED_MODEL_NAME HF_HOME
export GPUS_PER_NODE HTTP_PORT NIXL_SIDE_CHANNEL_PORT MAX_MODEL_LEN MAX_NUM_SEQS
export GPU_MEMORY_UTILIZATION

SERVER_BODY=$(cat <<'SERVER_BODY_EOF'
set -euo pipefail

node_ip=$(hostname -I | awk '{ print $1 }')
if [[ -z "${node_ip}" ]]; then
  echo "ERROR: could not determine this server's routable IPv4 address" >&2
  exit 1
fi

export VLLM_NIXL_SIDE_CHANNEL_HOST="${node_ip}"
export VLLM_NIXL_SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT}"
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
export UCX_MODULE_DIR="${PD_UCX_MODULE_DIR:-/usr/local/lib/python3.12/dist-packages/nixl_cu13.libs/ucx}"
export UCX_TLS="${PD_UCX_TLS:-all}"
export UCX_CUDA_IPC_ENABLE_MNNVL="${UCX_CUDA_IPC_ENABLE_MNNVL:-y}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"

if [[ "${PD_ROLE}" == "prefill" ]]; then
  kv_transfer_config='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
else
  kv_transfer_config='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
fi

echo "Starting ${PD_ROLE} server at ${node_ip}:${HTTP_PORT}"
echo "VLLM_NIXL_SIDE_CHANNEL_HOST=${VLLM_NIXL_SIDE_CHANNEL_HOST}"
echo "UCX_NET_DEVICES=${UCX_NET_DEVICES}"
echo "UCX_MODULE_DIR=${UCX_MODULE_DIR}"
echo "UCX_TLS=${UCX_TLS}"
echo "UCX_CUDA_IPC_ENABLE_MNNVL=${UCX_CUDA_IPC_ENABLE_MNNVL}"
if command -v ucx_info >/dev/null 2>&1; then
  ucx_info -d || true
fi

exec "${VLLM_EXECUTABLE}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size "${GPUS_PER_NODE}" \
  --distributed-executor-backend mp \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-cumem-allocator \
  --enable-expert-parallel \
  --attention-backend FLASH_ATTN \
  --mamba-ssm-cache-dtype float32 \
  --kv-transfer-config "${kv_transfer_config}" \
  --port "${HTTP_PORT}"
SERVER_BODY_EOF
)

server_pids=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for pid in "${server_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

for index in "${!NODES[@]}"; do
  if (( index < 2 )); then
    role=prefill
    role_index="${index}"
  else
    role=decode
    role_index="$((index - 2))"
  fi

  echo "Launching ${role} ${role_index} on ${NODES[${index}]}"
  srun \
    --nodes=1 \
    --ntasks=1 \
    --nodelist="${NODES[${index}]}" \
    --gres="gpu:${GPUS_PER_NODE}" \
    --overlap \
    --kill-on-bad-exit=1 \
    --no-container-mount-home \
    --container-image="${CONTAINER}" \
    --container-mounts=/lustre:/lustre \
    --export="ALL,PD_ROLE=${role}" \
    --output="${RUN_DIR}/${role}_${role_index}.log" \
    bash -c "${SERVER_BODY}" &
  server_pids+=("$!")
done

deadline=$((SECONDS + STARTUP_TIMEOUT))
all_urls=("${PREFILL_URLS[@]}" "${DECODE_URLS[@]}")
for url in "${all_urls[@]}"; do
  until curl --fail --silent --max-time 5 "${url}/health" >/dev/null; do
    for pid in "${server_pids[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "ERROR: a vLLM server exited before all replicas became healthy" >&2
        exit 1
      fi
    done
    if (( SECONDS >= deadline )); then
      echo "ERROR: timed out waiting for ${url}/health" >&2
      exit 1
    fi
    sleep 5
  done
  echo "Healthy: ${url}"
done

export PREFILL_URLS_ENV="${PREFILL_URLS[*]}"
export DECODE_URLS_ENV="${DECODE_URLS[*]}"

python3 - <<'PY'
import json
import os
import urllib.error
import urllib.request
import uuid


def post_json(url: str, payload: dict[str, object], request_id: str) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        print(error.read().decode(errors="replace"))
        raise
    if not isinstance(result, dict):
        raise TypeError(f"expected a JSON object from {url}, got {type(result).__name__}")
    return result


prefill_urls = os.environ["PREFILL_URLS_ENV"].split()
decode_urls = os.environ["DECODE_URLS_ENV"].split()
model = os.environ["SERVED_MODEL_NAME"]

for pair_index, (prefill_url, decode_url) in enumerate(
    zip(prefill_urls, decode_urls, strict=True)
):
    request_id = str(uuid.uuid4())
    base_payload: dict[str, object] = {
        "model": model,
        "prompt": "Explain prefill/decode disaggregation in two short sentences.",
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
    }
    prefill_payload = dict(base_payload)
    prefill_payload.update(
        {
            "max_tokens": 1,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            },
        }
    )
    prefill_response = post_json(
        f"{prefill_url}/v1/completions", prefill_payload, request_id
    )
    kv_transfer_params = prefill_response.get("kv_transfer_params")
    if not isinstance(kv_transfer_params, dict):
        raise RuntimeError(
            f"pair {pair_index}: prefill response omitted kv_transfer_params"
        )

    decode_payload = dict(base_payload)
    decode_payload["kv_transfer_params"] = kv_transfer_params
    decode_response = post_json(
        f"{decode_url}/v1/completions", decode_payload, request_id
    )
    choices = decode_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"pair {pair_index}: decode response has no choices")
    print(f"pair {pair_index} passed: {choices[0]}")

print("Standalone 2P/2D NIXL test passed")
PY

touch "${RUN_DIR}/PASSED"
