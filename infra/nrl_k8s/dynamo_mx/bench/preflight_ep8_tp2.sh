#!/usr/bin/env bash
set -euo pipefail

NS="${NS:-${USER}}"
TRAIN_PREFIX="${TRAIN_PREFIX:-${NS}-rc-moe-q3-30b-ep8-gpu-workers-worker}"
ROLLOUT_PREFIX="${ROLLOUT_PREFIX:-${NS}-dyn-moe-q3-30b-ep8-tp2-0-vllmdecodeworker}"

pods() {
  kubectl get pods -n "${NS}" --no-headers -o custom-columns=NAME:.metadata.name
}

mapfile -t TRAIN_PODS < <(pods | awk -v p="${TRAIN_PREFIX}" 'index($0,p)==1')
mapfile -t ROLLOUT_PODS < <(pods | awk -v p="${ROLLOUT_PREFIX}" 'index($0,p)==1')

if [[ "${#TRAIN_PODS[@]}" -ne 2 ]]; then
  echo "ERROR: expected 2 trainer pods matching ${TRAIN_PREFIX}, found ${#TRAIN_PODS[@]}" >&2
  exit 2
fi
if [[ "${#ROLLOUT_PODS[@]}" -ne 1 ]]; then
  echo "ERROR: expected 1 rollout pod matching ${ROLLOUT_PREFIX}, found ${#ROLLOUT_PODS[@]}" >&2
  exit 3
fi

check_pod() {
  local pod="$1"
  local expected_gpus="$2"
  local gpu_count
  gpu_count="$(kubectl exec -n "${NS}" "${pod}" -- nvidia-smi -L | wc -l)"
  if [[ "${gpu_count}" -ne "${expected_gpus}" ]]; then
    echo "ERROR: ${pod}: expected ${expected_gpus} GPUs, found ${gpu_count}" >&2
    exit 4
  fi
  for nic in rdma0 rdma1 rdma2 rdma3; do
    kubectl exec -n "${NS}" "${pod}" -- test -e "/sys/class/net/${nic}" || {
      echo "ERROR: ${pod}: missing ${nic}" >&2
      exit 5
    }
  done
  kubectl exec -n "${NS}" "${pod}" -- bash -lc \
    'test "${UCX_TLS:-}" = "^tcp" && test "${NIXL_UCX_TLS:-}" = "^tcp"' || {
      echo "ERROR: ${pod}: UCX/NIXL TCP deny-list is not active" >&2
      exit 6
    }
}

for pod in "${TRAIN_PODS[@]}"; do
  check_pod "${pod}" 4
done
check_pod "${ROLLOUT_PODS[0]}" 2

kubectl exec -n "${NS}" "${ROLLOUT_PODS[0]}" -- python3 -c '
from importlib.metadata import version
from modelexpress import register_modelexpress_loaders
from vllm.distributed.weight_transfer import WeightTransferEngineFactory
register_modelexpress_loaders()
from modelexpress.engines.vllm.weight_transfer import register
register()
assert "mx" in WeightTransferEngineFactory._registry
print("modelexpress", version("modelexpress"))
print("nixl", version("nixl-cu13"))
print("native_mx_backend=registered")
'

echo "PASS: EP8 trainer + TP2 rollout GPU, RDMA, transport, version, and backend checks"
