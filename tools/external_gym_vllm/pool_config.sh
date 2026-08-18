#!/bin/bash
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

# Public shell interface for registering external Gym vLLM pools before sbatch.

_external_vllm_set() {
  local pool="$1" suffix="$2" value="$3"
  printf -v "${pool}_${suffix}" '%s' "${value}"
  export "${pool}_${suffix}"
}

_external_vllm_is_registered() {
  local requested="$1" existing
  local -a existing_pools=()
  read -r -a existing_pools <<< "${EXTERNAL_VLLM_POOLS:-}"
  for existing in "${existing_pools[@]}"; do
    [[ "${existing}" == "${requested}" ]] && return 0
  done
  return 1
}

_external_vllm_append_lines() {
  local pool="$1" suffix="$2"
  shift 2
  local variable_name="${pool}_${suffix}"
  local current_value="${!variable_name-}"
  local new_value
  printf -v new_value '%s\n' "$@"
  new_value="${new_value%$'\n'}"
  if [[ -n "${current_value}" && -n "${new_value}" ]]; then
    new_value="${current_value}"$'\n'"${new_value}"
  elif [[ -n "${current_value}" ]]; then
    new_value="${current_value}"
  fi
  _external_vllm_set "${pool}" "${suffix}" "${new_value}"
}

_external_vllm_require_positive_integer() {
  local field="$1" value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "ERROR: ${field} must be a positive integer (got '${value}')" >&2
    return 2
  fi
}

_external_vllm_require_port() {
  local field="$1" value="$2"
  _external_vllm_require_positive_integer "${field}" "${value}" || return
  if (( value > 65535 )); then
    echo "ERROR: ${field} must be at most 65535 (got '${value}')" >&2
    return 2
  fi
}

_external_vllm_recompute_node_count() {
  local gpus_per_node="${GPUS_PER_NODE:-4}"
  local pool replicas_var tensor_parallel_size_var
  local total=0
  local -a pools=()

  read -r -a pools <<< "${EXTERNAL_VLLM_POOLS:-}"
  for pool in "${pools[@]}"; do
    replicas_var="${pool}_REPLICAS"
    tensor_parallel_size_var="${pool}_TENSOR_PARALLEL_SIZE"
    total=$((total + ${!replicas_var} * ${!tensor_parallel_size_var} / gpus_per_node))
  done
  EXTERNAL_VLLM_NUM_NODES="${total}"
  export EXTERNAL_VLLM_NUM_NODES
}

_external_vllm_require_shared_path() {
  local field="$1" path="$2"
  local allow_model_id="${3:-0}"
  local shared_root="${EXTERNAL_VLLM_SHARED_ROOT:-/lustre}"
  if [[ "${shared_root}" != /* ]]; then
    echo "ERROR: EXTERNAL_VLLM_SHARED_ROOT must be absolute (got '${shared_root}')" >&2
    return 2
  fi
  if [[ "${path}" != /* ]]; then
    if [[ "${allow_model_id}" == "1" ]]; then
      return 0
    fi
    echo "ERROR: ${field} must be an absolute path (got '${path}')" >&2
    return 2
  fi
  if [[ "${path}" != "${shared_root}" && "${path}" != "${shared_root}"/* ]]; then
    echo "ERROR: ${field} must be under ${shared_root} (got '${path}')" >&2
    return 2
  fi
}

register_external_vllm_pool() {
  local pool="${1:?pool name required}"
  shift
  if [[ ! "${pool}" =~ ^[A-Z][A-Z0-9_]*$ ]]; then
    echo "ERROR: invalid external vLLM pool name '${pool}'; use an uppercase shell identifier" >&2
    return 2
  fi
  if _external_vllm_is_registered "${pool}"; then
    echo "ERROR: external vLLM pool '${pool}' is already registered" >&2
    return 2
  fi

  local display_name="${pool}"
  local model=""
  local container=""
  local python=""
  local replicas=""
  local tensor_parallel_size=""
  local lb_port=""
  local url_placeholder=""
  local group_id=""
  local served_model_name="model"
  local vllm_port="8000"
  local startup_timeout="3600"
  local -a shared_paths=()

  while (( $# > 0 )); do
    case "$1" in
      --display-name) display_name="${2:?value required for $1}"; shift 2 ;;
      --model) model="${2:?value required for $1}"; shift 2 ;;
      --container) container="${2:?value required for $1}"; shift 2 ;;
      --python) python="${2:?value required for $1}"; shift 2 ;;
      --replicas) replicas="${2:?value required for $1}"; shift 2 ;;
      --tensor-parallel-size) tensor_parallel_size="${2:?value required for $1}"; shift 2 ;;
      --lb-port) lb_port="${2:?value required for $1}"; shift 2 ;;
      --url-placeholder) url_placeholder="${2:?value required for $1}"; shift 2 ;;
      --group-id) group_id="${2:?value required for $1}"; shift 2 ;;
      --served-model-name) served_model_name="${2:?value required for $1}"; shift 2 ;;
      --vllm-port) vllm_port="${2:?value required for $1}"; shift 2 ;;
      --startup-timeout) startup_timeout="${2:?value required for $1}"; shift 2 ;;
      --shared-path) shared_paths+=("${2:?value required for $1}"); shift 2 ;;
      *)
        echo "ERROR: unknown register_external_vllm_pool option: $1" >&2
        return 2
        ;;
    esac
  done

  local field value
  for field in model container python replicas tensor_parallel_size lb_port url_placeholder; do
    value="${!field}"
    if [[ -z "${value}" ]]; then
      echo "ERROR: ${field//_/-} is required for external vLLM pool ${pool}" >&2
      return 2
    fi
  done

  local gpus_per_node="${GPUS_PER_NODE:-4}"
  _external_vllm_require_positive_integer "GPUS_PER_NODE" "${gpus_per_node}" || return
  _external_vllm_require_positive_integer "${pool}_REPLICAS" "${replicas}" || return
  _external_vllm_require_positive_integer \
    "${pool}_TENSOR_PARALLEL_SIZE" "${tensor_parallel_size}" || return
  _external_vllm_require_port "${pool}_LB_PORT" "${lb_port}" || return
  _external_vllm_require_port "${pool}_VLLM_PORT" "${vllm_port}" || return
  _external_vllm_require_positive_integer \
    "${pool}_STARTUP_TIMEOUT" "${startup_timeout}" || return
  if (( tensor_parallel_size % gpus_per_node != 0 )); then
    echo "ERROR: ${pool}_TENSOR_PARALLEL_SIZE must be divisible by GPUS_PER_NODE=${gpus_per_node}" >&2
    return 2
  fi
  if [[ -n "${group_id}" && ! "${group_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: ${pool}_GROUP_ID may contain only letters, digits, '.', '_', and '-'" >&2
    return 2
  fi
  _external_vllm_require_shared_path "${pool}_MODEL" "${model}" 1 || return
  local shared_path
  for shared_path in "${shared_paths[@]}"; do
    _external_vllm_require_shared_path "${pool}_SHARED_PATHS" "${shared_path}" || return
  done

  local existing existing_lb_var existing_placeholder_var
  local -a existing_pools=()
  read -r -a existing_pools <<< "${EXTERNAL_VLLM_POOLS:-}"
  for existing in "${existing_pools[@]}"; do
    existing_lb_var="${existing}_LB_PORT"
    existing_placeholder_var="${existing}_URL_PLACEHOLDER"
    if [[ "${!existing_lb_var}" == "${lb_port}" ]]; then
      echo "ERROR: external vLLM pools ${existing} and ${pool} use LB port ${lb_port}" >&2
      return 2
    fi
    if [[ "${!existing_placeholder_var}" == "${url_placeholder}" ]]; then
      echo "ERROR: external vLLM pools ${existing} and ${pool} use URL placeholder ${url_placeholder}" >&2
      return 2
    fi
  done

  _external_vllm_set "${pool}" DISPLAY_NAME "${display_name}"
  _external_vllm_set "${pool}" MODEL "${model}"
  _external_vllm_set "${pool}" CONTAINER "${container}"
  _external_vllm_set "${pool}" VLLM_PYTHON "${python}"
  _external_vllm_set "${pool}" REPLICAS "${replicas}"
  _external_vllm_set "${pool}" TENSOR_PARALLEL_SIZE "${tensor_parallel_size}"
  _external_vllm_set "${pool}" LB_PORT "${lb_port}"
  _external_vllm_set "${pool}" URL_PLACEHOLDER "${url_placeholder}"
  _external_vllm_set "${pool}" SERVED_MODEL_NAME "${served_model_name}"
  _external_vllm_set "${pool}" VLLM_PORT "${vllm_port}"
  _external_vllm_set "${pool}" STARTUP_TIMEOUT "${startup_timeout}"
  if [[ -n "${group_id}" ]]; then
    _external_vllm_set "${pool}" GROUP_ID "${group_id}"
  fi
  if (( ${#shared_paths[@]} > 0 )); then
    _external_vllm_append_lines "${pool}" SHARED_PATHS "${shared_paths[@]}"
  else
    _external_vllm_set "${pool}" SHARED_PATHS ""
  fi
  _external_vllm_set "${pool}" ENV_VARS ""
  _external_vllm_set "${pool}" VLLM_ARGS ""

  EXTERNAL_VLLM_POOLS="${EXTERNAL_VLLM_POOLS:+${EXTERNAL_VLLM_POOLS} }${pool}"
  export EXTERNAL_VLLM_POOLS
  _external_vllm_recompute_node_count
}

validate_external_vllm_submission() {
  local command="${1:-${COMMAND:-}}"
  local expected_nodes="${2:-${NUM_EXTERNAL_SERVICE_NODES:-}}"
  local shared_root="${EXTERNAL_VLLM_SHARED_ROOT:-/lustre}"
  local pool placeholder_var path variable_name required_file
  local -a pools=()

  if [[ -z "${command}" ]]; then
    echo "ERROR: external vLLM submission command is empty" >&2
    return 2
  fi
  if [[ -z "${EXTERNAL_VLLM_POOLS:-}" ]]; then
    echo "ERROR: no external vLLM pools are registered" >&2
    return 2
  fi
  if [[ -n "${expected_nodes}" ]]; then
    _external_vllm_require_positive_integer \
      "NUM_EXTERNAL_SERVICE_NODES" "${expected_nodes}" || return
    if (( expected_nodes != EXTERNAL_VLLM_NUM_NODES )); then
      echo "ERROR: NUM_EXTERNAL_SERVICE_NODES=${expected_nodes}, expected ${EXTERNAL_VLLM_NUM_NODES} from registered pools" >&2
      return 2
    fi
  else
    echo "WARNING: NUM_EXTERNAL_SERVICE_NODES is unset; skipping external hetgroup node-count validation" >&2
  fi

  read -r -a pools <<< "${EXTERNAL_VLLM_POOLS}"
  for pool in "${pools[@]}"; do
    placeholder_var="${pool}_URL_PLACEHOLDER"
    if [[ "${command}" != *"${!placeholder_var}"* ]]; then
      echo "ERROR: submission command is missing ${!placeholder_var} for pool ${pool}" >&2
      return 2
    fi
  done

  for variable_name in BASE_LOG_DIR EXTERNAL_VLLM_TOOLS_DIR_HOST; do
    path="${!variable_name-}"
    if [[ -z "${path}" ]]; then
      echo "ERROR: ${variable_name} is required for external vLLM submission" >&2
      return 2
    fi
    _external_vllm_require_shared_path "${variable_name}" "${path}" || return
  done
  for required_file in vllm_backend_registry.sh vllm_pool_lb.py lb_watchdog.sh serve_vllm_on_ray.py; do
    if [[ ! -f "${EXTERNAL_VLLM_TOOLS_DIR_HOST}/${required_file}" ]]; then
      echo "ERROR: missing ${EXTERNAL_VLLM_TOOLS_DIR_HOST}/${required_file}" >&2
      return 2
    fi
  done
  if [[ "${shared_root}" != /* ]]; then
    echo "ERROR: EXTERNAL_VLLM_SHARED_ROOT must be absolute (got '${shared_root}')" >&2
    return 2
  fi
}

external_vllm_pool_env() {
  local pool="${1:?pool name required}"
  shift
  if ! _external_vllm_is_registered "${pool}"; then
    echo "ERROR: external vLLM pool '${pool}' is not registered" >&2
    return 2
  fi
  _external_vllm_append_lines "${pool}" ENV_VARS "$@"
}

external_vllm_pool_args() {
  local pool="${1:?pool name required}"
  shift
  if ! _external_vllm_is_registered "${pool}"; then
    echo "ERROR: external vLLM pool '${pool}' is not registered" >&2
    return 2
  fi
  _external_vllm_append_lines "${pool}" VLLM_ARGS "$@"
}
