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

# Batch script for a NeMo RL training job attached to a persistent pool job
# (run_in_allocation.sh, EXTERNAL_VLLM_MODE=serve). At job start it waits until
# the expected pool job publishes READY, resolves COMMAND's URL placeholders and
# the external-service readiness gate from the pool's manifest, then runs ray.sub
# while watching the pool: if the pool job stops, is replaced or goes silent,
# the training job is stopped rather than left training against dead judges.
#
# URLs are resolved here, at job start, never at submission: a pool job that was
# requeued or replaced in between serves from different nodes.
set -euo pipefail

: "${EXTERNAL_VLLM_POOL_DIR:?EXTERNAL_VLLM_POOL_DIR is required}"
: "${EXTERNAL_VLLM_POOL_JOB:?EXTERNAL_VLLM_POOL_JOB is required}"
: "${EXTERNAL_VLLM_POOLS:?EXTERNAL_VLLM_POOLS is required}"
: "${EXTERNAL_VLLM_TOOLS_DIR_HOST:?EXTERNAL_VLLM_TOOLS_DIR_HOST is required}"
: "${COMMAND:?COMMAND is required}"
: "${RAY_SUB:?RAY_SUB is required}"

# shellcheck source=pool_dir.sh
source "${EXTERNAL_VLLM_TOOLS_DIR_HOST}/pool_dir.sh"

# Keep in sync with pool_config.sh and run_in_allocation.sh.
EXTERNAL_SERVICE_READINESS_PLACEHOLDER="__EXTERNAL_SERVICE_READINESS__"

timeout_s="${EXTERNAL_VLLM_ATTACH_TIMEOUT:-3600}"
poll_s="${EXTERNAL_VLLM_ATTACH_POLL_S:-15}"
# Consecutive "finished or unknown" answers from Slurm before the pool job counts
# as gone. Failed queries do not count (see pool_job_gone).
gone_checks="${EXTERNAL_VLLM_POOL_GONE_CHECKS:-3}"
# A heartbeat older than this means the pool job's supervision loop is dead.
heartbeat_stale_s="${EXTERNAL_VLLM_HEARTBEAT_STALE_S:-300}"
watch_s="${EXTERNAL_VLLM_ATTACH_WATCH_S:-30}"
require_lifetime="${EXTERNAL_VLLM_ATTACH_REQUIRE_LIFETIME:-0}"
readiness_poll_s="${EXTERNAL_VLLM_READINESS_POLL_INTERVAL_SECONDS:-5}"
readiness_request_timeout_s="${EXTERNAL_VLLM_READINESS_REQUEST_TIMEOUT_SECONDS:-10}"
for variable in timeout_s gone_checks heartbeat_stale_s; do
  if [[ ! "${!variable}" =~ ^[0-9]+$ ]] || (( ${!variable} <= 0 )); then
    echo "[FATAL] ${variable} must be a positive integer (got '${!variable}')" >&2
    exit 1
  fi
done
for variable in poll_s watch_s readiness_poll_s readiness_request_timeout_s; do
  if [[ ! "${!variable}" =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ || ! "${!variable}" =~ [1-9] ]]; then
    echo "[FATAL] ${variable} must be a positive number of seconds (got '${!variable}')" >&2
    exit 1
  fi
done
if [[ "${COMMAND}" != *"${EXTERNAL_SERVICE_READINESS_PLACEHOLDER}"* ]]; then
  echo "[FATAL] Driver command is missing ${EXTERNAL_SERVICE_READINESS_PLACEHOLDER} for the external-service readiness gate" >&2
  exit 1
fi

pool_dir="${EXTERNAL_VLLM_POOL_DIR}"
pool_job="${EXTERNAL_VLLM_POOL_JOB}"

# pool_job_gone: record one Slurm observation of the pool job; return 0 once Slurm
# has reported it finished or unknown gone_checks times in a row. A failed query
# says nothing about the job and is only logged; the attach timeout bounds an
# outage, so a slurmctld hiccup cannot stop a training job at startup.
consecutive_gone=0
pool_job_gone() {
  local state
  if ! state=$(external_vllm_job_state "${pool_job}"); then
    echo "[WARN] pool job ${pool_job}: squeue failed; retrying" >&2
    return 1
  fi
  if external_vllm_job_is_live "${state}"; then
    consecutive_gone=0
    return 1
  fi
  consecutive_gone=$((consecutive_gone + 1))
  echo "[WARN] pool job ${pool_job}: ${state} (${consecutive_gone}/${gone_checks})" >&2
  (( consecutive_gone >= gone_checks ))
}

# pool_is_ready: READY is present, comes from the expected pool job, and the
# pool's heartbeat is fresh.
pool_is_ready() {
  local generation heartbeat_age
  [[ -f "${pool_dir}/READY" ]] || return 1
  generation=$(cat "${pool_dir}/generation" 2>/dev/null || true)
  [[ "${generation%%-*}" == "${pool_job}" ]] || return 1
  heartbeat_age=$(external_vllm_file_age_s "${pool_dir}/heartbeat")
  [[ -n "${heartbeat_age}" ]] && (( heartbeat_age <= heartbeat_stale_s ))
}

echo "[INFO] $(date -Iseconds) waiting for pool job ${pool_job} (${pool_dir})"
deadline=$((SECONDS + timeout_s))
until pool_is_ready; do
  if pool_job_gone; then
    echo "[FATAL] pool job ${pool_job} is gone" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "[FATAL] pool job ${pool_job} not READY after ${timeout_s}s" >&2
    exit 1
  fi
  sleep "${poll_s}"
done
generation=$(cat "${pool_dir}/generation")
echo "[INFO] $(date -Iseconds) pool generation ${generation} READY since $(cut -d' ' -f2 "${pool_dir}/READY" 2>/dev/null || true)"

# The pool must outlive this job, or it stops serving mid-run.
pool_end=$(cat "${pool_dir}/end_time" 2>/dev/null || true)
pool_end_epoch=""
[[ -n "${pool_end}" ]] && pool_end_epoch=$(date -d "${pool_end}" +%s 2>/dev/null || true)
own_end_epoch=$(external_vllm_slurm_end_epoch "${SLURM_JOB_ID:-}")
if [[ -n "${pool_end_epoch}" && -n "${own_end_epoch}" ]] && (( pool_end_epoch < own_end_epoch )); then
  message="pool job ${pool_job} ends at ${pool_end}, $(( (own_end_epoch - pool_end_epoch) / 60 )) min before this job's time limit"
  if [[ "${require_lifetime}" == 1 ]]; then
    echo "[FATAL] ${message} (EXTERNAL_VLLM_ATTACH_REQUIRE_LIFETIME=1)" >&2
    exit 1
  fi
  echo "[WARN] ${message}; training will stop when the pool does" >&2
fi

# Resolve every pool from the manifest: URL placeholder and readiness gate.
declare -A manifest_base_urls=() manifest_health_urls=() manifest_backends=()
max_startup_timeout=0
while IFS=$'\t' read -r name base_url health_url backends _served_model startup_timeout; do
  [[ -z "${name}" || "${name}" == \#* ]] && continue
  manifest_base_urls["${name}"]="${base_url}"
  manifest_health_urls["${name}"]="${health_url}"
  manifest_backends["${name}"]="${backends}"
  if [[ "${startup_timeout}" =~ ^[0-9]+$ ]] && (( startup_timeout > max_startup_timeout )); then
    max_startup_timeout="${startup_timeout}"
  fi
done < "${pool_dir}/manifest.tsv"
(( max_startup_timeout > 0 )) || max_startup_timeout=3600

readiness='{services:['
separator=""
for pool in ${EXTERNAL_VLLM_POOLS}; do
  placeholder_var="${pool}_URL_PLACEHOLDER"
  placeholder="${!placeholder_var:-}"
  url="${manifest_base_urls[${pool}]:-}"
  if [[ -z "${url}" ]]; then
    echo "[FATAL] pool job ${pool_job} does not serve ${pool} (not in ${pool_dir}/manifest.tsv)" >&2
    exit 1
  fi
  if [[ -z "${placeholder}" || "${COMMAND}" != *"${placeholder}"* ]]; then
    echo "[FATAL] Driver command is missing the URL placeholder for pool ${pool}" >&2
    exit 1
  fi
  COMMAND="${COMMAND//${placeholder}/${url}}"
  readiness+="${separator}{name:${pool},url:\"${manifest_health_urls[${pool}]}\",expected_backends:${manifest_backends[${pool}]}}"
  separator=","
  echo "[INFO] attach: ${pool} -> ${url}"
done
readiness+="],timeout_seconds:${max_startup_timeout},poll_interval_seconds:${readiness_poll_s},request_timeout_seconds:${readiness_request_timeout_s}}"
readiness_arg="++env.nemo_gym.external_service_readiness='${readiness}'"
COMMAND="${COMMAND//${EXTERNAL_SERVICE_READINESS_PLACEHOLDER}/${readiness_arg}}"
export COMMAND

# Must match ray.sub's LOG_DIR: ENDED is the teardown channel between them.
if [[ -n "${BASE_LOG_DIR:-}" ]]; then
  if [[ -n "${SLURM_RESTART_COUNT:-}" ]]; then
    LOG_DIR="${BASE_LOG_DIR}/${SLURM_JOB_ID}-${SLURM_RESTART_COUNT}-logs"
  else
    LOG_DIR="${BASE_LOG_DIR}/${SLURM_JOB_ID}-logs"
  fi
else
  LOG_DIR=""
fi

attached_file="${pool_dir}/attached/${SLURM_JOB_ID:-$$}"
ray_sub_pid=""
cleanup() {
  local status=$?
  trap - EXIT TERM INT
  rm -f "${attached_file}" 2>/dev/null || true
  if [[ -n "${ray_sub_pid}" ]] && kill -0 "${ray_sub_pid}" 2>/dev/null; then
    [[ -n "${LOG_DIR}" ]] && touch "${LOG_DIR}/ENDED" 2>/dev/null
    kill "${ray_sub_pid}" 2>/dev/null || true
    wait "${ray_sub_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

mkdir -p "${pool_dir}/attached"
echo "${generation} $(date -Iseconds)" > "${attached_file}"

bash "${RAY_SUB}" &
ray_sub_pid=$!

# Inline mode stops NeMo RL when a judge exits; do the same here, so a dead or
# replaced pool fails the run instead of silently producing bad rewards. A pool
# that is only degraded (READY withdrawn while a replica relaunches) keeps
# serving and is left alone.
pool_failure() {
  local stopped current heartbeat_age
  stopped=$(cat "${pool_dir}/STOPPED" 2>/dev/null || true)
  if [[ "${stopped%% *}" == "${generation}" ]]; then
    echo "pool job stopped (${stopped})"
    return 0
  fi
  current=$(cat "${pool_dir}/generation" 2>/dev/null || true)
  if [[ "${current}" != "${generation}" ]]; then
    echo "pool generation changed from ${generation} to ${current:-none}; the URLs this job uses are gone"
    return 0
  fi
  heartbeat_age=$(external_vllm_file_age_s "${pool_dir}/heartbeat")
  if [[ -z "${heartbeat_age}" ]] || (( heartbeat_age > heartbeat_stale_s )); then
    echo "pool heartbeat is ${heartbeat_age:-missing}s old (limit ${heartbeat_stale_s}s)"
    return 0
  fi
  return 1
}

while kill -0 "${ray_sub_pid}" 2>/dev/null; do
  if reason=$(pool_failure); then
    echo "[FATAL] $(date -Iseconds) ${reason}; stopping NeMo RL" >&2
    exit 1
  fi
  sleep "${watch_s}"
done

set +e
wait "${ray_sub_pid}"
status=$?
set -e
ray_sub_pid=""
exit "${status}"
