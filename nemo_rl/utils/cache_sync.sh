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

# Keep JIT compile caches (vLLM, TorchInductor, Triton, FlashInfer, ...) across
# jobs: copy a persistent directory on the shared filesystem to node-local
# storage, and publish what the job compiles back.
#
#   cache_sync.sh seed      <persistent_dir> <local_dir>
#   cache_sync.sh writeback <local_dir> <persistent_dir>
#   cache_sync.sh sidecar   <local_dir> <persistent_dir> [interval_s] [stop_file]
#   cache_sync.sh seed-type <type>
#   cache_sync.sh sidecar-types <interval_s> <stop_file> <type>...
#
# seed copies once; a failed copy only warns (the job compiles what is missing).
# writeback only adds files that are not in <persistent_dir> yet, appeared after
# the seed started or the last successful write-back, and have not changed for
# NRL_COMPILE_CACHE_MIN_AGE_MIN minutes (default 1): existing files are never
# replaced, so a file caught mid-write must never be published. Writers take
# turns on <persistent_dir>.lock (stale after 10 min); a busy lock skips the pass,
# except the last one, which waits up to 2 min. The cached files are content- or
# hash-addressed, so merging entries from several nodes and jobs is safe.
# sidecar runs writeback every interval_s seconds (default 60) and once more after
# stop_file appears.
#
# The <type> modes take their folders from the environment: <type> is copied from
# NRL_COMPILE_CACHE_<TYPE>_DIR (default ${NRL_COMPILE_CACHE_SHARED_DIR}/<type>) to
# ${NRL_COMPILE_CACHE_LOCAL_DIR}/<type>. seed-type seeds once per node under a
# node-local lock, so a launcher can start it in the background and a worker can
# call it again to wait for that copy (NRL_COMPILE_CACHE_FORCE_FRESH=1: start
# empty). sidecar-types writes back each type seeded on this node; for vllm it
# first merges the engines' own folders (<local>/vllm_engines/*) into vllm/vllm.
set -uo pipefail

log() { echo "[cache_sync $(hostname -s)] $*"; }
min_age() { echo "${NRL_COMPILE_CACHE_MIN_AGE_MIN:-1}"; }

seed() {
  local src="$1" dst="$2" rc
  mkdir -p "${dst}" || return 1
  # Dated before the copy, so files compiled while it runs count as new.
  touch "${dst}.seeding"
  if [[ -d "${src}" ]]; then
    # rsync leaves out a file it cannot copy instead of leaving it truncated.
    rsync -a "${src}/" "${dst}/"
    rc=$?
    if (( rc == 0 || rc == 24 )); then
      log "seeded ${dst} from ${src} ($(du -sh "${dst}" | cut -f1))"
    else
      log "WARNING: seeding ${dst} from ${src} failed (rsync exit ${rc}); this node compiles what is missing"
    fi
  fi
  mv "${dst}.seeding" "${dst}.seeded"
}

# writeback <src> <dst> [wait_s]: publish new settled files; wait_s > 0 waits for the lock.
writeback() {
  local src="$1" dst="$2" wait_s="${3:-0}" lock="$2.lock" list mark rc waited=0
  local -a newer=()
  [[ -d "${src}" ]] || return 0
  if [[ -e "${src}.sent" ]]; then
    newer=(-newer "${src}.sent")
  elif [[ -e "${src}.seeded" ]]; then
    newer=(-newer "${src}.seeded")
  fi
  # Files younger than the age limit are left for the next pass, so the "sent"
  # mark is dated at the age limit, not now.
  mark="$(mktemp "${src}.sending.XXXXXX")"
  touch -d "@$(( $(date +%s) - $(min_age) * 60 ))" "${mark}"
  list="$(mktemp)"
  (cd "${src}" && find . -type f -mmin "+$(min_age)" "${newer[@]}") >"${list}"
  if [[ ! -s "${list}" ]]; then
    mv "${mark}" "${src}.sent"
    rm -f "${list}"
    return 0
  fi
  mkdir -p "${dst}"
  until mkdir "${lock}" 2>/dev/null; do
    if [[ -n "$(find "${lock}" -maxdepth 0 -mmin +10 2>/dev/null)" ]]; then
      log "removing stale lock ${lock}"
      rmdir "${lock}" 2>/dev/null
      continue
    fi
    if (( waited >= wait_s )); then
      (( wait_s > 0 )) && log "WARNING: ${lock} busy for ${wait_s}s; $(wc -l <"${list}") new files in ${src} not published"
      rm -f "${mark}" "${list}"
      return 0
    fi
    sleep 5 || break
    (( waited += 5 ))
  done
  rsync -a --ignore-existing --files-from="${list}" "${src}/" "${dst}/"
  rc=$?
  rmdir "${lock}"
  if (( rc == 0 || rc == 24 )); then
    mv "${mark}" "${src}.sent"
  else
    log "WARNING: write-back ${src} -> ${dst} failed (rsync exit ${rc}); retried next pass"
    rm -f "${mark}"
  fi
  rm -f "${list}"
}

sidecar() {
  local src="$1" dst="$2" interval="${3:-60}" stop="${4:-}" last=0
  until (( last )); do
    sleep "${interval}" || return  # e.g. the container is gone
    [[ -n "${stop}" && -e "${stop}" ]] && last=1
    writeback "${src}" "${dst}" "$(( last ? 120 : 0 ))"
  done
}

type_dirs() {  # sets type_shared and type_local for cache type $1
  local var="NRL_COMPILE_CACHE_${1^^}_DIR"
  : "${NRL_COMPILE_CACHE_SHARED_DIR:?NRL_COMPILE_CACHE_SHARED_DIR is required}"
  : "${NRL_COMPILE_CACHE_LOCAL_DIR:?NRL_COMPILE_CACHE_LOCAL_DIR is required}"
  type_shared="${!var:-${NRL_COMPILE_CACHE_SHARED_DIR}/$1}"
  type_local="${NRL_COMPILE_CACHE_LOCAL_DIR}/$1"
}

seed_type() {
  local type_shared type_local
  type_dirs "$1"
  mkdir -p "$(dirname "${type_local}")" || exit 1
  exec 9>"${type_local}.seedlock"
  flock 9 || { log "ERROR: flock on ${type_local}.seedlock failed"; exit 1; }
  if [[ ! -e "${type_local}.seeded" ]]; then
    if [[ "${NRL_COMPILE_CACHE_FORCE_FRESH:-0}" == "1" ]]; then
      mkdir -p "${type_local}" && touch "${type_local}.seeded"
    else
      seed "${type_shared}" "${type_local}"
    fi
  fi
  exec 9>&-
}

# Merge each vLLM engine's own folder into <local>/vllm/vllm. Settled files only,
# and without their times, so the next write-back sees them as new.
merge_vllm_engines() {
  local root="$1" engine list
  for engine in "${root}"/vllm_engines/*/; do
    [[ -d "${engine}" ]] || continue
    list="$(mktemp)"
    (cd "${engine}" && find . -type f -mmin "+$(min_age)") >"${list}"
    if [[ -s "${list}" ]]; then
      mkdir -p "${root}/vllm/vllm"
      rsync -rl --ignore-existing --files-from="${list}" "${engine}" "${root}/vllm/vllm/"
    fi
    rm -f "${list}"
  done
}

sidecar_types() {
  local interval="$1" stop="$2" cache_type last=0 type_shared type_local
  shift 2
  until (( last )); do
    sleep "${interval}" || return  # e.g. the container is gone
    [[ -e "${stop}" ]] && last=1
    for cache_type in "$@"; do
      type_dirs "${cache_type}"
      [[ -e "${type_local}.seeded" ]] || continue
      [[ "${cache_type}" == vllm ]] && merge_vllm_engines "${NRL_COMPILE_CACHE_LOCAL_DIR}"
      writeback "${type_local}" "${type_shared}" "$(( last ? 120 : 0 ))"
    done
  done
}

for tool in rsync flock; do
  command -v "${tool}" >/dev/null || { log "ERROR: ${tool} is required"; exit 1; }
done
cmd="${1:-}"
shift || true
case "${cmd}:$#" in
  seed:2) seed "$@" ;;
  writeback:2) writeback "$@" ;;
  sidecar:2 | sidecar:3 | sidecar:4) sidecar "$@" ;;
  seed-type:1) seed_type "$@" ;;
  sidecar-types:[3-9]) sidecar_types "$@" ;;
  *) echo "usage: cache_sync.sh {seed|writeback|sidecar|seed-type|sidecar-types} ..." >&2; exit 2 ;;
esac
