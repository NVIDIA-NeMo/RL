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

# Keep JIT compile caches (vLLM, TorchInductor, Triton, FlashInfer, ...) across
# jobs: copy a persistent directory on the shared filesystem to node-local
# storage, and publish what the job compiles back.
#
#   cache_sync.sh seed      <persistent_dir> <local_dir>
#   cache_sync.sh writeback <local_dir> <persistent_dir>
#   cache_sync.sh sidecar   <local_dir> <persistent_dir> [interval_s] [stop_file]
#   cache_sync.sh seed-once <persistent_dir> <local_dir>
#   cache_sync.sh sidecar-types <local_root> <persistent_root> <interval_s> <stop_file> <type>...
#
# writeback only adds files that are not in <persistent_dir> yet, are newer than
# the seed and have not changed for CACHE_SYNC_MIN_AGE_MIN minutes (default 1):
# existing files are never replaced, so a file caught mid-write must never be
# published. Writers take turns on <persistent_dir>.lock (stale after 10 min).
# The cached files are content- or hash-addressed, so merging entries from
# several nodes and jobs is safe. sidecar runs writeback every interval_s
# seconds (default 60) and once more after stop_file appears.
# seed-once seeds a node once under a node-local lock: a launcher can start it in
# the background and a worker can call it again to wait for that copy
# (CACHE_FORCE_FRESH=1: start empty). sidecar-types writes back
# <local_root>/<type> for each type seeded on this node.
set -uo pipefail

log() { echo "[cache_sync $(hostname -s)] $*"; }

seed() {
  local src="$1" dst="$2" rc
  mkdir -p "${dst}"
  if [[ -d "${src}" ]]; then
    # rsync leaves out a file it cannot copy instead of leaving it truncated.
    rsync -a "${src}/" "${dst}/"
    rc=$?
    (( rc == 0 || rc == 24 )) || log "WARNING: seeding ${dst} incomplete (rsync exit ${rc})"
    log "seeded ${dst} from ${src} ($(du -sh "${dst}" | cut -f1))"
  fi
  touch "${dst}.seeded"
}

writeback() {
  local src="$1" dst="$2" lock="$2.lock" list rc
  local -a newer=()
  [[ -d "${src}" ]] || return 0
  [[ -e "${src}.seeded" ]] && newer=(-newer "${src}.seeded")
  list="$(mktemp)"
  (cd "${src}" && find . -type f -mmin "+${CACHE_SYNC_MIN_AGE_MIN:-1}" "${newer[@]}") >"${list}"
  if [[ -s "${list}" ]] && mkdir -p "${dst}" && mkdir "${lock}" 2>/dev/null; then
    rsync -a --ignore-existing --files-from="${list}" "${src}/" "${dst}/"
    rc=$?
    rmdir "${lock}"
    (( rc == 0 || rc == 24 )) || log "WARNING: write-back ${src} -> ${dst} failed (rsync exit ${rc})"
  elif [[ -n "$(find "${lock}" -maxdepth 0 -mmin +10 2>/dev/null)" ]]; then
    log "removing stale lock ${lock}"
    rmdir "${lock}"
  fi
  rm -f "${list}"
}

sidecar() {
  local src="$1" dst="$2" interval="${3:-60}" stop="${4:-}" last=0
  until (( last )); do
    sleep "${interval}"
    [[ -n "${stop}" && -e "${stop}" ]] && last=1
    writeback "${src}" "${dst}"
  done
}

seed_once() {
  mkdir -p "$(dirname "$2")"
  exec 9>"$2.seedlock"
  flock 9
  if [[ ! -e "$2.seeded" ]]; then
    if [[ "${CACHE_FORCE_FRESH:-0}" == "1" ]]; then
      mkdir -p "$2" && touch "$2.seeded"
    else
      seed "$1" "$2"
    fi
  fi
  exec 9>&-
}

sidecar_types() {
  local root="$1" persistent="$2" interval="$3" stop="$4" cache_type last=0
  shift 4
  until (( last )); do
    sleep "${interval}"
    [[ -e "${stop}" ]] && last=1
    for cache_type in "$@"; do
      [[ -e "${root}/${cache_type}.seeded" ]] && writeback "${root}/${cache_type}" "${persistent}/${cache_type}"
    done
  done
}

command -v rsync >/dev/null || { log "ERROR: rsync is required"; exit 1; }
cmd="${1:-}"
shift || true
case "${cmd}:$#" in
  seed:2) seed "$@" ;;
  writeback:2) writeback "$@" ;;
  sidecar:2 | sidecar:3 | sidecar:4) sidecar "$@" ;;
  seed-once:2) seed_once "$@" ;;
  sidecar-types:[5-9] | sidecar-types:[1-9][0-9]) sidecar_types "$@" ;;
  *) echo "usage: cache_sync.sh {seed|writeback|sidecar|seed-once|sidecar-types} ..." >&2; exit 2 ;;
esac
