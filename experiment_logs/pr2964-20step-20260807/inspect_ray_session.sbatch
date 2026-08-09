#!/bin/bash

set -euo pipefail

test -n "${RAY_SESSION:-}"
test -n "${TARGET_PID:-}"
[[ "${RAY_SESSION}" == /tmp/ray/session_* ]]
test -d "${RAY_SESSION}/logs"

printf 'host=%s\nsession=%s\ntarget_pid=%s\n' \
  "$(hostname)" "${RAY_SESSION}" "${TARGET_PID}"

find "${RAY_SESSION}/logs" -maxdepth 1 -type f \
  \( -name "*${TARGET_PID}*" -o -name '*engine*' -o -name 'raylet.err' \
  -o -name 'raylet.out' -o -name 'runtime_env_agent*' \) \
  -printf '%f %s\n' | sort

grep -RniE \
  "(${TARGET_PID}|SIG(SEGV|ABRT|KILL)|Killed|OOM|out of memory|CUDA error|illegal memory|core dump|SYSTEM_ERROR|connection.*EOF|cancelled)" \
  "${RAY_SESSION}/logs" || true
